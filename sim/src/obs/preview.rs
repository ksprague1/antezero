//! Build previews: outside rounds (shops, packs, blind select), the scoring
//! capacity of the current build against the coming blinds, and what a shop
//! or pack item would add to it.
//!
//! Like the play candidates, everything is estimated on worlds derived from
//! visible state only: the playing cards the player knows (deck, hand and
//! discard pile as one pool) are put in a canonical order and shuffled with a
//! seed taken from the ante and round, a hand is dealt against the blind
//! (its boss effects included) and scored with a reseeded RNG. The best play
//! of each of `PREVIEW_HANDS` sampled hands (over the `PREVIEW_PLAYS` plays
//! with the highest joker-free estimate) gives the capacity; an item's
//! preview repeats this with the item applied to a copy, on the first
//! `PREVIEW_ITEM_HANDS` of the same hands (paired with the base).
//!
//! Jokers are previewed in a canonical order (additive effects before
//! multiplicative ones, then by content: see `virtual_round`), so the
//! preview measures the build, not its current arrangement, and moving
//! jokers around does not change (or recompute) it.

use std::hash::{Hash, Hasher};
use std::{array, iter};

use super::candidates::{kicker_plays, rough_score, shortlist};
use super::digest::{
	SipHasher13, blinds_done, canon_key, card_hash, hand_digest, reads_money, state_key,
};
use super::layout::shown_hand;
use super::{ItemZone, Zone};
use crate::actions::Action;
use crate::calculate_joker::LifecycleCtx;
use crate::card::{AnyCard, Card, CardLike, Kind};
use crate::card_mask::CardMask;
use crate::data::{BlindKind, Edition, HandType, Joker, Spectral};
use crate::enum_sets::HandSet;
use crate::event::Event;
use crate::flow::FlowEvent;
use crate::game::{Areas, BlindSelect, BlindSlot, BlindStatus, Game, Hand, Phase, Scratch, State};
use crate::joker_lifecycle::JokerEvent;
use crate::rng::{self, Key, PseudoRandom};
use crate::shop::VoucherSource;
use crate::{actions, areas, blind, calculate_joker, card_lifecycle, flow, num, shop};

/// Hands a build preview (shops, packs, blind select) samples per blind.
pub const PREVIEW_HANDS: usize = 4;
/// Plays a build preview simulates per hand (the best by a cheap joker-free
/// estimate).
pub const PREVIEW_PLAYS: usize = 6;
/// Recent build previews kept per env (a shop visit returns to the same
/// build after joker moves, rerolls and packs).
pub const PREVIEW_CACHE: usize = 4;

/// Hands (of `PREVIEW_HANDS`) an item is previewed on (the first ones; the
/// next blind is previewed on as many).
pub const PREVIEW_ITEM_HANDS: usize = 1;
/// Token float columns of an item's preview (see [`ItemPreview::to_columns`]).
pub const PREVIEW_TOK: usize = 4;
/// The bound of the previews' log-ratio features (a capacity's
/// [`Capacity::log_ratio`], an item's [`ItemPreview::gain`]): they lie in
/// `-PREVIEW_LOG_MAX..=PREVIEW_LOG_MAX`.
pub const PREVIEW_LOG_MAX: f64 = 4.0;
/// The cap of the previews' capacity-ratio features ([`Capacity::ratio_feat`]).
pub const PREVIEW_RATIO_MAX: f64 = 4.0;

/// Scoring capacity against one blind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Capacity {
	/// Mean over the sampled hands of the best play's score.
	pub mean: f64,
	/// Hands the round would start with (after the blind's effect).
	pub hands: f64,
	pub chips: f64,
	/// Fraction of sampled hands whose best play times `hands` reaches `chips`.
	pub p_clear: f64,
}

impl Capacity {
	/// How many times over the blind is covered.
	#[must_use]
	pub fn ratio(&self) -> f64 {
		self.hands * self.mean / self.chips.max(1.0)
	}

	#[must_use]
	pub fn log_ratio(&self) -> f32 {
		clamped_feat(
			((self.hands * self.mean + 1.0) / self.chips.max(1.0)).ln(),
			-PREVIEW_LOG_MAX,
			PREVIEW_LOG_MAX,
		)
	}

	/// [`Capacity::ratio`] as a feature, capped at [`PREVIEW_RATIO_MAX`].
	#[must_use]
	pub fn ratio_feat(&self) -> f32 {
		num::feat(self.ratio().min(PREVIEW_RATIO_MAX))
	}
}

/// An item's build preview: the build with the item acquired (a joker:
/// sold) against the boss, on the same hands as without it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ItemPreview {
	/// Log of the capacity with the item over without, within
	/// [`PREVIEW_LOG_MAX`].
	pub gain: f32,
	/// The capacity ratio with the item ([`Capacity::ratio_feat`]).
	pub ratio: f32,
	/// The clear fraction with the item.
	pub p_clear: f32,
}

impl ItemPreview {
	/// Its token columns: gain, ratio, clear fraction and a 1 marking a
	/// previewed item (an item without a preview has zeros there).
	#[must_use]
	pub fn to_columns(self) -> [f32; PREVIEW_TOK] {
		[self.gain, self.ratio, self.p_clear, 1.0]
	}

	/// The preview a token's preview columns hold, or `None` if they do not
	/// mark one (see [`ItemPreview::to_columns`]).
	#[must_use]
	pub fn from_columns(columns: [f32; PREVIEW_TOK]) -> Option<ItemPreview> {
		let [gain, ratio, p_clear, mark] = columns;
		(mark == 1.0).then_some(ItemPreview {
			gain,
			ratio,
			p_clear,
		})
	}
}

/// One sampled hand against the boss: its best score, the plays ranked for
/// it (reused by item previews that deal the same hand) and a digest of its
/// cards.
#[derive(Clone, Debug, PartialEq)]
struct Sample {
	best: f64,
	plays: Vec<CardMask>,
	digest: u64,
}

/// The capacity of a build against this ante's boss and against the next
/// blind to play (the same blind when the boss is next), plus what is
/// needed to preview items on the same sampled hands and a cache of the
/// item previews already computed for this build.
#[derive(Clone, Debug, PartialEq)]
pub struct BuildPreview {
	key: u64,
	seeds: [u64; PREVIEW_HANDS],
	pub boss: BlindKind,
	pub next: BlindKind,
	pub cap_boss: Capacity,
	pub cap_next: Capacity,
	samples: Vec<Sample>,
	/// What a hand is worth in a round against the boss over its best play
	/// alone (below 1 against The Eye and The Mouth): item previews, made on
	/// single hands, are put on the same scale.
	round_share: f64,
	items: Vec<(u64, Option<ItemPreview>)>,
}

// A preview renumbers the cards it orders (the game breaks ties by id) into
// ranges above any real card's.
/// The ids of a virtual round's pool of playing cards.
const PREVIEW_CARD_IDS: u32 = 1_000_000;
/// The ids of a virtual round's jokers, in their canonical order.
const PREVIEW_JOKER_IDS: u32 = 2_000_000;
/// The ids of the joker rows `place_copies` tries.
const TRIAL_JOKER_IDS: u32 = 3_000_000;

/// Numbers `cards` in order from `base`.
fn renumber(cards: &mut [impl CardLike], base: u32) {
	for (i, c) in cards.iter_mut().enumerate() {
		c.core_mut().id = base + num::index32(i);
	}
}

fn next_blind(blind_select: &BlindSelect) -> BlindKind {
	BlindSlot::ALL
		.into_iter()
		.find(|&slot| {
			!matches!(
				blind_select.states[slot],
				BlindStatus::Defeated | BlindStatus::Skipped
			)
		})
		.map_or(blind_select.boss, |slot| blind_select.kind(slot))
}

/// `x` clamped to `lo..=hi` as a feature; NaN, which a ratio of two
/// overflowed (infinite) scores in endless mode gives, is 0.
fn clamped_feat(x: f64, lo: f64, hi: f64) -> f32 {
	if x.is_nan() {
		0.0
	} else {
		num::feat(x.clamp(lo, hi))
	}
}

/// Jokers whose scoring effect multiplies (they go last in the canonical
/// order, where they are worth most).
fn multiplies(c: &Card) -> bool {
	use Joker as J;
	c.ab.x_mult > 1.0
		|| c.ab.caino_xmult > 0.0
		|| matches!(
			c.kind,
			Kind::Joker(
				J::Stencil
					| J::LoyaltyCard
					| J::SteelJoker
					| J::Blackboard
					| J::Constellation
					| J::Cavendish
					| J::CardSharp
					| J::Madness
					| J::Vampire
					| J::Hologram
					| J::Baron
					| J::Obelisk
					| J::Photograph
					| J::LuckyCat
					| J::Baseball
					| J::Ancient
					| J::Ramen
					| J::Campfire
					| J::Acrobat
					| J::Throwback
					| J::Bloodstone
					| J::Glass
					| J::FlowerPot
					| J::Idol
					| J::SeeingDouble
					| J::HitTheRoad
					| J::Duo
					| J::Trio
					| J::Family
					| J::Order
					| J::Tribe
					| J::DriversLicense
					| J::Caino
					| J::Triboulet
					| J::Yorick
			)
		)
}

/// Applies the numeric effects of the queued events (hands, discards, hand
/// size, ante) and drops the rest: a copy that previews a round never runs
/// its animations.
fn settle(v: &mut Game) {
	loop {
		let pending = v.events.take_pending();
		if pending.is_empty() {
			break;
		}
		for e in pending {
			match e {
				Event::EaseHands(d) => v.round.hands_left += d,
				Event::EaseDiscard(d) => v.round.discards_left += d,
				Event::HandSize(d) => v.areas.hand.limit.change(d),
				Event::EaseAnte(d) => v.run.ante += d,
				Event::Flow(FlowEvent::AcornOuter) => {
					flow::shuffle_area(&mut v.areas.jokers, &mut v.rng, &Key::from("aajk"));
				}
				// Round-start jokers (see `virtual_round`); disabling a blind queues its own effects.
				Event::Joker(JokerEvent::Burglar) => {
					v.round.hands_left += 3.0;
					v.round.discards_left = 0.0;
				}
				Event::Joker(JokerEvent::ChicotOuter) => blind::disable_blind(v),
				_ => {}
			}
		}
	}
}

/// Gathers every playing card into the deck, face down, in the canonical
/// order shuffled with `seed`.
fn pool_into_deck(areas: &mut Areas, seed: u64) {
	areas.deck.append(&mut areas.hand.cards);
	areas.deck.append(&mut areas.play);
	areas.deck.append(&mut areas.discard);
	for c in &mut areas.deck {
		c.core.face_down = true;
		c.wheel_flipped = false;
		c.forced_selection = false;
	}
	areas.deck.sort_by_key(canon_key);
	renumber(&mut areas.deck, PREVIEW_CARD_IDS);
	rng::splitmix_shuffle(&mut areas.deck, seed);
}

/// `v` (a copy of a game) at the first decision of a round against `blind`: the
/// pool of playing cards in a canonical order shuffled with `seed`, the
/// hand dealt (face up: it is a sample), the jokers in the canonical order
/// with ids renumbered (the game breaks ties by id), the blind set with its
/// effects (hands, discards, hand size, debuffs, flipped jokers, Cerulean
/// Bell, Crimson Heart), the RNG reseeded from `seed`. Depends on visible
/// state only.
#[must_use]
pub fn virtual_round(mut v: Game, blind: BlindKind, seed: u64) -> Game {
	v.rng = PseudoRandom::new_untraced(&format!("BP{seed:X}"));
	// Leave the shop or pack and its pending animation behind.
	v.events.clear();
	v.areas.consumed.clear();
	v.pack.open = None;
	v.areas.pack_cards.clear();
	v.shop.open = false;
	v.flow.state = State::SelectingHand;
	v.flow.state_complete = true;
	v.flow.phase = Phase::Round;
	v.flow.end_round_pending = false;
	v.flow.scored = false;
	flow::reset_round_counters(&mut v.round, &mut v.hands);
	pool_into_deck(&mut v.areas, seed);
	// Jokers in a canonical order that ignores the player's (cached previews
	// are keyed by the joker multiset): additive before multiplicative, a
	// Polychrome last among the additive and a Holo first among the
	// multiplicative (where their editions are worth most), then by content.
	v.areas.jokers.sort_by_cached_key(|c| {
		let m = multiplies(c);
		(
			m,
			if m {
				c.core.edition != Some(Edition::Holo)
			} else {
				c.core.edition == Some(Edition::Polychrome)
			},
			card_hash(c),
		)
	});
	renumber(&mut v.areas.jokers, PREVIEW_JOKER_IDS);
	v.blind_select.states[BlindSlot::of(blind)] = BlindStatus::Current;
	v.blind_select.selected = Some(blind);
	v.round.facing_blind = true;
	blind::set_blind(&mut v, Some(blind));
	settle(&mut v);
	// The round-start joker effects that are certain: Burglar trades the
	// discards for three hands, Chicot disables the boss. (The others
	// destroy or create cards at random and are left out.) A copy joker
	// repeats Burglar when it stands by it as the blind is selected, which
	// is how a player has them then; Chicot cannot be copied.
	for i in 0..v.areas.jokers.len() {
		if matches!(
			v.areas.jokers[i].kind,
			Kind::Joker(Joker::Burglar | Joker::Chicot)
		) {
			calculate_joker::calc_joker(&mut v, i, LifecycleCtx::SettingBlind {
				boss: blind.is_boss(),
			});
		}
	}
	settle(&mut v);
	if v.areas
		.jokers
		.iter()
		.any(|c| c.kind == Kind::Joker(Joker::Burglar) && !c.core.debuff)
	{
		v.round.hands_left += 3.0
			* num::float(
				v.areas
					.jokers
					.iter()
					.filter(|c| {
						matches!(c.kind, Kind::Joker(Joker::Blueprint | Joker::Brainstorm))
							&& !c.core.debuff
					})
					.count(),
			);
	}
	let n = num::trunc(v.areas.hand.limit.limit.max(0.0)).min(v.areas.deck.len());
	let top = v.areas.deck.len() - n;
	for mut c in v.areas.deck.drain(top..).rev() {
		c.core.face_down = false;
		v.areas.hand.cards.push(c);
	}
	v.areas.hand.sort();
	blind::drawn_to_hand(&mut v);
	place_copies(&mut v);
	v
}

/// Moves the copy jokers of a dealt round to where its hand scores most (a
/// player arranges them before playing; by content alone they copied a
/// neighbour picked by a hash): Brainstorm gets the best joker to lead the
/// row (and stays put or goes last), then each Blueprint the best joker to
/// stand before, each tried on
/// the hand's two best plays. Face-down jokers stay as dealt (their order is
/// hidden).
fn place_copies(v: &mut Game) {
	let is_copy = |c: &Card| matches!(c.kind, Kind::Joker(Joker::Blueprint | Joker::Brainstorm));
	if !v.areas.jokers.iter().any(is_copy)
		|| v.areas.jokers.len() < 2
		|| v.areas.jokers.iter().any(|c| c.core.face_down)
	{
		return;
	}
	let mut scratch = Scratch::default();
	let mut plays: Vec<(f64, CardMask)> = rank_plays(v)
		.into_iter()
		.filter_map(|m| {
			let score = scratch.copy(v, &v.rng).preview_play(m).ok()?;
			Some((score, m))
		})
		.collect();
	plays.sort_by(|a, b| b.0.total_cmp(&a.0));
	plays.truncate(2);
	// Ids name the jokers while they move; the game breaks ties by id, so a row is scored (and left) renumbered.
	let ids: Vec<(u32, Kind)> = v.areas.jokers.iter().map(|c| (c.core.id, c.kind)).collect();
	let score = |scratch: &mut Scratch, v: &Game, row: &[Card]| {
		plays
			.iter()
			.filter_map(|&(_, m)| {
				let copy = scratch.copy(v, &v.rng);
				copy.areas.jokers.clone_from_slice(row);
				renumber(&mut copy.areas.jokers, TRIAL_JOKER_IDS);
				copy.preview_play(m).ok()
			})
			.fold(0.0, f64::max)
	};
	let pos = |row: &[Card], id: u32| {
		areas::position_of(row, id).expect("every order holds the same jokers")
	};
	let mut best = score(&mut scratch, v, &v.areas.jokers);
	let mut keep = |row: &[Card], v: &mut Game| {
		let s = score(&mut scratch, v, row);
		if s > best {
			best = s;
			v.areas.jokers.clone_from_slice(row);
		}
	};
	// The row being tried, refreshed from the current one each time.
	let mut row: Vec<Card> = Vec::with_capacity(v.areas.jokers.len());
	if ids.iter().any(|x| x.1 == Kind::Joker(Joker::Brainstorm)) {
		for &(t, k) in &ids {
			if !matches!(k, Kind::Joker(Joker::Blueprint | Joker::Brainstorm)) {
				// Its own slot matters too: a copied multiplier is worth most at the end of the row.
				for last in [false, true] {
					row.clone_from(&v.areas.jokers);
					let c = row.remove(pos(&row, t));
					row.insert(0, c);
					if last {
						let i = row
							.iter()
							.position(|c| c.kind == Kind::Joker(Joker::Brainstorm))
							.expect("this branch runs only when a Brainstorm is owned");
						let c = row.remove(i);
						row.push(c);
					}
					keep(&row, v);
				}
			}
		}
	}
	for &(b, _) in ids.iter().filter(|x| x.1 == Kind::Joker(Joker::Blueprint)) {
		for &(t, _) in ids.iter().filter(|x| x.0 != b) {
			row.clone_from(&v.areas.jokers);
			let c = row.remove(pos(&row, b));
			let at = pos(&row, t);
			row.insert(at, c);
			keep(&row, v);
		}
	}
	renumber(&mut v.areas.jokers, PREVIEW_JOKER_IDS);
}

fn rank_plays(v: &Game) -> Vec<CardMask> {
	let flags = v.areas.eval_flags();
	let nh = shown_hand(&v.areas.hand.cards).len();
	let forced = actions::forced_mask(&v.areas.hand.cards);
	// The Psychic scores any play of fewer than five cards 0 (skipped, unless
	// the hand has fewer than five).
	let min = if v.blind.active(BlindKind::Psychic) && nh >= 5 {
		5
	} else {
		1
	};
	let mut plays = Vec::new();
	CardMask::for_each_subset(nh, Hand::HIGHLIGHT_LIMIT, |m| {
		if m.contains_all(forced) && m.fits(nh) && m.len() >= min {
			plays.push(rough_score(&v.areas, &v.blind, &v.hands, m, flags));
		}
	});
	let mut out = shortlist(plays, PREVIEW_PLAYS);
	for m in kicker_plays(&v.areas, &out, nh) {
		if !out.contains(&m) && v.check(Action::Play(m)).is_ok() {
			out.push(m);
		}
	}
	out
}

fn best_of(scratch: &mut Scratch, v: &Game, plays: &[CardMask]) -> f64 {
	plays
		.iter()
		.filter_map(|&m| scratch.copy(v, &v.rng).preview_play(m).ok())
		.fold(0.0, f64::max)
}

#[must_use]
pub fn best_play(v: &Game) -> f64 {
	best_of(&mut Scratch::default(), v, &rank_plays(v))
}

#[must_use]
pub fn capacity(g: &Game, blind: BlindKind, seeds: &[u64]) -> Capacity {
	capacity_samples(g, blind, seeds).0
}

fn capacity_samples(g: &Game, blind: BlindKind, seeds: &[u64]) -> (Capacity, Vec<Sample>) {
	let (mut hands, mut chips) = (0.0, 0.0);
	// The Eye and The Mouth are previewed on whole rounds of the sampled hands,
	// which are kept; otherwise one copy of the game is refreshed for each.
	let whole_rounds = matches!(blind, BlindKind::Eye | BlindKind::Mouth);
	let mut samples = Vec::with_capacity(seeds.len());
	let mut rounds = Vec::with_capacity(if whole_rounds { seeds.len() } else { 0 });
	let mut spare: Option<Game> = None;
	let mut scratch = Scratch::default();
	for &s in seeds {
		let copy = if let Some(mut v) = spare.take() {
			v.copy_from(g, &g.rng);
			v
		} else {
			g.clone()
		};
		let v = virtual_round(copy, blind, s);
		let plays = rank_plays(&v);
		let best = best_of(&mut scratch, &v, &plays);
		hands = v.round.hands_left;
		chips = v.blind.chips;
		samples.push(Sample {
			best,
			plays,
			digest: hand_digest(&v.areas, &v.blind, &v.hands),
		});
		if whole_rounds {
			rounds.push(v);
		} else {
			spare = Some(v);
		}
	}
	let n = num::float(seeds.len().max(1));
	let per_round = num::trunc(hands.max(1.0));
	if !(whole_rounds && per_round > 1 && seeds.len() > 1 && !rounds[0].blind.disabled) {
		let clear = samples.iter().filter(|s| hands * s.best >= chips).count();
		return (
			Capacity {
				mean: samples.iter().map(|s| s.best).sum::<f64>() / n,
				hands,
				chips,
				p_clear: num::float(clear) / n,
			},
			samples,
		);
	}
	// Against The Eye and The Mouth a hand is not worth its best play: the
	// sampled hands are played as rounds of the round's own number of hands
	// (dealt again from the first when it has more hands than were sampled),
	// and a round clears or not as a whole.
	let n_sampled = rounds.len();
	let mut deals = rounds;
	for i in n_sampled..per_round {
		let again = deals[i % n_sampled].clone();
		deals.push(again);
	}
	let (mut total, mut played, mut clear, mut n_rounds) = (0.0, 0usize, 0usize, 0usize);
	for round in deals.chunks_mut(per_round) {
		let sum = if blind == BlindKind::Eye {
			eye_round(&mut scratch, round)
		} else {
			mouth_round(&mut scratch, round)
		};
		total += sum;
		played += round.len();
		n_rounds += 1;
		clear += usize::from(sum * hands / num::float(round.len()) >= chips); // a short last round counts at full length
	}
	(
		Capacity {
			mean: total / num::float(played),
			hands,
			chips,
			p_clear: num::float(clear) / num::float(n_rounds),
		},
		samples,
	)
}

fn best_typed(scratch: &mut Scratch, v: &Game) -> (f64, Option<HandType>) {
	let flags = v.areas.eval_flags();
	let mut best = (0.0, None);
	for m in rank_plays(v) {
		if let Ok(s) = scratch.copy(v, &v.rng).preview_play(m)
			&& s > best.0
		{
			best = (
				s,
				rough_score(&v.areas, &v.blind, &v.hands, m, flags)
					.core
					.map(|(hand, _)| hand),
			);
		}
	}
	best
}

/// The Eye: no hand type scores twice in a round, so each sampled hand
/// plays its best type that the earlier ones left. Returns the round's score.
fn eye_round(scratch: &mut Scratch, rounds: &mut [Game]) -> f64 {
	let mut used = HandSet::default();
	let mut total = 0.0;
	for v in rounds {
		v.blind.eye_hands = used;
		let (s, t) = best_typed(scratch, v);
		if let Some(t) = t {
			used.insert(t);
		}
		total += s;
	}
	total
}

/// The Mouth: only the first hand's type scores all round. Each sampled
/// hand's best type is tried as that type on every hand; returns the best
/// round's score.
fn mouth_round(scratch: &mut Scratch, rounds: &mut [Game]) -> f64 {
	let mut types: Vec<HandType> = rounds
		.iter()
		.filter_map(|v| best_typed(scratch, v).1)
		.collect();
	types.sort_unstable();
	types.dedup();
	let mut best = 0.0;
	for t in types {
		let mut total = 0.0;
		for v in rounds.iter_mut() {
			v.blind.only_hand = Some(t);
			total += best_typed(scratch, v).0;
		}
		if total > best {
			best = total;
		}
	}
	best
}

/// Makes the preview of `g` the front of `cache` (this env's recent
/// previews): a cached preview of the same build is moved there and reused
/// (with its item previews), otherwise a new one is computed and inserted
/// (the cache keeps `PREVIEW_CACHE` entries). Returns that preview, or
/// `None` outside shops, packs and blind select (nothing is previewed
/// there).
pub fn build_preview<'cache>(
	g: &Game, cache: &'cache mut Vec<BuildPreview>,
) -> Option<&'cache mut BuildPreview> {
	if !matches!(g.flow.phase, Phase::Shop | Phase::Pack | Phase::BlindSelect) {
		return None;
	}
	let key = state_key(g);
	if let Some(i) = cache.iter().position(|p| p.key == key) {
		cache[..=i].rotate_right(1);
		return cache.first_mut();
	}
	// The boss and the hands played so far (both visible, both fixed while a
	// shop is open) make the sampled hands differ between games: seeded from
	// the ante and round alone, every game with the same cards previewed its
	// items on the same hand.
	let base = {
		let mut h = SipHasher13::new();
		(
			g.run.ante,
			g.run.round,
			blinds_done(g.blind_select.states),
			g.blind_select.boss,
		)
			.hash(&mut h);
		for l in &g.hands {
			l.played.to_bits().hash(&mut h);
		}
		h.finish()
	};
	let seeds: [u64; PREVIEW_HANDS] =
		array::from_fn(|k| rng::splitmix64(base ^ (num::index64(k) + 1)));
	let boss = g.blind_select.boss;
	let next = next_blind(&g.blind_select);
	let (cap_boss, samples) = capacity_samples(g, boss, &seeds);
	let cap_next = if next == boss {
		cap_boss
	} else {
		capacity(g, next, &seeds[..PREVIEW_ITEM_HANDS])
	};
	let free = samples.iter().map(|s| s.best).sum::<f64>() / num::float(samples.len());
	let round_share = if free > 0.0 {
		cap_boss.mean / free
	} else {
		1.0
	};
	cache.insert(0, BuildPreview {
		key,
		seeds,
		boss,
		next,
		cap_boss,
		cap_next,
		samples,
		round_share,
		items: Vec::new(),
	});
	cache.truncate(PREVIEW_CACHE);
	cache.first_mut()
}

/// The reaction of every `j` owned to what a preview just did: the growth
/// that comes with using a planet (Constellation) or selling a card (Campfire).
fn react(v: &mut Game, j: Joker, ctx: LifecycleCtx) {
	for i in 0..v.areas.jokers.len() {
		if v.areas.jokers[i].kind == Kind::Joker(j) {
			calculate_joker::calc_joker(v, i, ctx);
		}
	}
}

/// Adds a bought, picked or redeemed card to the build: jokers join the
/// joker area, planets and Black Hole are used, vouchers are redeemed. Other
/// consumables are not previewed (`None`).
fn acquire(v: &mut Game, mut c: Card) -> Option<()> {
	match c.kind {
		Kind::Joker(_) => {
			card_lifecycle::add_to_deck(v, &mut c);
			v.areas.jokers.push(c);
		}
		Kind::Planet(p) => {
			v.hands.level_up(p.hand(), 1.0);
			react(v, Joker::Constellation, LifecycleCtx::UsingConsumeable {
				kind: c.kind,
				hanged_glass: 0,
			});
		}
		Kind::Spectral(Spectral::BlackHole) => {
			for h in HandType::ALL {
				v.hands.level_up(h, 1.0);
			}
		}
		Kind::Voucher(vc) => {
			v.history.used_vouchers.insert(vc);
			shop::apply_voucher(&mut v.round, &mut v.events, vc, VoucherSource::Redeemed);
		}
		_ => return None,
	}
	Some(())
}

/// The hash of the item at `index` of `zone`, and whether its preview reads
/// the money; `None` if there is no item or it is a playing card, which is
/// never previewed.
fn item_digest(areas: &Areas, zone: ItemZone, index: usize) -> Option<(u64, bool)> {
	let of_card = |c: &Card| (card_hash(c), reads_money(c));
	let of_any = |c: &AnyCard| match c {
		AnyCard::Playing(_) => None,
		AnyCard::Card(c) => Some(of_card(c)),
	};
	match zone {
		ItemZone::ShopCard => areas.shop_jokers.get(index).and_then(of_any),
		ItemZone::ShopVoucher => areas.shop_vouchers.get(index).map(of_card),
		ItemZone::Pack => areas.pack_cards.get(index).and_then(of_any),
		ItemZone::Joker => areas.jokers.get(index).map(of_card),
		ItemZone::Consumable => areas.consumables.get(index).map(of_card),
	}
}

/// A copy of `g` with the item at `index` of `zone` acquired (shop card:
/// bought, consumables used at once; pack card: picked; voucher: redeemed;
/// consumable: used; joker: sold), money effects included, or `None` if the
/// item is not previewed.
fn with_item(g: &Game, zone: ItemZone, index: usize) -> Option<Game> {
	let mut v = g.clone();
	match zone {
		ItemZone::ShopCard => {
			let AnyCard::Card(c) = v.areas.shop_jokers.remove(index) else {
				return None;
			};
			v.run.dollars -= c.core.cost;
			acquire(&mut v, c)?;
		}
		ItemZone::Pack => {
			let AnyCard::Card(c) = v.areas.pack_cards.remove(index) else {
				return None;
			};
			acquire(&mut v, c)?;
		}
		ItemZone::Consumable => {
			let c = v.areas.consumables.remove(index);
			acquire(&mut v, c)?;
		}
		ItemZone::ShopVoucher => {
			let c = v.areas.shop_vouchers.remove(index);
			v.run.dollars -= c.core.cost;
			acquire(&mut v, c)?;
		}
		ItemZone::Joker => {
			let mut c = v.areas.jokers.remove(index);
			card_lifecycle::remove_from_deck(
				&mut v.areas.limits,
				&mut v.run,
				&mut v.round,
				&mut v.shop,
				&mut v.events,
				&mut c,
			);
			v.run.dollars += c.core.sell_cost;
			react(&mut v, Joker::Campfire, LifecycleCtx::SellingCard);
		}
	}
	settle(&mut v);
	Some(v)
}

impl BuildPreview {
	/// Token features of the item at `index` of `zone` (shop cards, pack
	/// cards, vouchers, consumables, jokers): log of the capacity with the
	/// item acquired (jokers: sold) over without, against the boss on the
	/// same hands; the capacity ratio and clear fraction with it; and a 1
	/// marking a previewed item. `None` for items without a preview (their
	/// columns stay zero). Results are cached per item.
	pub fn item_feats(&mut self, g: &Game, zone: ItemZone, index: usize) -> Option<ItemPreview> {
		debug_assert_eq!(
			state_key(g),
			self.key,
			"item_feats needs a state of the build this preview was made for"
		);
		let (card_hash, money) = item_digest(&g.areas, zone, index)?;
		let item_key = {
			let mut hasher = SipHasher13::new();
			// A money joker's own preview reads the money (`state_key` leaves it out).
			(
				Zone::from(zone).id(),
				index,
				card_hash,
				money.then(|| g.run.dollars.to_bits()),
			)
				.hash(&mut hasher);
			hasher.finish()
		};
		if let Some((_, feats)) = self.items.iter().find(|(k, _)| *k == item_key) {
			return *feats;
		}
		let Some(after) = with_item(g, zone, index) else {
			self.items.push((item_key, None));
			return None;
		};
		let changes_ante = after.run.ante != g.run.ante;
		let n = PREVIEW_ITEM_HANDS.min(self.samples.len());
		let (mut sum, mut base, mut clear, mut hands, mut chips, mut share) = (
			0.0,
			0.0,
			0usize,
			self.cap_boss.hands,
			self.cap_boss.chips,
			1.0,
		);
		let mut scratch = Scratch::default();
		// The item's game for each hand, moved into the last one.
		let worlds = iter::repeat_n(after, n).zip(self.seeds.iter().zip(&self.samples));
		for (game, (&seed, sample)) in worlds {
			let world = virtual_round(game, self.boss, seed);
			let best = if hand_digest(&world.areas, &world.blind, &world.hands) == sample.digest {
				best_of(&mut scratch, &world, &sample.plays)
			} else {
				best_of(&mut scratch, &world, &rank_plays(&world))
			};
			// The boss's rule on the round, unless the item disables the boss.
			share = if world.blind.disabled {
				1.0
			} else {
				self.round_share
			};
			hands = world.round.hands_left;
			chips = world.blind.chips;
			sum += best;
			base += sample.best;
			clear += usize::from(hands * best * share >= chips);
		}
		let n = num::float(n.max(1));
		let with = Capacity {
			mean: sum / n * share,
			hands,
			chips,
			p_clear: num::float(clear) / n,
		};
		// The change of capacity (hands x score), so Grabber and Troubadour count; over the target when the item
		// moves this boss's own target (Chicot). Hieroglyph and Petroglyph lower the ante instead: the run's last
		// boss is as far as before, so their gain is what they cost (the ratio column shows the nearer target).
		let target = if changes_ante {
			1.0
		} else {
			self.cap_boss.chips.max(1.0) / chips.max(1.0)
		};
		let feats = ItemPreview {
			gain: clamped_feat(
				((with.hands * sum / n + 1.0) / (self.cap_boss.hands * base / n + 1.0) * target)
					.ln(),
				-PREVIEW_LOG_MAX,
				PREVIEW_LOG_MAX,
			),
			ratio: with.ratio_feat(),
			p_clear: num::feat(with.p_clear),
		};
		self.items.push((item_key, Some(feats)));
		Some(feats)
	}
}
