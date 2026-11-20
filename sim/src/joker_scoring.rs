//! Joker contexts of hand scoring: before, individual cards, repetitions,
//! the joker pass (`main` / other jokers), after, and card destruction.
//!
//! `Card:calculate_joker` is an if-chain over the joker's name, so only
//! the checks of the joker at hand ever run: here a joker's checks for a
//! context are its `match` arms, in the chain's order.

use crate::blind::CardDebuff;
use crate::calculate_joker::{CardArea, Effect, Scoring, ScoringCtx};
use crate::card::{Ability, Card, Kind};
use crate::data::{Enhancement, HandType, Joker, Rank, Rarity, Suit};
use crate::enum_sets::SuitSet;
use crate::event::Event;
use crate::event_queue::EventQueue;
use crate::game::{Areas, Blind, CardSet, Game, HandLevels};
use crate::joker_lifecycle::JokerEvent;
use crate::playing_card::{PlayingCard, SuitTest};
use crate::rng::{Key, PseudoRandom};
use crate::round::Round;
use crate::run::Run;
use crate::{calculate_joker, card_lifecycle, flow, num, rng};

/// One `calculate_joker` call of hand scoring: the joker whose logic runs,
/// whether a copy joker runs it, and the hand being scored.
#[derive(Clone, Copy)]
struct Call<'sc> {
	/// The joker's index in the joker area.
	index: usize,
	joker: Joker,
	/// A Blueprint or Brainstorm runs the logic of the joker it copies.
	blueprinted: bool,
	/// The hand being scored.
	scoring: &'sc Scoring,
}

/// `copier` is the Blueprint or Brainstorm the joker's logic runs for, if
/// it is being copied.
pub(crate) fn calculate(
	g: &mut Game, ji: usize, j: Joker, ctx: ScoringCtx, sc: &Scoring, copier: Option<usize>,
) -> Option<Effect> {
	use Joker as J;
	let call = Call {
		index: ji,
		joker: j,
		blueprinted: copier.is_some(),
		scoring: sc,
	};
	match ctx {
		ScoringCtx::DestroyingCard { play_idx, full_len }
			if j == J::SixthSense
				&& !call.blueprinted
				&& full_len == 1
				&& g.areas.play[play_idx].has_rank(Rank::Six)
				&& g.round.hands_played == 0.0 =>
		{
			if g.areas.consumable_room_after_queued() {
				calculate_joker::queue_consumable(
					&mut g.areas,
					&mut g.events,
					CardSet::Spectral,
					"sixth",
				);
				g.events.status_extra();
			}
			Some(Effect {
				remove: true,
				..Effect::NONE
			})
		}
		ScoringCtx::DebuffedHand if j == J::Matador && g.blind.triggered => Some(
			calculate_joker::ease_and_pay(&mut g.run, &mut g.events, 8.0),
		),
		ScoringCtx::Individual {
			area: CardArea::Play,
			other,
		} => individual_played(
			&mut g.areas,
			&mut g.run,
			&mut g.rng,
			&g.round,
			&mut g.events,
			call,
			other,
		),
		ScoringCtx::Individual {
			area: CardArea::Hand,
			other,
		} => individual_held(&g.areas, &mut g.run, &mut g.rng, &mut g.events, j, other),
		ScoringCtx::Repetition {
			area: CardArea::Play,
			other,
			..
		} => {
			let oc = &g.areas.play[other];
			let repetitions = match j {
				J::SockAndBuskin if oc.is_face(g.areas.pareidolia()) => Some(1),
				J::HangingChad if sc.scoring().next() == Some(other) => Some(2),
				J::Dusk if g.round.hands_left == 0.0 => Some(1),
				J::Selzer => Some(1),
				J::Hack if oc.effective_rank().is_some_and(|r| r <= Rank::Five) => Some(1),
				_ => None,
			};
			repetitions.map(|repetitions| Effect {
				repetitions,
				..Effect::NONE
			})
		}
		ScoringCtx::Repetition {
			area: CardArea::Hand,
			effects: true,
			..
		} if j == J::Mime => Some(Effect {
			repetitions: 1,
			..Effect::NONE
		}),
		ScoringCtx::OtherJoker { other: Some(other) }
			if j == J::Baseball
				&& other != ji
				&& g.areas.jokers[other]
					.joker()
					.is_some_and(|oj| oj.rarity() == Rarity::Uncommon) =>
		{
			g.events.push(Event::Wait);
			Some(Effect::xmult_mod(1.5))
		}
		ScoringCtx::Before => joker_before(g, call),
		ScoringCtx::After => {
			joker_after(&mut g.areas.jokers[ji], &mut g.events, j, call.blueprinted)
		}
		ScoringCtx::Main => joker_main(g, call),
		ScoringCtx::DestroyingCard { .. }
		| ScoringCtx::DebuffedHand
		| ScoringCtx::Repetition { .. }
		| ScoringCtx::OtherJoker { .. } => None,
	}
}

/// `context.individual` for the scored card `other` of the play area.
fn individual_played(
	areas: &mut Areas, run: &mut Run, rng: &mut PseudoRandom, round: &Round,
	events: &mut EventQueue, call: Call<'_>, other: usize,
) -> Option<Effect> {
	use Joker as J;
	let Call {
		index: ji,
		joker: j,
		blueprinted,
		scoring: sc,
	} = call;
	let prob = run.probabilities;
	let oc = &areas.play[other];
	match j {
		J::Hiker => {
			areas.play[other].perma_bonus += 5.0;
			Some(Effect::EXTRA)
		}
		J::LuckyCat if oc.lucky_trigger && !blueprinted => {
			areas.jokers[ji].ab.x_mult += 0.25;
			Some(Effect::EXTRA)
		}
		J::Wee if oc.has_rank(Rank::Two) && !blueprinted => {
			areas.jokers[ji].ab.chips += 8.0;
			Some(Effect::EXTRA)
		}
		J::Photograph => {
			let par = areas.pareidolia();
			let first_face = sc.scoring().find(|&i| areas.play[i].is_face(par));
			(first_face == Some(other)).then_some(Effect::x_mult(2.0))
		}
		J::J8Ball if areas.consumable_room_after_queued() && oc.has_rank(Rank::Eight) => {
			let k = Key::from("8ball");
			if rng.random(&k) < prob / 4.0 {
				areas.consumable_buffer += 1.0;
				Some(Effect {
					extra: true,
					eight_ball: true,
					..Effect::NONE
				})
			} else {
				None
			}
		}
		J::Idol
			if oc.effective_rank() == Some(round.idol_card.rank)
				&& oc.is_suit(round.idol_card.suit, SuitTest::Scoring, areas.smeared()) =>
		{
			Some(Effect::x_mult(2.0))
		}
		J::ScaryFace if oc.is_face(areas.pareidolia()) => Some(Effect::chips(30.0)),
		J::Smiley if oc.is_face(areas.pareidolia()) => Some(Effect::mult(5.0)),
		J::Ticket if oc.enhancement == Enhancement::Gold => {
			Some(calculate_joker::pay(run, events, 4.0))
		}
		J::Business if oc.is_face(areas.pareidolia()) => {
			let k = Key::from("business");
			if rng.random(&k) < prob / 2.0 {
				Some(calculate_joker::pay(run, events, 2.0))
			} else {
				None
			}
		}
		_ => rank_joker(oc, j)
			.or_else(|| suit_joker(oc, areas.smeared(), run, round.ancient_suit, rng, events, j)),
	}
}

/// The jokers that pay for the rank of the scored card `oc`.
fn rank_joker(oc: &PlayingCard, j: Joker) -> Option<Effect> {
	use Joker as J;
	match j {
		J::Scholar if oc.has_rank(Rank::Ace) => Some(Effect {
			chips: Some(20.0),
			mult: Some(4.0),
			..Effect::NONE
		}),
		J::WalkieTalkie if oc.has_rank(Rank::Ten) || oc.has_rank(Rank::Four) => Some(Effect {
			chips: Some(10.0),
			mult: Some(4.0),
			..Effect::NONE
		}),
		J::Fibonacci
			if matches!(
				oc.effective_rank(),
				Some(Rank::Two | Rank::Three | Rank::Five | Rank::Eight | Rank::Ace)
			) =>
		{
			Some(Effect::mult(8.0))
		}
		J::EvenSteven
			if oc
				.effective_rank()
				.is_some_and(|r| r <= Rank::Ten && r.id() % 2 == 0) =>
		{
			Some(Effect::mult(4.0))
		}
		J::OddTodd
			if oc
				.effective_rank()
				.is_some_and(|r| (r <= Rank::Ten && r.id() % 2 == 1) || r == Rank::Ace) =>
		{
			Some(Effect::chips(31.0))
		}
		J::Triboulet if oc.has_rank(Rank::Queen) || oc.has_rank(Rank::King) => {
			Some(Effect::x_mult(2.0))
		}
		_ => None,
	}
}

/// The jokers that pay for the suit of the scored card `oc`.
fn suit_joker(
	oc: &PlayingCard, smeared: bool, run: &mut Run, ancient_suit: Suit, rng: &mut PseudoRandom,
	events: &mut EventQueue, j: Joker,
) -> Option<Effect> {
	use Joker as J;
	let scores = |suit: Suit| oc.is_suit(suit, SuitTest::Scoring, smeared);
	match j {
		J::GreedyJoker if scores(Suit::Diamonds) => Some(Effect::mult(3.0)),
		J::LustyJoker if scores(Suit::Hearts) => Some(Effect::mult(3.0)),
		J::WrathfulJoker if scores(Suit::Spades) => Some(Effect::mult(3.0)),
		J::GluttenousJoker if scores(Suit::Clubs) => Some(Effect::mult(3.0)),
		J::RoughGem if scores(Suit::Diamonds) => Some(calculate_joker::pay(run, events, 1.0)),
		J::OnyxAgate if scores(Suit::Clubs) => Some(Effect::mult(7.0)),
		J::Arrowhead if scores(Suit::Spades) => Some(Effect::chips(50.0)),
		J::Bloodstone if scores(Suit::Hearts) => {
			let k = Key::from("bloodstone");
			(rng.random(&k) < run.probabilities / 2.0).then_some(Effect::x_mult(1.5))
		}
		J::Ancient if scores(ancient_suit) => Some(Effect::x_mult(1.5)),
		_ => None,
	}
}

/// `context.individual` for the card `other` held in hand. A debuffed card
/// still triggers the jokers that pay for it, which then return only `extra`.
fn individual_held(
	areas: &Areas, run: &mut Run, rng: &mut PseudoRandom, events: &mut EventQueue, j: Joker,
	other: usize,
) -> Option<Effect> {
	use Joker as J;
	let oc = &areas.hand.cards[other];
	let unless_debuffed = |e: Effect| Some(if oc.core.debuff { Effect::EXTRA } else { e });
	match j {
		J::ShootTheMoon if oc.has_rank(Rank::Queen) => unless_debuffed(Effect::h_mult(13.0)),
		J::Baron if oc.has_rank(Rank::King) => unless_debuffed(Effect::x_mult(1.5)),
		J::ReservedParking if oc.is_face(areas.pareidolia()) => {
			let k = Key::from("parking");
			if rng.random(&k) < run.probabilities / 2.0 {
				if oc.core.debuff {
					Some(Effect::EXTRA)
				} else {
					Some(calculate_joker::pay(run, events, 1.0))
				}
			} else {
				None
			}
		}
		J::RaisedFist => {
			let mut lowest: Option<(usize, Rank)> = None;
			for (i, c) in areas.hand.cards.iter().enumerate() {
				if !c.is_stone() && lowest.is_none_or(|(_, low)| low >= c.rank) {
					lowest = Some((i, c.rank));
				}
			}
			match lowest {
				Some((i, rank)) if i == other => {
					unless_debuffed(Effect::h_mult(2.0 * rank.nominal()))
				}
				_ => None,
			}
		}
		_ => None,
	}
}

/// `context.before`: the jokers that react to the hand as it is played.
fn joker_before(g: &mut Game, call: Call<'_>) -> Option<Effect> {
	use Joker as J;
	let Call {
		index: ji,
		joker: j,
		blueprinted,
		scoring: sc,
	} = call;
	let ab = g.areas.jokers[ji].ab;
	let hands = sc.hands;
	match j {
		J::Trousers
			if (hands.has(HandType::TwoPair) || hands.has(HandType::FullHouse)) && !blueprinted =>
		{
			g.areas.jokers[ji].ab.mult += 2.0;
			Some(Effect::EXTRA)
		}
		J::Space => {
			let k = Key::from("space");
			(g.rng.random(&k) < g.run.probabilities / 4.0).then_some(Effect {
				level_up: true,
				..Effect::NONE
			})
		}
		J::Square if g.areas.play.len() == 4 && !blueprinted => {
			g.areas.jokers[ji].ab.chips += 4.0;
			Some(Effect::EXTRA)
		}
		J::Runner if hands.has(HandType::Straight) && !blueprinted => {
			g.areas.jokers[ji].ab.chips += 15.0;
			Some(Effect::EXTRA)
		}
		J::MidasMask if !blueprinted => gild_faces(
			&mut g.areas,
			g.run.hands_played,
			&g.blind,
			&mut g.events,
			sc,
		)
		.then_some(Effect::EXTRA),
		J::Vampire if !blueprinted => {
			let drained = drain_enhancements(
				&mut g.areas,
				g.run.hands_played,
				&g.blind,
				&mut g.events,
				sc,
			);
			if drained > 0 {
				g.areas.jokers[ji].ab.x_mult += 0.1 * f64::from(drained);
				Some(Effect::EXTRA)
			} else {
				None
			}
		}
		J::TodoList if Some(sc.name) == ab.to_do => Some(calculate_joker::ease_and_pay(
			&mut g.run,
			&mut g.events,
			4.0,
		)),
		J::Dna if g.round.hands_played == 0.0 && g.areas.play.len() == 1 => {
			let original = g.areas.play[0].clone();
			let mut c = flow::copy_playing_card(&mut g.areas, g.shop.discount_percent, original);
			card_lifecycle::add_to_deck(g, &mut c);
			g.areas.hand.cards.push(c);
			g.events.push(Event::Materialize);
			Some(Effect {
				playing_cards_created: true,
				extra: true,
				..Effect::NONE
			})
		}
		J::RideTheBus if !blueprinted => {
			let par = g.areas.pareidolia();
			let faces = sc.scoring().any(|i| g.areas.play[i].is_face(par));
			if faces {
				g.areas.jokers[ji].ab.mult = 0.0;
				(ab.mult > 0.0).then_some(Effect::EXTRA)
			} else {
				g.areas.jokers[ji].ab.mult += 1.0;
				None
			}
		}
		J::Obelisk if !blueprinted => {
			if !played_most(&g.hands, sc.name) {
				g.areas.jokers[ji].ab.x_mult += 0.2;
				None
			} else if ab.x_mult > 1.0 {
				g.areas.jokers[ji].ab.x_mult = 1.0;
				Some(Effect::EXTRA)
			} else {
				None
			}
		}
		J::GreenJoker if !blueprinted => {
			g.areas.jokers[ji].ab.mult += 1.0;
			Some(Effect::EXTRA)
		}
		_ => None,
	}
}

/// Midas Mask: every scoring face card turns Gold. Whether there was one.
fn gild_faces(
	areas: &mut Areas, hands_played: f64, blind: &Blind, events: &mut EventQueue, sc: &Scoring,
) -> bool {
	let par = areas.pareidolia();
	let debuff = CardDebuff::new(blind, &areas.jokers);
	let mut any = false;
	for i in sc.scoring() {
		if areas.play[i].is_face(par) {
			any = true;
			card_lifecycle::set_enhancement(
				&mut areas.play[i],
				hands_played,
				debuff,
				Enhancement::Gold,
			);
			events.push(Event::Wait);
		}
	}
	any
}

/// Vampire: strips the enhancement of every scoring card that has one;
/// returns how many.
fn drain_enhancements(
	areas: &mut Areas, hands_played: f64, blind: &Blind, events: &mut EventQueue, sc: &Scoring,
) -> u32 {
	let debuff = CardDebuff::new(blind, &areas.jokers);
	let mut drained = 0;
	for i in sc.scoring() {
		let c = &mut areas.play[i];
		if c.enhancement != Enhancement::Base && !c.core.debuff && !c.vampired {
			drained += 1;
			c.vampired = true;
			card_lifecycle::set_enhancement(c, hands_played, debuff, Enhancement::Base);
			let id = c.core.id;
			events.push(Event::Joker(JokerEvent::Unvampire { id }));
		}
	}
	drained
}

/// Obelisk's reset test: no other visible hand has been played as often as
/// `name`.
fn played_most(hands: &HandLevels, name: HandType) -> bool {
	let more_than = hands[name].played;
	HandType::PAIRS_ORDER.into_iter().all(|h| {
		let hl = &hands[h];
		h == name || hl.played < more_than || !hl.visible
	})
}

/// `context.after`: Ice Cream and Seltzer run down, and go when they run out.
fn joker_after(
	joker: &mut Card, events: &mut EventQueue, j: Joker, blueprinted: bool,
) -> Option<Effect> {
	let ab = &mut joker.ab;
	match j {
		Joker::IceCream if !blueprinted => {
			if ab.chips - 5.0 <= 0.0 {
				calculate_joker::schedule_self_destruct(events, joker.core.id);
			} else {
				ab.chips -= 5.0;
			}
			Some(Effect::EXTRA)
		}
		Joker::Selzer if !blueprinted => {
			if ab.extra - 1.0 <= 0.0 {
				calculate_joker::schedule_self_destruct(events, joker.core.id);
			} else {
				ab.extra -= 1.0;
			}
			Some(Effect::EXTRA)
		}
		_ => None,
	}
}

/// Advances Loyalty Card's countdown, refreshed every hand; returns whether
/// this is the hand that pays.
fn tick_loyalty(loyalty: &mut Card, hands_played: f64) -> bool {
	let every = 5.0;
	let rem = rng::lua_mod(
		every - 1.0 - (hands_played - loyalty.core.hands_played_at_create),
		every + 1.0,
	);
	loyalty.ab.loyalty_remaining = rem;
	rem == every
}

/// Flower Pot: the scoring cards cover all four suits (Wild cards fill the
/// suits the others leave open).
fn all_suits_scored(areas: &Areas, sc: &Scoring) -> bool {
	let smeared = areas.smeared();
	let order = [Suit::Hearts, Suit::Diamonds, Suit::Spades, Suit::Clubs];
	let mut seen = SuitSet::default();
	for wild in [false, true] {
		let test = if wild {
			SuitTest::Scoring
		} else {
			SuitTest::IgnoreDebuff
		};
		for i in sc.scoring() {
			let c = &areas.play[i];
			if (c.enhancement == Enhancement::Wild) == wild
				&& let Some(s) = order
					.into_iter()
					.find(|&s| c.is_suit(s, test, smeared) && !seen.contains(s))
			{
				seen.insert(s);
			}
		}
	}
	seen.count() == num::index32(Suit::ALL.len())
}

/// Seeing Double: a scoring Club and a scoring card of another suit (Wild
/// cards fill the first suit still open, Clubs first).
fn club_and_other_scored(areas: &Areas, sc: &Scoring) -> bool {
	let smeared = areas.smeared();
	let mut seen = SuitSet::default();
	for i in sc.scoring() {
		let c = &areas.play[i];
		if c.enhancement != Enhancement::Wild {
			for s in Suit::ALL {
				if c.is_suit(s, SuitTest::Scoring, smeared) {
					seen.insert(s);
				}
			}
		}
	}
	for i in sc.scoring() {
		let c = &areas.play[i];
		if c.enhancement == Enhancement::Wild
			&& let Some(s) = [Suit::Clubs, Suit::Diamonds, Suit::Spades, Suit::Hearts]
				.into_iter()
				.find(|&s| c.is_suit(s, SuitTest::Scoring, smeared) && !seen.contains(s))
		{
			seen.insert(s);
		}
	}
	seen.contains(Suit::Clubs)
		&& [Suit::Hearts, Suit::Diamonds, Suit::Spades]
			.into_iter()
			.any(|s| seen.contains(s))
}

/// `context.joker_main`: the joker's own effect in the joker pass.
fn joker_main(g: &mut Game, call: Call<'_>) -> Option<Effect> {
	use Joker as J;
	let Call {
		index: ji,
		joker: j,
		scoring: sc,
		..
	} = call;
	if j == J::LoyaltyCard && tick_loyalty(&mut g.areas.jokers[ji], g.run.hands_played) {
		return Some(Effect::xmult_mod(4.0));
	}
	let ab = g.areas.jokers[ji].ab;
	let hands = sc.hands;
	let needs = hand_joker(j);
	let has_type = needs.is_none_or(|(h, _)| hands.has(h));
	if j != J::SeeingDouble && ab.x_mult > 1.0 && has_type {
		return Some(Effect::xmult_mod(ab.x_mult));
	}
	if let Some((_, pay)) = needs
		&& has_type
	{
		match pay {
			HandPay::Mult(m) => return Some(Effect::mult_mod(m)),
			HandPay::Chips(c) => return Some(Effect::chip_mod(c)),
			HandPay::XMult => {}
		}
	}
	own_effect(g, j, &ab, sc)
}

/// The joker pass effect of `j` (ability `ab`) that the general rules of
/// [`joker_main`] leave to the joker itself.
fn own_effect(g: &mut Game, j: Joker, ab: &Ability, sc: &Scoring) -> Option<Effect> {
	use Joker as J;
	let money = g.run.dollars + g.run.dollar_buffer;
	match j {
		J::Half if g.areas.play.len() <= 3 => Some(Effect::mult_mod(20.0)),
		J::Abstract => {
			let owned = g
				.areas
				.jokers
				.iter()
				.filter(|c| matches!(c.kind, Kind::Joker(_)));
			Some(Effect::mult_mod(num::float(owned.count()) * 3.0))
		}
		J::Acrobat if g.round.hands_left == 0.0 => Some(Effect::xmult_mod(3.0)),
		J::MysticSummit if g.round.discards_left == 0.0 => Some(Effect::mult_mod(15.0)),
		J::Misprint => {
			let k = Key::from("misprint");
			Some(Effect::mult_mod(g.rng.random_range(&k, 0.0, 23.0)))
		}
		J::Banner if g.round.discards_left > 0.0 => {
			Some(Effect::chip_mod(g.round.discards_left * 30.0))
		}
		J::Stuntman => Some(Effect::chip_mod(250.0)),
		J::Matador if g.blind.triggered => Some(calculate_joker::ease_and_pay(
			&mut g.run,
			&mut g.events,
			8.0,
		)),
		J::Supernova => Some(Effect::mult_mod(g.hands[sc.name].played)),
		J::Vagabond if g.run.dollars <= 4.0 => {
			create_consumable(&mut g.areas, &mut g.events, CardSet::Tarot, "vag")
		}
		J::Superposition
			if sc.scoring().any(|i| g.areas.play[i].has_rank(Rank::Ace))
				&& sc.hands.has(HandType::Straight) =>
		{
			create_consumable(&mut g.areas, &mut g.events, CardSet::Tarot, "sup")
		}
		J::Seance if sc.hands.has(HandType::StraightFlush) => {
			create_consumable(&mut g.areas, &mut g.events, CardSet::Spectral, "sea")
		}
		J::FlowerPot if all_suits_scored(&g.areas, sc) => Some(Effect::xmult_mod(3.0)),
		J::SeeingDouble if club_and_other_scored(&g.areas, sc) => Some(Effect::xmult_mod(2.0)),
		J::BlueJoker if !g.areas.deck.is_empty() => {
			Some(Effect::chip_mod(2.0 * num::float(g.areas.deck.len())))
		}
		J::Erosion => {
			let missing = g.run.starting_deck_size - num::float(g.areas.playing_card_count());
			if missing > 0.0 {
				Some(Effect::mult_mod(4.0 * missing))
			} else {
				None
			}
		}
		J::Bull if money > 0.0 => Some(Effect::chip_mod(2.0 * money.max(0.0))),
		J::Blackboard => {
			let smeared = g.areas.smeared();
			if g.areas.hand.cards.iter().all(|c| c.is_black(smeared)) {
				Some(Effect::xmult_mod(3.0))
			} else {
				None
			}
		}
		J::Stencil if g.areas.joker_room() => Some(Effect::xmult_mod(ab.x_mult)),
		J::FortuneTeller if g.history.consumable_usage_tarot > 0.0 => {
			Some(Effect::mult_mod(g.history.consumable_usage_tarot))
		}
		J::GrosMichel => Some(Effect::mult_mod(15.0)),
		J::Cavendish => Some(Effect::xmult_mod(3.0)),
		J::CardSharp if g.hands[sc.name].played_this_round > 1.0 => Some(Effect::xmult_mod(3.0)),
		J::Bootstraps => {
			let fives = (money / 5.0).floor();
			if fives >= 1.0 {
				Some(Effect::mult_mod(2.0 * fives))
			} else {
				None
			}
		}
		_ => stored_value(j, ab),
	}
}

/// A joker of the joker pass that creates a consumable of `set` (its key
/// `append`), if the consumable area has room.
fn create_consumable(
	areas: &mut Areas, events: &mut EventQueue, set: CardSet, append: &'static str,
) -> Option<Effect> {
	if !areas.consumable_room_after_queued() {
		return None;
	}
	calculate_joker::queue_consumable(areas, events, set, append);
	Some(Effect::EXTRA)
}

/// The jokers whose main effect is a value kept on the card.
fn stored_value(j: Joker, ab: &Ability) -> Option<Effect> {
	use Joker as J;
	match j {
		J::Wee | J::Square | J::Runner | J::IceCream => Some(Effect::chip_mod(ab.chips)),
		J::Castle if ab.chips > 0.0 => Some(Effect::chip_mod(ab.chips)),
		J::Stone if ab.stone_tally > 0.0 => Some(Effect::chip_mod(25.0 * ab.stone_tally)),
		J::SteelJoker if ab.steel_tally > 0.0 => {
			Some(Effect::xmult_mod(1.0 + 0.2 * ab.steel_tally))
		}
		J::DriversLicense if ab.driver_tally >= 16.0 => Some(Effect::xmult_mod(3.0)),
		J::Joker => Some(Effect::mult_mod(ab.mult)),
		J::Ceremonial
		| J::Swashbuckler
		| J::Trousers
		| J::RideTheBus
		| J::Flash
		| J::Popcorn
		| J::GreenJoker
		| J::RedCard
			if ab.mult > 0.0 =>
		{
			Some(Effect::mult_mod(ab.mult))
		}
		J::Caino if ab.caino_xmult > 1.0 => Some(Effect::xmult_mod(ab.caino_xmult)),
		_ => None,
	}
}

/// What a joker pays for the poker hand it needs: its `t_mult`, its
/// `t_chips`, or the `x_mult` kept in its ability.
#[derive(Clone, Copy)]
enum HandPay {
	Mult(f64),
	Chips(f64),
	XMult,
}

/// `ability.type`, the poker hand a joker needs in the play, and what it
/// pays for it.
fn hand_joker(j: Joker) -> Option<(HandType, HandPay)> {
	use HandType as H;
	use Joker as J;
	Some(match j {
		J::Jolly => (H::Pair, HandPay::Mult(8.0)),
		J::Sly => (H::Pair, HandPay::Chips(50.0)),
		J::Duo => (H::Pair, HandPay::XMult),
		J::Zany => (H::ThreeOfAKind, HandPay::Mult(12.0)),
		J::Wily => (H::ThreeOfAKind, HandPay::Chips(100.0)),
		J::Trio => (H::ThreeOfAKind, HandPay::XMult),
		J::Mad => (H::TwoPair, HandPay::Mult(10.0)),
		J::Clever => (H::TwoPair, HandPay::Chips(80.0)),
		J::Crazy => (H::Straight, HandPay::Mult(12.0)),
		J::Devious => (H::Straight, HandPay::Chips(100.0)),
		J::Order => (H::Straight, HandPay::XMult),
		J::Droll => (H::Flush, HandPay::Mult(10.0)),
		J::Crafty => (H::Flush, HandPay::Chips(80.0)),
		J::Tribe => (H::Flush, HandPay::XMult),
		J::Family => (H::FourOfAKind, HandPay::XMult),
		_ => return None,
	})
}
