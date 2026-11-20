//! The build-preview cache gives what a fresh encode gives.

use sim::data::{Back, Edition, Joker, Planet, Voucher};
use sim::game::{Game, Phase};
use sim::inject::{Consumable, Inject};
use sim::obs::{self, BuildPreview, ObsBuf};
use sim::rng::Xorshift;
use sim::{Action, Card, Kind, Stake, num};

use crate::{WALK_SEED, assert_same, best_round_play, observe, random_action, walk_with};

/// The observation of `g` encoded with the previews `cache` holds for it,
/// checked against a fresh encode; also whether `g` has a build preview.
fn cached_encode(g: &Game, cache: &mut Vec<BuildPreview>, what: &str) -> (ObsBuf, bool) {
	let mut cached = ObsBuf::new();
	let preview = obs::build_preview(g, cache);
	let has = preview.is_some();
	let deck_tokens = true;
	obs::encode_with(g, &mut cached.as_mut(), deck_tokens, preview);
	assert_same(
		&cached,
		&observe(g),
		&format!(
			"cached preview vs a fresh one in {:?} ({what})",
			g.flow.phase
		),
	);
	(cached, has)
}

#[test]
fn build_preview_cache_matches_fresh_encode() {
	// The env reuses a cached preview while the joker multiset, money and
	// cards stay the same (joker moves, the blind select after a shop); it
	// must equal a fresh encode, also with order-sensitive jokers, and the
	// blind select must show the capacity the shop showed.
	let (mut after_moves, mut after_shops) = (0, 0);
	for s in 0..30u64 {
		let mut g = Game::new_run(&format!("CM{s}"), Back::Red, Stake::White);
		let mut cache = Vec::new();
		let mut rng = Xorshift::new(WALK_SEED ^ (s << 1));
		let mut acts = Vec::new();
		let (mut injected, mut moved) = (false, false);
		let mut shop_capacity: Option<Vec<f32>> = None;
		walk_with(&mut g, 300, |g| {
			if g.flow.phase == Phase::Shop && !injected {
				injected = true;
				for (joker, edition) in [
					(Joker::Joker, Some(Edition::Polychrome)),
					(Joker::Blueprint, None),
					(Joker::Cavendish, Some(Edition::Holo)),
					(Joker::Cavendish, None),
				] {
					g.inject(Inject::Joker {
						joker,
						edition,
						eternal: false,
					});
				}
				if s % 2 == 0 {
					g.inject(Inject::joker(Joker::Bull));
				}
			}
			let (cached, has) = cached_encode(g, &mut cache, &format!("game {s}"));
			after_moves += usize::from(has && moved);
			if let Some(capacity) = shop_capacity.take() {
				assert_eq!(
					&cached.glob_f[obs::GLOB_PREVIEW],
					&capacity[..],
					"the build did not change, its capacity did (shop -> {:?})",
					g.flow.phase
				);
				after_shops += 1;
			}
			let x = rng.next_u64();
			let act = match best_round_play(g) {
				Some(play) if !x.is_multiple_of(7) => play,
				_ if g.flow.phase == Phase::Shop
					&& g.areas.jokers.len() >= 2
					&& x.is_multiple_of(3) =>
				{
					let count = num::byte(g.areas.jokers.len());
					let [_, from, to, ..] = x.to_le_bytes();
					let (from, to) = (from % count, to % count);
					Action::MoveJoker {
						from,
						to: if to == from { (to + 1) % count } else { to },
					}
				}
				_ => random_action(g, x, &mut acts),
			};
			moved = matches!(act, Action::MoveJoker { .. });
			shop_capacity = (g.flow.phase == Phase::Shop && act == Action::EndShop)
				.then(|| cached.glob_f[obs::GLOB_PREVIEW].to_vec());
			act
		});
	}
	assert!(
		after_moves > 30 && after_shops > 20,
		"{after_moves} states after joker moves, {after_shops} after shops"
	);
}

fn money_joker(c: &Card) -> bool {
	matches!(c.kind, Kind::Joker(Joker::Bull | Joker::Bootstraps))
}

/// On entering a shop: money, and at random an edition planet, the
/// Observatory voucher for sale, a money joker for sale; in every eighth
/// game the owned money jokers are debuffed.
fn stock_shop(g: &mut Game, rng: &mut Xorshift, game_index: u64) {
	g.inject(Inject::Dollars(25.0));
	if rng.below(5) < 2 {
		let editions = [
			None,
			Some(Edition::Foil),
			Some(Edition::Holo),
			Some(Edition::Polychrome),
		];
		let edition = editions[rng.below(editions.len())];
		let planets = [
			Planet::Pluto,
			Planet::Mercury,
			Planet::Uranus,
			Planet::Venus,
		];
		let planet = planets[rng.below(planets.len())];
		g.inject(Inject::Consumable {
			card: Consumable::Planet(planet),
			edition,
		});
	}
	if rng.below(5) < 2
		&& !g.areas.shop_vouchers.is_empty()
		&& !g.history.used_vouchers.contains(Voucher::Observatory)
	{
		g.areas.shop_vouchers[0].kind = Kind::Voucher(Voucher::Observatory);
	}
	if rng.below(10) < 3 && !g.areas.shop_jokers.is_empty() {
		let mut donor = g.clone();
		donor.inject(Inject::joker(if rng.below(2) == 0 {
			Joker::Bull
		} else {
			Joker::Bootstraps
		}));
		g.areas.shop_jokers[0] = donor
			.areas
			.jokers
			.pop()
			.expect("the joker injected into the copy")
			.into();
	}
	if game_index.is_multiple_of(8) {
		// The previewed blind clears this debuff: the joker still counts.
		for c in g.areas.jokers.iter_mut().filter(|c| money_joker(c)) {
			c.core.debuff = true;
		}
	}
}

/// The best play in a round; elsewhere a random action that keeps the
/// order as it is and mostly keeps the shop open (one decision in eight in
/// a shop leaves it, as does one with nothing else to do), selling a joker
/// now and then.
fn stay_and_shop(g: &Game, rng: &mut Xorshift, acts: &mut Vec<Action>) -> Action {
	if let Some(play) = best_round_play(g) {
		return play;
	}
	g.legal_actions(acts);
	let sell = rng.below(7) == 0;
	let pool: Vec<Action> = acts
		.iter()
		.copied()
		.filter(|a| {
			!matches!(
				a,
				Action::EndShop
					| Action::MoveJoker { .. }
					| Action::MoveHand { .. }
					| Action::SortHand(_)
			) && (sell || !matches!(a, Action::SellJoker(_)))
		})
		.collect();
	if pool.is_empty() || (g.flow.phase == Phase::Shop && rng.below(8) == 0) {
		acts.iter()
			.copied()
			.find(|a| matches!(a, Action::EndShop))
			.unwrap_or(acts[0])
	} else {
		pool[rng.below(pool.len())]
	}
}

/// What a state puts in play that the preview cache key leaves out until
/// a preview reads it: one flag per case, or a count of states per case.
#[derive(Clone, Copy, Debug, Default)]
struct Cover<T> {
	money_owned: T,
	money_debuffed: T,
	money_for_sale: T,
	observatory_owned: T,
	observatory_for_sale: T,
	edition_consumable: T,
}

impl Cover<bool> {
	fn of(g: &Game) -> Cover<bool> {
		use sim::CardLike;

		Cover {
			money_owned: g.areas.jokers.iter().any(money_joker),
			money_debuffed: g
				.areas
				.jokers
				.iter()
				.any(|c| money_joker(c) && c.core.debuff),
			money_for_sale: g
				.areas
				.shop_jokers
				.iter()
				.chain(&g.areas.pack_cards)
				.filter_map(CardLike::as_card)
				.any(money_joker),
			observatory_owned: g.history.used_vouchers.contains(Voucher::Observatory),
			observatory_for_sale: g
				.areas
				.shop_vouchers
				.iter()
				.any(|c| c.kind == Kind::Voucher(Voucher::Observatory)),
			edition_consumable: g.areas.consumables.iter().any(|c| {
				matches!(
					c.core.edition,
					Some(Edition::Foil | Edition::Holo | Edition::Polychrome)
				)
			}),
		}
	}
}

impl Cover<usize> {
	fn add(&mut self, hit: Cover<bool>) {
		self.money_owned += usize::from(hit.money_owned);
		self.money_debuffed += usize::from(hit.money_debuffed);
		self.money_for_sale += usize::from(hit.money_for_sale);
		self.observatory_owned += usize::from(hit.observatory_owned);
		self.observatory_for_sale += usize::from(hit.observatory_for_sale);
		self.edition_consumable += usize::from(hit.edition_consumable);
	}

	/// The fewest states any case was seen in.
	fn least(&self) -> usize {
		[
			self.money_owned,
			self.money_debuffed,
			self.money_for_sale,
			self.observatory_owned,
			self.observatory_for_sale,
			self.edition_consumable,
		]
		.into_iter()
		.min()
		.expect("`Cover` has six cases")
	}
}

#[test]
fn build_preview_cache_ignores_money_and_consumables_only_when_unread() {
	// The cache key leaves money and the held consumables out while no
	// preview reads them. Everything that does read them is put in play here
	// (Bull / Bootstraps owned, debuffed, or for sale; Observatory owned or
	// for sale; consumables with a scoring edition) while money and
	// consumables change under a warm cache: it must always equal a fresh one.
	let mut rng = Xorshift::new(WALK_SEED);
	let (mut cover, mut acts) = (Cover::<usize>::default(), Vec::new());
	for s in 0..40u64 {
		let mut g = Game::new_run(&format!("CK{s}"), Back::Red, Stake::White);
		match s % 4 {
			0 => g.inject(Inject::joker(Joker::Bull)),
			1 => g.inject(Inject::joker(Joker::Bootstraps)),
			2 => g.inject(Inject::Voucher(Voucher::Observatory)),
			_ => {}
		}
		if s % 3 == 0 {
			g.inject(Inject::joker(Joker::Blueprint));
		}
		g.inject(Inject::joker(Joker::Jolly));
		let (mut cache, mut last) = (Vec::new(), g.flow.phase);
		walk_with(&mut g, 300, |g| {
			if g.flow.phase == Phase::Shop && last != Phase::Shop {
				stock_shop(g, &mut rng, s);
			}
			last = g.flow.phase;
			if matches!(g.flow.phase, Phase::Shop | Phase::Pack | Phase::BlindSelect) {
				cached_encode(g, &mut cache, &format!("game {s}"));
				let hit = Cover::of(g);
				cover.add(hit);
				if rng.below(4) == 0 {
					g.inject(Inject::Dollars(if rng.below(2) == 0 { 3.0 } else { -2.0 }));
				}
				if hit.observatory_for_sale && rng.below(3) == 0 {
					// A planet arrives while the Observatory voucher's preview is cached.
					g.inject(Inject::Consumable {
						card: Consumable::Planet(if rng.below(2) == 0 {
							Planet::Pluto
						} else {
							Planet::Mercury
						}),
						edition: None,
					});
				}
			}
			stay_and_shop(g, &mut rng, &mut acts)
		});
	}
	assert!(
		cover.least() > 20,
		"a case is in play in too few states: {cover:?}"
	);
}

#[test]
fn held_planets_change_a_cached_observatory_preview() {
	let (mut changed, mut acts) = (0, Vec::new());
	for s in 0..12u64 {
		let mut g = Game::new_run(&format!("CK{s}"), Back::Red, Stake::White);
		g.inject(Inject::joker(Joker::Jolly));
		while !g.is_terminal() && (g.flow.phase != Phase::Shop || g.areas.shop_vouchers.is_empty())
		{
			let act = best_round_play(&g).unwrap_or_else(|| {
				g.legal_actions(&mut acts);
				acts[0]
			});
			g.step(act).unwrap();
		}
		if g.is_terminal() {
			continue;
		}
		g.areas.shop_vouchers[0].kind = Kind::Voucher(Voucher::Observatory);
		let observatory = |g: &Game, cache: &mut Vec<BuildPreview>| {
			obs::build_preview(g, cache)
				.expect("a shop has a build preview")
				.item_feats(g, obs::ItemZone::ShopVoucher, 0)
		};
		let mut cache = Vec::new();
		let before = observatory(&g, &mut cache);
		for p in [Planet::Pluto, Planet::Mercury] {
			g.inject(Inject::Consumable {
				card: Consumable::Planet(p),
				edition: None,
			});
		}
		let warm = observatory(&g, &mut cache);
		assert_eq!(
			warm,
			observatory(&g, &mut Vec::new()),
			"stale Observatory preview (game {s})"
		);
		changed += usize::from(warm != before);
	}
	assert!(
		changed >= 3,
		"the held planets changed Observatory's preview in only {changed} games"
	);
}
