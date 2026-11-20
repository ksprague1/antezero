//! Build previews: what they show, what they hide, and the effects they count.

use std::mem;

use sim::data::{Back, BlindKind, HandType, Joker, Planet, Rank, Suit, Voucher};
use sim::game::{Game, Hand, Phase};
use sim::inject::Inject;
use sim::rng::PseudoRandom;
use sim::{Card, Kind, Stake, num, obs};

use crate::{
	assert_same, best_candidate, best_legal_play, first_shop, hand_of, obs_and_ops, observe, tok,
	walk_build,
};

/// Panics unless a preview's features are in their ranges: the log ratio
/// (a capacity's, or an item's gain) within `PREVIEW_LOG_MAX`, the ratio up
/// to `PREVIEW_RATIO_MAX` and the clear fraction a fraction.
fn assert_preview_in_range(log_ratio: f32, ratio: f32, p_clear: f32, what: &str) {
	assert!(
		(-obs::PREVIEW_LOG_MAX..=obs::PREVIEW_LOG_MAX).contains(&f64::from(log_ratio)),
		"{what}: log ratio {log_ratio}"
	);
	assert!(
		(0.0..=obs::PREVIEW_RATIO_MAX).contains(&f64::from(ratio)),
		"{what}: ratio {ratio}"
	);
	assert!(
		(0.0..=1.0).contains(&p_clear),
		"{what}: clear fraction {p_clear}"
	);
}

#[test]
fn build_previews_do_not_reveal_hidden_state() {
	use sim::CardLike;

	/// A bijection of the ids that also scrambles their order.
	fn scramble(cards: &mut [impl CardLike]) {
		for c in cards {
			let core = c.core_mut();
			core.id = core.id.wrapping_mul(0x9e37_79b1) ^ 0x5bd1_e995;
		}
	}

	// Shops, packs and blind select: permuting everything the player cannot
	// see (deck order, card ids, the RNG, a face-down hand card's identity)
	// must leave the observation and every op feature unchanged. Misprint
	// and Crimson Heart make the previewed scoring draw random numbers.
	let mut n = 0;
	for s in 0..30 {
		walk_build(&format!("BP{s}"), 250, |g| {
			if !matches!(g.flow.phase, Phase::Shop | Phase::Pack | Phase::BlindSelect) {
				return;
			}
			let mut a = g.clone();
			if !a.areas.hand.cards.is_empty() && !a.areas.deck.is_empty() {
				a.areas.hand.cards[0].core.face_down = true;
			}
			if n % 2 == 0 {
				a.inject(Inject::joker(Joker::Misprint));
			}
			if n % 3 == 0 {
				a.blind_select.boss = BlindKind::FinalHeart;
			}
			let (oa, pa) = obs_and_ops(&a);
			assert_eq!(
				oa.glob_f[obs::GLOB_HAS_PREVIEW],
				1.0,
				"no preview in {:?}",
				g.flow.phase
			);
			let mut b = a.clone();
			b.areas.deck.reverse();
			b.rng = PseudoRandom::new("ELSEWHERE");
			scramble(&mut b.areas.deck);
			scramble(&mut b.areas.hand.cards);
			scramble(&mut b.areas.discard);
			scramble(&mut b.areas.jokers);
			scramble(&mut b.areas.consumables);
			scramble(&mut b.areas.shop_jokers);
			scramble(&mut b.areas.shop_vouchers);
			scramble(&mut b.areas.shop_boosters);
			scramble(&mut b.areas.pack_cards);
			if b.areas.hand.cards.first().is_some_and(|c| c.core.face_down) {
				let k = b.areas.deck.len() / 2;
				mem::swap(&mut b.areas.hand.cards[0], &mut b.areas.deck[k]);
				b.areas.hand.cards[0].core.face_down = true;
				b.areas.deck[k].core.face_down = true;
			}
			let (ob, pb) = obs_and_ops(&b);
			assert_same(
				&oa,
				&ob,
				&format!("hidden state permuted in {:?}", g.flow.phase),
			);
			assert_eq!(
				pa, pb,
				"op features reveal hidden state in {:?}",
				g.flow.phase
			);
			n += 1;
		});
	}
	assert!(n > 100, "only {n} states checked");
}

#[test]
fn build_previews_are_consistent_and_bounded() {
	let item_zone = |z: i32| {
		obs::ItemZone::ALL
			.into_iter()
			.any(|zone| obs::Zone::from(zone).id() == z)
	};
	let (mut previewed, mut states) = (0, 0);
	for s in 0..40 {
		walk_build(&format!("PB{s}"), 250, |g| {
			let (a, ops) = obs_and_ops(g);
			let (b, ops2) = obs_and_ops(g);
			assert_same(&a, &b, "repeated encoding");
			assert_eq!(ops, ops2);
			let shopping = matches!(g.flow.phase, Phase::Shop | Phase::Pack | Phase::BlindSelect);
			assert_eq!(
				a.glob_f[obs::GLOB_HAS_PREVIEW],
				if shopping { 1.0 } else { 0.0 }
			);
			let (capacities, _) =
				a.glob_f[obs::GLOB_PREVIEW].as_chunks::<{ obs::CAPACITY_FLOATS }>();
			for &[lr, r, p] in capacities {
				assert_preview_in_range(lr, r, p, "capacity");
				if !shopping {
					assert_eq!((lr, r, p), (0.0, 0.0, 0.0));
				}
			}
			let (rows, _) = a.tok_f.as_chunks::<{ obs::TOK_FLOAT }>();
			for (i, row) in rows.iter().enumerate() {
				let z = tok(&a, i).zone;
				let columns: [f32; obs::PREVIEW_TOK] = row[obs::TOK_PREVIEW..]
					.try_into()
					.expect("the preview columns end a token row");
				if let Some(p) = obs::ItemPreview::from_columns(columns) {
					assert!(shopping && item_zone(z), "preview on token {i} zone {z}");
					assert_preview_in_range(p.gain, p.ratio, p.p_clear, &format!("token {i}"));
					previewed += 1;
				} else {
					assert_eq!(columns, [0.0; obs::PREVIEW_TOK], "token {i} zone {z}");
				}
			}
			for o in &ops {
				assert!(
					o.kind == obs::OpKind::PlayTop
						|| o.feats[..obs::OP_ORDER_GAIN] == [0.0; obs::OP_ORDER_GAIN],
					"{o:?}"
				);
			}
			states += 1;
		});
	}
	assert!(
		states > 200 && previewed > 100,
		"{states} states, {previewed} previewed items"
	);
}

#[test]
fn item_previews_see_planets_for_other_hands() {
	// Every hand dealt is the same 8 cards (a Pair of Queens); a Pluto makes a
	// lone Ace the best play, so it must preview as a gain.
	let mut g = Game::new_run("PLUTO", Back::Red, Stake::White);
	g.flow.phase = Phase::Shop;
	g.blind_select.boss = BlindKind::Small;
	g.run.dollars = 50.0;
	g.areas.deck.clear();
	let specs = [
		(Rank::Queen, Suit::Spades),
		(Rank::Queen, Suit::Hearts),
		(Rank::Ace, Suit::Diamonds),
		(Rank::King, Suit::Clubs),
		(Rank::Eight, Suit::Clubs),
		(Rank::Six, Suit::Spades),
		(Rank::Four, Suit::Hearts),
		(Rank::Two, Suit::Diamonds),
	];
	g.areas.deck.extend(hand_of(&specs));
	let h = &mut g.hands[HandType::HighCard];
	(h.level, h.chips, h.mult) = (2.0, 15.0, 2.0);
	let mut pluto = Card::new(999, Kind::Planet(Planet::Pluto));
	(pluto.core.cost, pluto.core.base_cost) = (3.0, 3.0);
	g.areas.shop_jokers.push(pluto.into());
	let mut cache = Vec::new();
	let preview = obs::build_preview(&g, &mut cache).expect("a shop has a build preview");
	let base = preview.cap_boss;
	let p = preview
		.item_feats(&g, obs::ItemZone::ShopCard, 0)
		.expect("a planet for sale has a preview");
	let mut after = g;
	let h = &mut after.hands[HandType::HighCard];
	(h.level, h.chips, h.mult) = (3.0, 25.0, 3.0);
	let fresh = obs::capacity(&after, BlindKind::Small, &[1, 2, 3, 4]);
	assert!(fresh.mean > base.mean, "{base:?} {fresh:?}");
	let want = num::feat(((fresh.hands * fresh.mean + 1.0) / (base.hands * base.mean + 1.0)).ln());
	assert!(
		(p.gain - want).abs() < 0.01,
		"Pluto's preview {p:?}, want gain {want}"
	);
}

#[test]
fn kickers_are_tried_for_blackboard_and_raised_fist() {
	// Blackboard and Raised Fist pay for what is not held, so the best play often carries kickers; the shop preview's
	// best play and the round's scored candidates must find it (the scoring cores alone miss it on most hands).
	let shop = first_shop("KK", 0);
	for j in [Joker::Blackboard, Joker::RaisedFist] {
		let mut g = shop.clone();
		g.inject(Inject::joker(j));
		let (mut found, mut offered, mut kicker, n) = (0, 0, 0, 40u64);
		for s in 0..n {
			let v = obs::virtual_round(g.clone(), BlindKind::Small, s);
			let (got, full) = (
				obs::best_play(&v),
				best_legal_play(&v, Hand::HIGHLIGHT_LIMIT),
			);
			found += u64::from(got >= full * 0.999);
			offered += u64::from(best_candidate(&v) >= full * 0.999);
			let small = best_legal_play(&v, 2);
			kicker += u64::from(full > small * 1.5);
		}
		assert!(
			found >= n * 9 / 10,
			"{j:?}: the previewed play is the best legal one in only {found} of {n} hands"
		);
		assert!(
			offered >= n * 9 / 10,
			"{j:?}: the best legal play is among the round's scored candidates in only {offered} of {n} hands"
		);
		assert!(
			kicker >= 8,
			"{j:?}: kickers decide only {kicker} of {n} hands (the test needs hands where they do)"
		);
	}
}

#[test]
fn previews_place_copy_jokers_where_the_hand_scores_most() {
	// Blueprint copies its right neighbour and Brainstorm the leftmost joker: the previewed row must score as much as
	// the best of all its orders (three jokers: every order is tried here).
	let shop = first_shop("CP", 0);
	for copy in [Joker::Blueprint, Joker::Brainstorm] {
		let mut g = shop.clone();
		for j in [Joker::Joker, Joker::Cavendish, copy] {
			g.inject(Inject::joker(j));
		}
		let mut matters = 0;
		for s in 0..16u64 {
			let v = obs::virtual_round(g.clone(), BlindKind::Small, s);
			let got = obs::best_play(&v);
			let (mut best, mut worst) = (0.0f64, f64::INFINITY);
			for p in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [
				2, 1, 0,
			]] {
				let mut w = v.clone();
				w.areas.jokers = p.iter().map(|&i| v.areas.jokers[i].clone()).collect();
				for (i, c) in w.areas.jokers.iter_mut().enumerate() {
					c.core.id = 5_000_000 + num::index32(i);
				}
				let x = obs::best_play(&w);
				(best, worst) = (best.max(x), worst.min(x));
			}
			assert!(
				got >= best * 0.9999,
				"{copy:?} seed {s}: previewed {got}, the best order scores {best}"
			);
			matters += usize::from(best > worst * 1.2);
		}
		assert!(
			matters >= 8,
			"{copy:?}: the order matters in only {matters} hands"
		);
	}
}

#[test]
fn hieroglyph_preview_shows_its_cost_and_the_nearer_target() {
	// Hieroglyph costs a hand and lowers the ante (the same hand is dealt with and without it).
	let mut g = first_shop("HG", 0);
	g.inject(Inject::joker(Joker::Joker));
	let mut v = Card::new(1 << 30, Kind::Voucher(Voucher::Hieroglyph));
	v.core.cost = 10.0;
	g.areas.shop_vouchers.push(v);
	let mut cache = Vec::new();
	let preview = obs::build_preview(&g, &mut cache).expect("a shop has a build preview");
	let p = preview
		.item_feats(
			&g,
			obs::ItemZone::ShopVoucher,
			g.areas.shop_vouchers.len() - 1,
		)
		.expect("a voucher for sale has a preview");
	// 3 hands for 4: the gain is the cost (the last boss is no nearer); the ratio column is against the
	// ante-0 boss, a third of the ante-1 target.
	assert!(
		(p.gain - 0.75f32.ln()).abs() < 0.02,
		"Hieroglyph's preview {p:?}"
	);
	let sold = preview
		.item_feats(&g, obs::ItemZone::Joker, g.areas.jokers.len() - 1)
		.expect("a joker has a sale preview")
		.ratio;
	assert!(
		p.ratio > sold,
		"Hieroglyph's ratio {} is against the lower target (selling a joker leaves {sold})",
		p.ratio
	);
}

#[test]
fn item_previews_count_the_growth_a_purchase_or_sale_triggers() {
	// Using a planet grows Constellation (x0.1) and selling a card grows Campfire (x0.25): a preview of the item
	// includes that growth, so it is worth exactly that much more than the same item without the joker.
	let shop = first_shop("GR", 0);
	let gain = |with: Option<Joker>, zone: obs::ItemZone| {
		let mut g = shop.clone();
		g.areas.shop_jokers.clear();
		g.inject(Inject::joker(Joker::Joker));
		if let Some(j) = with {
			g.inject(Inject::joker(j));
		}
		let mut p = Card::new(999, Kind::Planet(Planet::Pluto));
		(p.core.cost, p.core.base_cost) = (3.0, 3.0);
		g.areas.shop_jokers.push(p.into());
		let mut cache = Vec::new();
		obs::build_preview(&g, &mut cache)
			.expect("a shop has a build preview")
			.item_feats(&g, zone, 0)
			.expect("the item has a preview")
			.gain
	};
	let planet = gain(Some(Joker::Constellation), obs::ItemZone::ShopCard)
		- gain(None, obs::ItemZone::ShopCard);
	assert!(
		(planet - 1.1f32.ln()).abs() < 0.01,
		"Constellation adds {planet} to a planet's preview"
	);
	// Selling the +4 Mult joker: Campfire's x1.25 comes on top of the loss.
	let sale = gain(Some(Joker::Campfire), obs::ItemZone::Joker) - gain(None, obs::ItemZone::Joker);
	assert!(
		(sale - 1.25f32.ln()).abs() < 0.01,
		"Campfire adds {sale} to a sale's preview"
	);
}

#[test]
fn satellite_shows_its_payout() {
	// Satellite pays a dollar per distinct planet used this run; the count is on its card, so in its token.
	let mut g = first_shop("SAT", 0);
	g.inject(Inject::joker(Joker::Satellite));
	let mut a = observe(&g);
	g.history.planets_used.insert(Planet::Pluto);
	g.history.planets_used.insert(Planet::Mars);
	let b = observe(&g);
	let column = a
		.tok_f
		.iter()
		.zip(&b.tok_f)
		.position(|(x, y)| x != y)
		.expect("Satellite's payout changes a token column");
	assert_eq!((a.tok_f[column], b.tok_f[column]), (0.0, 0.2));
	a.tok_f[column] = b.tok_f[column];
	assert_same(&a, &b, "Satellite's payout aside");
}
