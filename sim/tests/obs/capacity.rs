//! What a build can score over a round (`obs::capacity`) and the rounds the previews play.

use sim::data::{BlindKind, Joker, Planet, Voucher};
use sim::game::Phase;
use sim::inject::Inject;
use sim::{Action, Card, Kind, num, obs};

use crate::{first_shop, first_shops};

#[test]
fn capacity_orders_jokers_and_counts_grabber() {
	// At a shop: a x3 joker adds more than a +4 mult joker, which adds
	// something; Grabber adds a hand and nothing else.
	let g = first_shop("CAP", 0);
	let seeds = [1u64, 2, 3, 4];
	let boss = g.blind_select.boss;
	let base = obs::capacity(&g, boss, &seeds);
	assert!(base.mean > 0.0, "no score: {base:?}");
	assert_eq!(base.hands, 4.0, "{base:?}");
	assert!(base.chips > 0.0, "no target: {base:?}");
	assert!(base.ratio() > 0.0, "no ratio: {base:?}");
	let mut plus = g.clone();
	plus.inject(Inject::joker(Joker::Joker));
	let mut times = g.clone();
	times.inject(Inject::joker(Joker::Cavendish));
	let (cp, ct) = (
		obs::capacity(&plus, boss, &seeds),
		obs::capacity(&times, boss, &seeds),
	);
	assert!(
		ct.mean > cp.mean && cp.mean > base.mean,
		"{base:?} {cp:?} {ct:?}"
	);
	let mut grab = g;
	grab.inject(Inject::Voucher(Voucher::Grabber));
	let cg = obs::capacity(&grab, boss, &seeds);
	assert_eq!(cg.hands, 5.0);
	assert_eq!(cg.mean, base.mean, "the same hands are dealt");
}

#[test]
fn grabber_preview_is_its_extra_hand() {
	// As a shop item, Grabber's gain is its extra hand (same hands dealt).
	let mut shop = first_shop("CAP", 0);
	let mut grabber = Card::new(1 << 30, Kind::Voucher(Voucher::Grabber));
	grabber.core.cost = 10.0;
	shop.areas.shop_vouchers.push(grabber);
	let mut cache = Vec::new();
	let preview = obs::build_preview(&shop, &mut cache).expect("a shop has a build preview");
	let p = preview
		.item_feats(
			&shop,
			obs::ItemZone::ShopVoucher,
			shop.areas.shop_vouchers.len() - 1,
		)
		.expect("a voucher for sale has a preview");
	assert!(
		(p.gain - (5.0f32 / 4.0).ln()).abs() < 0.01,
		"Grabber's preview {p:?}"
	);
}

#[test]
fn psychic_rounds_preview_a_scoring_play() {
	// The Psychic: only five-card plays score, and they are previewed.
	let g = first_shop("CAP", 0);
	for s in 0..8 {
		let v = obs::virtual_round(g.clone(), BlindKind::Psychic, s);
		assert!(
			obs::best_play(&v) > 0.0,
			"no scoring play previewed against The Psychic (seed {s})"
		);
	}
}

#[test]
fn virtual_round_starts_a_fresh_round() {
	let g = first_shop("CAP", 0);
	let boss = g.blind_select.boss;
	let v = obs::virtual_round(g.clone(), boss, 7);
	assert_eq!(v.flow.phase, Phase::Round);
	assert_eq!(
		v.areas.hand.cards.len(),
		num::trunc(g.areas.hand.limit.limit)
	);
	assert_eq!(v.blind.kind, Some(boss));
	assert!(
		v.areas.hand.cards.iter().all(|c| !c.core.face_down),
		"a face-down card in the dealt hand"
	);
	assert_eq!(v.areas.play.len(), 0, "cards left in play");
	assert_eq!(v.areas.discard.len(), 0, "cards left in the discard pile");
	assert_eq!(
		v.areas.deck.len() + v.areas.hand.cards.len(),
		g.areas.playing_card_count()
	);
}

#[test]
fn preview_hands_differ_between_games() {
	use std::collections::{HashMap, HashSet};

	// Two games at the same ante and round, with the same cards and the same boss, must not preview on the same
	// sampled hands (their histories differ): otherwise an item's preview is the same wrong number in every game.
	let mut by_boss: HashMap<BlindKind, HashSet<u64>> = HashMap::new();
	let shops = first_shops("DV", 0).take(64);
	for mut g in shops {
		g.inject(Inject::joker(Joker::Jolly));
		let mut cache = Vec::new();
		let preview = obs::build_preview(&g, &mut cache).expect("a shop has a build preview");
		by_boss
			.entry(g.blind_select.boss)
			.or_default()
			.insert(preview.cap_boss.mean.to_bits());
	}
	// The Pillar debuffs the cards a game has played, so its capacities differ anyway.
	let most = by_boss
		.iter()
		.filter(|(boss, _)| **boss != BlindKind::Pillar)
		.map(|(_, means)| means.len())
		.max()
		.unwrap();
	assert!(
		most >= 4,
		"at most {most} different capacities among the games facing the same boss"
	);
}

#[test]
fn previews_start_the_round_as_burglar_and_chicot_do() {
	// Burglar trades the discards for three hands when the blind is selected, and Chicot disables the boss: the
	// previewed round must start like the real one.
	let mut burglar = first_shop("BG", 0);
	burglar.inject(Inject::joker(Joker::Burglar));
	let preview = obs::virtual_round(burglar.clone(), BlindKind::Big, 3);
	let mut real = burglar.clone();
	real.step(Action::EndShop).unwrap();
	real.step(Action::SelectBlind).unwrap();
	assert_eq!(
		(preview.round.hands_left, preview.round.discards_left),
		(real.round.hands_left, real.round.discards_left)
	);
	assert_eq!(
		(preview.round.hands_left, preview.round.discards_left),
		(7.0, 0.0)
	);
	assert_eq!(obs::capacity(&burglar, BlindKind::Big, &[1, 2]).hands, 7.0);
	// A copy joker next to Burglar repeats it: ten hands, in the preview whatever the current order.
	let mut copied = first_shop("BG", 0);
	for j in [Joker::Blueprint, Joker::Burglar] {
		copied.inject(Inject::joker(j));
	}
	let mut real = copied.clone();
	real.step(Action::EndShop).unwrap();
	real.step(Action::SelectBlind).unwrap();
	assert_eq!(real.round.hands_left, 10.0);
	copied.areas.jokers.reverse();
	assert_eq!(
		obs::virtual_round(copied.clone(), BlindKind::Big, 3)
			.round
			.hands_left,
		10.0
	);
	let mut chicot = first_shop("BG", 0);
	let wall = obs::virtual_round(chicot.clone(), BlindKind::Wall, 3)
		.blind
		.chips;
	chicot.inject(Inject::joker(Joker::Chicot));
	let preview = obs::virtual_round(chicot.clone(), BlindKind::Wall, 3);
	assert!(
		preview.blind.disabled && preview.blind.chips == wall / 2.0,
		"The Wall with Chicot: {} chips (without: {wall})",
		preview.blind.chips
	);
	assert!(
		!obs::virtual_round(chicot.clone(), BlindKind::Big, 3)
			.blind
			.disabled
	);
	// As a shop item against The Wall, Chicot's gain counts the halved target.
	let mut shop = first_shop("BG", 0);
	shop.blind_select.boss = BlindKind::Wall;
	shop.run.dollars = 50.0;
	shop.areas.shop_jokers.clear();
	shop.areas
		.shop_jokers
		.push(chicot.areas.jokers.last().unwrap().clone().into());
	let mut cache = Vec::new();
	let preview = obs::build_preview(&shop, &mut cache).expect("a shop has a build preview");
	let p = preview
		.item_feats(&shop, obs::ItemZone::ShopCard, 0)
		.expect("a joker for sale has a preview");
	assert!(
		(p.gain - 2f32.ln()).abs() < 0.02,
		"Chicot's preview against The Wall {p:?}"
	);
}

#[test]
fn capacity_against_the_eye_and_the_mouth_is_a_whole_round() {
	// The Eye scores no hand type twice and The Mouth only the first hand's type, so the round is worth less than
	// its hands' best plays added up; one hand alone is worth the same.
	let (mut eye_less, mut mouth_less) = (0, 0);
	for s in 0..12u64 {
		let mut g = first_shop("EM", s * 7);
		g.inject(Inject::joker(Joker::Jolly));
		let seeds = [s + 1, s + 2, s + 3, s + 4];
		let free = obs::capacity(&g, BlindKind::Small, &seeds).mean;
		for (boss, less) in [
			(BlindKind::Eye, &mut eye_less),
			(BlindKind::Mouth, &mut mouth_less),
		] {
			let c = obs::capacity(&g, boss, &seeds);
			assert!(
				c.mean <= free + 1e-9,
				"{boss:?}: {} above the unrestricted {free}",
				c.mean
			);
			*less += usize::from(c.mean < free * 0.98);
			// Four hands, four sampled hands: one whole round, which clears exactly when its capacity covers the target.
			assert_eq!(
				c.p_clear,
				f64::from(u8::from(c.ratio() >= 1.0)),
				"{boss:?}: clear fraction {} at ratio {}",
				c.p_clear,
				c.ratio()
			);
			// A longer round (Burglar: seven hands) keeps using up hand types past the four sampled hands.
			let mut long = g.clone();
			long.inject(Inject::joker(Joker::Burglar));
			let (l, f) = (
				obs::capacity(&long, boss, &seeds),
				obs::capacity(&long, BlindKind::Small, &seeds),
			);
			assert!(
				l.hands == 7.0
					&& l.mean <= f.mean + 1e-9
					&& (boss != BlindKind::Eye || l.mean <= c.mean + 1e-9),
				"{boss:?}: seven hands {l:?} vs four {c:?}"
			);
			assert_eq!(
				obs::capacity(&g, boss, &seeds[..1]).mean,
				obs::capacity(&g, BlindKind::Small, &seeds[..1]).mean,
				"{boss:?}: one hand"
			);
			// A round of one hand has nothing to restrict, however many hands are sampled.
			let mut one = g.clone();
			one.round.base_hands = 1.0;
			assert_eq!(
				obs::capacity(&one, boss, &seeds).mean,
				obs::capacity(&one, BlindKind::Small, &seeds).mean,
				"{boss:?}: rounds of one hand"
			);
		}
	}
	assert!(
		eye_less >= 9 && mouth_less >= 6,
		"the restriction costs something in only {eye_less} (Eye) and {mouth_less} (Mouth) of 12 builds"
	);
}

#[test]
fn item_capacity_against_the_eye_is_on_the_rounds_scale() {
	// An item's capacity columns are on the round's scale: against The Eye its ratio is lower than with the boss
	// disabled (Chicot owned, which scores nothing itself) by what the rule costs the build.
	let mut scale = 0;
	for s in 0..12u64 {
		let shop = first_shop("EM", s * 7);
		let item = |chicot: bool| {
			let mut g = shop.clone();
			g.blind_select.boss = BlindKind::Eye;
			g.run.dollars = 50.0;
			g.inject(Inject::joker(Joker::Jolly));
			if chicot {
				g.inject(Inject::joker(Joker::Chicot));
			}
			g.areas.shop_jokers.clear();
			let mut p = Card::new(999, Kind::Planet(Planet::Pluto));
			(p.core.cost, p.core.base_cost) = (3.0, 3.0);
			g.areas.shop_jokers.push(p.into());
			let mut cache = Vec::new();
			let preview = obs::build_preview(&g, &mut cache).expect("a shop has a build preview");
			let ratio = preview
				.item_feats(&g, obs::ItemZone::ShopCard, 0)
				.expect("a planet for sale has a preview")
				.ratio;
			(f64::from(ratio), preview.cap_boss.mean)
		};
		let ((r_eye, m_eye), (r_free, m_free)) =
			(item(/* chicot= */ false), item(/* chicot= */ true));
		// Only ratios below the feature's cap show the scale.
		if r_free < obs::PREVIEW_RATIO_MAX - 0.1 && m_eye < m_free * 0.98 {
			assert!(
				(r_eye / r_free - m_eye / m_free).abs() < 0.01,
				"item ratio {r_eye} vs {r_free} with the boss disabled; the build's capacity {m_eye} vs {m_free}"
			);
			scale += 1;
		}
	}
	assert!(
		scale >= 5,
		"only {scale} builds where The Eye costs something"
	);
}
