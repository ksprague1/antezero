use std::mem;

use sim::data::{Back, BlindKind, Joker, Rank, Suit};
use sim::game::{Game, Hand, Phase, Scratch};
use sim::inject::Inject;
use sim::obs::{self, ObsBuf};
use sim::rng::PseudoRandom;
use sim::{Action, Stake};

use crate::{best_candidate, best_legal_play, encode_with_deck, hand_of, walk};

#[test]
fn preview_matches_full_step() {
	let alt = PseudoRandom::new("PREVIEW0");
	let mut scratch = Scratch::default();
	let mut checked = 0;
	for s in 0..6 {
		walk(&format!("PV{s}"), 150, |g| {
			if g.flow.phase != Phase::Round {
				return;
			}
			let mut acts = Vec::new();
			g.legal_actions(&mut acts);
			let play_masks = acts
				.iter()
				.filter_map(|a| match *a {
					Action::Play(m) => Some(m),
					_ => None,
				})
				.take(25);
			for m in play_masks {
				let p = obs::preview_score(&mut scratch, g, m, &alt).unwrap().score;
				let mut c = g.clone();
				c.rng = alt.clone();
				c.step(Action::Play(m)).unwrap();
				assert_eq!(p, c.run.last_hand_score, "mask {:b}", m.bits());
				checked += 1;
			}
		});
	}
	assert!(checked > 100, "only {checked} plays checked");
}

#[test]
fn face_down_jokers_get_candidates_that_ignore_their_hidden_order() {
	let mut g = Game::new_run("ACORN", Back::Red, Stake::White);
	// +mult before or after xmult changes the score, so the order matters.
	for j in [Joker::Joker, Joker::Cavendish, Joker::Misprint] {
		g.inject(Inject::joker(j));
	}
	g.step(Action::SelectBlind).unwrap();
	for c in &mut g.areas.jokers {
		c.core.face_down = true;
	}
	let top = obs::top_plays(&g, 8);
	assert!(
		!top.is_empty(),
		"face-down jokers must not disable the previews"
	);
	let mut r = g.clone();
	r.areas.jokers.reverse();
	assert_eq!(
		top,
		obs::top_plays(&r, 8),
		"candidates reveal the hidden joker order"
	);
}

#[test]
fn face_down_candidates_do_not_reveal_identity() {
	for seed in ["FDC1", "FDC2", "FDC3", "FDC4"] {
		let mut g = Game::new_run(seed, Back::Red, Stake::White);
		g.step(Action::SelectBlind).unwrap();
		g.areas.hand.cards[0].core.face_down = true;
		g.areas.hand.cards[3].core.face_down = true;
		let a = obs::top_plays(&g, 8);
		assert!(!a.is_empty(), "no candidates with face-down cards ({seed})");
		let k = g.areas.deck.len() / 2;
		mem::swap(&mut g.areas.hand.cards[0], &mut g.areas.deck[k]);
		g.areas.hand.cards[0].core.face_down = true;
		g.areas.deck[k].core.face_down = false;
		g.areas.deck.reverse();
		let b = obs::top_plays(&g, 8);
		assert_eq!(a, b, "candidates reveal a face-down card ({seed})");
	}
}

#[test]
fn candidates_are_consistent() {
	let mut n = 0;
	for s in 0..20 {
		walk(&format!("CD{s}"), 150, |g| {
			let a = obs::top_plays(g, 8);
			let b = obs::top_plays(g, 8);
			assert_eq!(a, b, "candidates must be deterministic");
			for c in &a {
				assert!(c.min <= c.mean + 1e-9 && c.mean <= c.max + 1e-9, "{c:?}");
				assert!(
					(0.0..=1.0).contains(&c.p_clear),
					"clear fraction out of range: {c:?}"
				);
				n += 1;
			}
			for w in a.windows(2) {
				assert!(
					w[0].p_clear >= w[1].p_clear,
					"candidates not sorted by clear fraction: {w:?}"
				);
			}
		});
	}
	assert!(n > 200, "only {n} candidates checked");
}

#[test]
fn candidates_never_crowd_out_other_ops() {
	let mut g = Game::new_run("ROOM", Back::Red, Stake::White);
	g.step(Action::SelectBlind).unwrap();
	let mut buf = ObsBuf::new();
	let lay = encode_with_deck(&g, &mut buf);
	let (mut few, mut many) = (Vec::new(), Vec::new());
	obs::ops(
		&g,
		&lay,
		obs::OpOptions {
			candidates: 0,
			..Default::default()
		},
		&mut few,
	);
	obs::ops(
		&g,
		&lay,
		obs::OpOptions {
			candidates: obs::MAX_OPS,
			..Default::default()
		},
		&mut many,
	);
	assert!(
		many.len() <= obs::MAX_OPS,
		"{} ops, over MAX_OPS",
		many.len()
	);
	let base: Vec<_> = many
		.iter()
		.filter(|o| o.kind != obs::OpKind::PlayTop)
		.copied()
		.collect();
	assert_eq!(base, few, "candidates replaced other ops");
	assert!(many.len() > few.len(), "no candidates");
}

#[test]
fn candidates_ignore_hidden_card_ids() {
	let mut n = 0;
	for s in 0..10 {
		walk(&format!("ID{s}"), 120, |g| {
			if g.flow.phase != Phase::Round {
				return;
			}
			// Misprint makes every score depend on the chance scenarios.
			let mut a = g.clone();
			a.inject(Inject::joker(Joker::Misprint));
			let mut r = a.clone();
			for c in &mut r.areas.hand.cards {
				c.core.id += 100_000;
			}
			let top = obs::top_plays(&a, 8);
			assert!(
				top.iter().all(|c| c.min < c.max),
				"Misprint plays should vary"
			);
			assert_eq!(top, obs::top_plays(&r, 8), "candidates depend on card ids");
			n += 1;
		});
	}
	assert!(n > 20, "only {n} rounds checked");
}

#[test]
fn candidates_keep_plays_that_hold_cards_back() {
	// Baron (copied by Blueprint and Brainstorm) scores every King held in
	// hand: the best play is a lone Ace that holds all three Kings, which the
	// cheap estimate ranks below every play with a King in it.
	let mut g = Game::new_run("HOLD", Back::Red, Stake::White);
	for j in [Joker::Blueprint, Joker::Baron, Joker::Brainstorm] {
		g.inject(Inject::joker(j));
	}
	g.step(Action::SelectBlind).unwrap();
	g.areas.hand.cards = hand_of(&[
		(Rank::King, Suit::Spades),
		(Rank::King, Suit::Hearts),
		(Rank::King, Suit::Clubs),
		(Rank::Ace, Suit::Diamonds),
		(Rank::Eight, Suit::Clubs),
		(Rank::Six, Suit::Spades),
		(Rank::Four, Suit::Hearts),
		(Rank::Two, Suit::Diamonds),
	]);
	let best = best_legal_play(&g, Hand::HIGHLIGHT_LIMIT);
	assert_eq!(best_candidate(&g), best, "scored candidates");
	assert_eq!(obs::best_play(&g), best, "build preview");
}

#[test]
fn candidates_carry_the_kickers_blackboard_and_raised_fist_need() {
	// No pair, three red cards: with Blackboard the best play is the Ace with all three reds (x3: only black cards
	// stay held); with Raised Fist it is the Ace with the four lowest cards (a Jack is then the lowest card held).
	// Neither is a scoring core, and among the Ace's 98 kicker variants both fall outside the simulated budget.
	for j in [Joker::Blackboard, Joker::RaisedFist] {
		let mut g = Game::new_run("KICK", Back::Red, Stake::White);
		g.inject(Inject::joker(j));
		g.step(Action::SelectBlind).unwrap();
		g.areas.hand.cards = hand_of(&[
			(Rank::Ace, Suit::Spades),
			(Rank::King, Suit::Clubs),
			(Rank::Queen, Suit::Spades),
			(Rank::Jack, Suit::Clubs),
			(Rank::Nine, Suit::Spades),
			(Rank::Seven, Suit::Hearts),
			(Rank::Four, Suit::Diamonds),
			(Rank::Two, Suit::Hearts),
		]);
		let best = best_legal_play(&g, Hand::HIGHLIGHT_LIMIT);
		assert!(best > 3.0 * 16.0 - 0.5, "{j:?}: best legal play {best}");
		assert_eq!(best_candidate(&g), best, "{j:?}: scored candidates");
		assert_eq!(obs::best_play(&g), best, "{j:?}: build preview");
	}
}

#[test]
fn candidates_follow_psychic_and_splash() {
	// No pair: every scoring core is a single card. The Psychic scores only
	// five-card plays; Splash scores every played card (kickers included).
	let mut g = Game::new_run("PSYCHIC", Back::Red, Stake::White);
	g.step(Action::SelectBlind).unwrap();
	g.areas.hand.cards = hand_of(&[
		(Rank::Ace, Suit::Spades),
		(Rank::King, Suit::Hearts),
		(Rank::Queen, Suit::Clubs),
		(Rank::Jack, Suit::Diamonds),
		(Rank::Nine, Suit::Spades),
		(Rank::Seven, Suit::Hearts),
		(Rank::Five, Suit::Clubs),
		(Rank::Two, Suit::Diamonds),
	]);
	let mut psychic = g.clone();
	psychic.blind.kind = Some(BlindKind::Psychic);
	assert_eq!(
		best_candidate(&psychic),
		best_legal_play(&psychic, Hand::HIGHLIGHT_LIMIT),
		"The Psychic"
	);
	g.inject(Inject::joker(Joker::Splash));
	let best = best_legal_play(&g, Hand::HIGHLIGHT_LIMIT);
	assert_eq!(best_candidate(&g), best, "Splash candidates");
	assert_eq!(obs::best_play(&g), best, "Splash build preview");
}
