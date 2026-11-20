//! Tests of the learning interface (`sim::obs`).

mod candidates;
mod capacity;
mod encoding;
mod order;
mod preview_cache;
mod previews;

use std::fmt;

use sim::data::{Back, Rank, Suit};
use sim::game::{Game, Phase};
use sim::obs::{self, ObsBuf, TokInts};
use sim::rng::Xorshift;
use sim::{Action, PlayingCard, Stake, num};

/// The seed of the walks' random choices.
const WALK_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

/// Panics with the first differing token / global if `a != b`.
fn assert_same(a: &ObsBuf, b: &ObsBuf, what: &str) {
	first_diff(a, "tok_i", &a.tok_i, &b.tok_i, Some(obs::TOK_INT), what);
	first_diff(a, "tok_f", &a.tok_f, &b.tok_f, Some(obs::TOK_FLOAT), what);
	first_diff(a, "glob_i", &a.glob_i, &b.glob_i, None, what);
	first_diff(a, "glob_f", &a.glob_f, &b.glob_f, None, what);
}

/// Panics at the first index where `a` and `b` differ. Token arrays have
/// `row_width` features per token, and the message shows the categorical
/// features `buf` has for the token.
fn first_diff<T: PartialEq + fmt::Debug>(
	buf: &ObsBuf, name: &str, a: &[T], b: &[T], row_width: Option<usize>, what: &str,
) {
	let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) else {
		return;
	};
	let (feature, row) = match row_width {
		Some(width) => (
			i % width,
			format!(" token {} {:?}", i / width, tok(buf, i / width)),
		),
		None => (i, String::new()),
	};
	panic!(
		"{what}: {name}[{i}] (feature {feature}){row}: {:?} != {:?}",
		a[i], b[i]
	);
}

/// Writes the observation of `g` (with deck tokens and a fresh build
/// preview) into `buf`; returns its layout.
fn encode_with_deck(g: &Game, buf: &mut ObsBuf) -> obs::Layout {
	obs::encode_opts(g, &mut buf.as_mut(), /* deck_tokens= */ true)
}

/// The observation of `g` (see [`encode_with_deck`]).
fn observe(g: &Game) -> ObsBuf {
	let mut b = ObsBuf::new();
	encode_with_deck(g, &mut b);
	b
}

/// Plays `g` for up to `steps` decisions or until the run ends, taking the
/// action `turn` returns for each state (which it may change first).
fn walk_with(g: &mut Game, steps: usize, mut turn: impl FnMut(&mut Game) -> Action) {
	for _ in 0..steps {
		if g.is_terminal() {
			break;
		}
		let a = turn(g);
		g.step(a).expect("a turn returns a legal action");
	}
}

/// Plays a game with a simple seeded policy, calling `f` at every decision.
fn walk(seed: &str, steps: usize, mut f: impl FnMut(&Game)) {
	let mut g = Game::new_run(seed, Back::Red, Stake::White);
	let (mut rng, mut acts) = (Xorshift::new(WALK_SEED), Vec::new());
	walk_with(&mut g, steps, |g| {
		f(g);
		g.legal_actions(&mut acts);
		let x = rng.next_u64();
		// Prefer progress: play more often than anything else in rounds.
		let plays: Vec<Action> = acts
			.iter()
			.copied()
			.filter(|a| matches!(a, Action::Play(_)))
			.collect();
		if g.flow.phase == Phase::Round && !x.is_multiple_of(4) && !plays.is_empty() {
			plays[num::pick(x >> 8, plays.len())]
		} else {
			acts[num::pick(x >> 8, acts.len())]
		}
	});
}

/// Like `walk`, but plays the best candidate in rounds (so the games
/// reach shops, packs and later antes) and acts at random elsewhere.
fn walk_build(seed: &str, steps: usize, mut f: impl FnMut(&Game)) {
	let mut g = Game::new_run(seed, Back::Red, Stake::White);
	let (mut rng, mut acts) = (Xorshift::new(WALK_SEED), Vec::new());
	walk_with(&mut g, steps, |g| {
		f(g);
		let x = rng.next_u64();
		match best_round_play(g) {
			Some(play) if !x.is_multiple_of(7) => play,
			_ => random_action(g, x, &mut acts),
		}
	});
}

/// The legal action of `g` the random number `x` picks.
fn random_action(g: &Game, x: u64, acts: &mut Vec<Action>) -> Action {
	g.legal_actions(acts);
	acts[num::pick(x >> 8, acts.len())]
}

/// The best scored candidate, in a round.
fn best_round_play(g: &Game) -> Option<Action> {
	if g.flow.phase != Phase::Round {
		return None;
	}
	obs::top_plays(g, 1).first().map(|c| Action::Play(c.mask))
}

/// The categorical features of token `i` of `b`.
fn tok(b: &ObsBuf, i: usize) -> TokInts {
	TokInts::from_array(b.tok_i.as_chunks::<{ obs::TOK_INT }>().0[i])
}

/// Observation and ops of `g` (with 4 candidates, the build preview and the
/// joker-order gains).
fn obs_and_ops(g: &Game) -> (ObsBuf, Vec<obs::Op>) {
	let mut b = ObsBuf::new();
	let lay = encode_with_deck(g, &mut b);
	let mut ops = Vec::new();
	obs::ops(
		g,
		&lay,
		obs::OpOptions {
			candidates: 4,
			..Default::default()
		},
		&mut ops,
	);
	obs::order_gain(g, &mut ops);
	(b, ops)
}

fn hand_of(cards: &[(Rank, Suit)]) -> Vec<PlayingCard> {
	cards
		.iter()
		.enumerate()
		.map(|(i, &(rank, suit))| PlayingCard::new(100 + num::index32(i), suit, rank))
		.collect()
}

/// The best score of any legal play of at most `max_cards` cards (every
/// such play simulated).
fn best_legal_play(g: &Game, max_cards: usize) -> f64 {
	let mut acts = Vec::new();
	g.legal_actions(&mut acts);
	let plays = acts.iter().filter_map(|a| match *a {
		Action::Play(m) if m.len() <= max_cards => Some(m),
		_ => None,
	});
	plays
		.filter_map(|m| g.clone().preview_play(m).ok())
		.fold(0.0, f64::max)
}

fn best_candidate(g: &Game) -> f64 {
	obs::top_plays(g, 8)
		.iter()
		.map(|c| c.mean)
		.fold(0.0, f64::max)
}

/// The games of the seeds `{prefix}{from}`, `{prefix}{from + 1}`, ... at
/// their first shop; the seeds whose best-candidate play does not get
/// there are left out.
fn first_shops(prefix: &str, from: u64) -> impl Iterator<Item = Game> {
	(from..).filter_map(move |s| {
		let mut g = Game::new_run(&format!("{prefix}{s}"), Back::Red, Stake::White);
		g.step(Action::SelectBlind)
			.expect("a new run starts at blind select");
		while matches!(g.flow.phase, Phase::Round | Phase::RoundEval) {
			let a = if g.flow.phase == Phase::Round {
				best_round_play(&g).expect("a round has a scored candidate")
			} else {
				Action::CashOut
			};
			g.step(a)
				.expect("a candidate play and cashing out are legal");
		}
		(g.flow.phase == Phase::Shop).then_some(g)
	})
}

/// The first of [`first_shops`].
fn first_shop(prefix: &str, from: u64) -> Game {
	first_shops(prefix, from)
		.next()
		.expect("best-candidate play reaches a shop on most seeds")
}
