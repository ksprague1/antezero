//! Throughput benchmark: `cargo run --release --example bench [seconds]`.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::env;
use std::hint::black_box;
use std::time::Instant;

use sim::data::{Back, Joker};
use sim::game::{Game, Hand, Phase, Scratch};
use sim::inject::Inject;
use sim::rng::{PseudoRandom, Xorshift};
use sim::{Action, CardMask, PlayingCard, Stake, hand_eval, num, obs};

/// Spreads a game's number into its policy's seed.
const SEED_MIX: u64 = 2_654_435_761;

fn random_game(seed: u64, buf: &mut Vec<Action>) -> usize {
	let mut g = Game::new_run(&format!("R{seed}"), Back::Red, Stake::White);
	let mut x = Xorshift::new(seed.wrapping_mul(SEED_MIX));
	let mut n = 0;
	while !g.is_terminal() && n < 2000 {
		g.legal_actions(buf);
		let a = buf[x.below(buf.len())];
		g.step(a).expect("a legal action steps");
		n += 1;
	}
	n
}

/// Greedy-ish policy; returns (steps, plays evaluated by lookahead, final ante).
fn greedy_game(seed: u64, buf: &mut Vec<Action>) -> (usize, usize, i32) {
	let mut g = Game::new_run(&format!("G{seed}"), Back::Red, Stake::White);
	let mut x = Xorshift::new(seed.wrapping_mul(SEED_MIX));
	let (mut n, mut looked) = (0, 0);
	let is_buy = |a: &&Action| matches!(a, Action::BuyShopCard(_));
	while !g.is_terminal() && n < 3000 {
		g.legal_actions(buf);
		let a = match g.flow.phase {
			Phase::Round => {
				let mut best: Option<(Action, f64)> = None;
				for &a in buf.iter().filter(|a| matches!(a, Action::Play(_))) {
					let mut c = g.clone();
					c.step(a).expect("a legal play steps");
					looked += 1;
					let score = c.run.last_hand_score;
					if best.is_none_or(|(_, b)| score > b) {
						best = Some((a, score));
					}
				}
				best.expect("a round always has a legal play").0
			}
			Phase::Shop => {
				let buys = buf.iter().filter(is_buy).count();
				if buys > 0 && x.below(2) == 0 {
					*buf.iter()
						.filter(is_buy)
						.nth(x.below(buys))
						.expect("the index is below the number of buys")
				} else {
					Action::EndShop
				}
			}
			Phase::BlindSelect => Action::SelectBlind,
			Phase::RoundEval => Action::CashOut,
			_ => buf[x.below(buf.len())],
		};
		g.step(a).expect("a legal action steps");
		n += 1;
	}
	(n, looked, g.run.ante)
}

/// Seconds since `since`, as microseconds per item.
fn us_per(since: Instant, items: usize) -> f64 {
	since.elapsed().as_secs_f64() * 1e6 / num::float(items)
}

fn time_random_games(secs: f64, buf: &mut Vec<Action>) {
	let start = Instant::now();
	let (mut games, mut steps) = (0u64, 0usize);
	while start.elapsed().as_secs_f64() < secs {
		steps += random_game(games, buf);
		games += 1;
	}
	let us = us_per(start, steps);
	println!(
		"random : {games} games, {steps} steps, {:.0} steps/s, {us:.1} us/step (incl. new_run)",
		1e6 / us
	);
}

fn time_greedy_games(secs: f64, buf: &mut Vec<Action>) {
	let start = Instant::now();
	let (mut games, mut steps, mut looked, mut antes) = (0u32, 0usize, 0usize, 0.0);
	while start.elapsed().as_secs_f64() < secs {
		let (game_steps, game_looked, ante) = greedy_game(u64::from(games), buf);
		steps += game_steps;
		looked += game_looked;
		antes += f64::from(ante);
		games += 1;
	}
	println!(
		"greedy : {games} games (avg ante {:.2}), {steps} steps + {looked} lookahead plays; {:.1} us per simulated step (a clone and a play per lookahead, plus the steps)",
		antes / f64::from(games),
		us_per(start, looked + steps)
	);
}

/// Clone, legal actions, and one play / discard on a clone.
fn time_basics(g: &Game, buf: &mut Vec<Action>) {
	let n = 200_000;
	let start = Instant::now();
	for _ in 0..n {
		black_box(g.clone());
	}
	println!("clone  : {:.0} ns", us_per(start, n) * 1e3);
	let start = Instant::now();
	for _ in 0..n {
		g.legal_actions(buf);
		black_box(&buf);
	}
	println!(
		"legal  : {:.0} ns ({} actions)",
		us_per(start, n) * 1e3,
		buf.len()
	);
	let play = *buf
		.iter()
		.find(|a| matches!(a, Action::Play(_)))
		.expect("the first hand can be played");
	let discard = *buf
		.iter()
		.find(|a| matches!(a, Action::Discard(_)))
		.expect("the first hand can be discarded");
	let n = 50_000;
	for (label, what, action) in [("play   ", "hand", play), ("discard", "discard", discard)] {
		let start = Instant::now();
		for _ in 0..n {
			let mut c = g.clone();
			c.step(black_box(action)).expect("the action is legal");
			black_box(&c);
		}
		println!("{label}: {:.2} us (clone + one {what})", us_per(start, n));
	}
}

/// `g` with five jokers that make every play chance-dependent.
fn with_jokers(mut g: Game) -> Game {
	for joker in [
		Joker::Misprint,
		Joker::Bloodstone,
		Joker::Blueprint,
		Joker::Hiker,
		Joker::Photograph,
	] {
		g.inject(Inject::joker(joker));
	}
	g
}

/// Scored play candidates (the best `PREVIEW_BUDGET` plays by a cheap
/// estimate are previewed; chance-dependent ones over several scenarios).
fn time_top_plays(g: &Game, what: &str) {
	let reps = 200;
	let start = Instant::now();
	let mut total = 0;
	for _ in 0..reps {
		total += black_box(obs::top_plays(g, 8)).len();
	}
	println!(
		"top8   : {:.1} us per decision ({what})",
		us_per(start, reps)
	);
	black_box(total);
}

/// Hand evaluation of every play, and one preview.
fn time_eval_and_preview(g: &Game, plays: &[CardMask], alt: &PseudoRandom) {
	let reps = 200;
	let flags = g.areas.eval_flags();
	let hand = &g.areas.hand.cards;
	let mut cards: Vec<&PlayingCard> = Vec::with_capacity(Hand::HIGHLIGHT_LIMIT);
	let start = Instant::now();
	let mut sink = 0usize;
	for _ in 0..reps {
		for &mask in plays {
			cards.clear();
			cards.extend(mask.indices().map(|i| &hand[i]));
			let top = black_box(hand_eval::evaluate(&cards, flags)).top;
			sink += top.map_or(0, |top| usize::from(top.mask));
		}
	}
	println!(
		"eval   : {:.1} us for all {} plays",
		us_per(start, reps),
		plays.len()
	);
	let previewed = plays.len().min(64);
	let mut scratch = Scratch::default();
	let start = Instant::now();
	for _ in 0..reps {
		for &mask in &plays[..previewed] {
			sink +=
				usize::from(black_box(obs::preview_score(&mut scratch, g, mask, alt)).is_some());
		}
	}
	println!("preview: {:.2} us each", us_per(start, reps * previewed));
	black_box(sink);
}

fn main() {
	let secs: f64 = env::args().nth(1).map_or(3.0, |s| {
		s.parse().expect("the argument is a number of seconds")
	});
	let mut buf = Vec::with_capacity(512);
	time_random_games(secs, &mut buf);
	time_greedy_games(secs, &mut buf);

	// Micro: the first round, hand drawn.
	let mut g = Game::new_run("MICRO", Back::Red, Stake::White);
	g.step(Action::SelectBlind)
		.expect("a new run starts at blind select");
	time_basics(&g, &mut buf);
	g.legal_actions(&mut buf);
	let plays: Vec<CardMask> = buf
		.iter()
		.filter_map(|a| {
			if let Action::Play(mask) = *a {
				Some(mask)
			} else {
				None
			}
		})
		.collect();
	let jokered = with_jokers(g.clone());
	time_top_plays(&g, &format!("{} legal plays, no jokers", plays.len()));
	time_top_plays(
		&jokered,
		"5 jokers incl. Misprint: every play is chance-dependent",
	);
	let alt = PseudoRandom::new("X");
	time_eval_and_preview(&g, &plays, &alt);
	let mut scratch = Scratch::default();
	for (name, game) in [("no jokers", &g), ("5 jokers", &jokered)] {
		let chance = plays
			.iter()
			.filter(|&&mask| {
				obs::preview_score(&mut scratch, game, mask, &alt).is_some_and(|p| p.chance)
			})
			.count();
		println!(
			"chance : {chance}/{} plays draw randomness ({name})",
			plays.len()
		);
	}
	let n = 20_000;
	let seeds: Vec<String> = (0..n).map(|i| format!("N{i}")).collect();
	let start = Instant::now();
	for seed in &seeds {
		black_box(Game::new_run(seed, Back::Red, Stake::White));
	}
	println!("new_run: {:.2} us", us_per(start, n));
}
