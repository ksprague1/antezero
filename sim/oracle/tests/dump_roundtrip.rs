//! A state dump written by the simulator (`Game::to_dump`) reads back as the
//! same game (`Game::from_dump`), and reads as the game's own dump would
//! (`canon::from_lua`), at every decision point of seeded games over every
//! deck and stake. No game needed: `dumpcheck` checks the reader against the
//! real game's dumps.

use oracle::canon;
use oracle::dumpcheck::part_diffs;
use oracle::fingerprint::obs_fingerprint;
use oracle::policy::Policy;
use sim::data::Back;
use sim::game::Game;
use sim::{Stake, json, num};

const GAMES: u64 = 60;
const STEPS: usize = 250;

fn check(g: &mut Game, n: u64, step: usize) {
	let text = g.to_dump().to_string();
	let dump = json::parse(&text).expect("a written dump parses");
	let d = canon::diff(&canon::from_sim(g), &canon::from_lua(&dump));
	assert!(
		d.is_empty(),
		"game {n} step {step}: the dump reads as {d:?}"
	);
	let mut back = Game::from_dump(&dump).unwrap_or_else(|e| panic!("game {n} step {step}: {e}"));
	back.flow.rearranges = g.flow.rearranges;
	let parts = part_diffs(&mut back, g);
	assert!(parts.is_empty(), "game {n} step {step}: {parts:?}");
	assert_eq!(
		obs_fingerprint(&mut back),
		obs_fingerprint(g),
		"game {n} step {step}: the observation differs"
	);
}

#[test]
fn a_written_dump_reads_back_as_the_same_game() {
	let backs: Vec<Back> = Back::simulated().collect();
	for n in 0..GAMES {
		let back = backs[num::pick(n, backs.len())];
		let stake = Stake::ALL[num::pick(n / 3, Stake::ALL.len())];
		let mut g = Game::new_run(&format!("DUMP{n:04}"), back, stake);
		let mut policy = Policy::for_game(n, 0.1);
		check(&mut g, n, 0);
		for step in 1..=STEPS {
			let Some(command) = policy.next(&g) else {
				break;
			};
			command
				.apply(&mut g)
				.unwrap_or_else(|e| panic!("game {n}: {command:?} rejected: {e}"));
			check(&mut g, n, step);
		}
	}
}
