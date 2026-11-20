//! The state dump importer against the real game (see `oracle::dumpcheck`;
//! the `dumpcheck` binary runs more games). It needs the game's `lua51.dll`
//! and scripts, so it only runs when asked for:
//! `cargo test -p oracle --test dumpcheck -- --include-ignored`.

use oracle::command::Start;
use oracle::difftest::GameSpec;
use oracle::{Oracle, dumpcheck};
use sim::data::Back;
use sim::{Stake, num};

#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn imported_games_are_the_game() {
	assert!(
		oracle::available(),
		"the oracle needs the game's lua51.dll and scripts (see oracle/src/lib.rs)"
	);
	let mut o = Oracle::new().expect("the harness loads (`oracle::available()` holds)");
	let backs: Vec<Back> = Back::simulated().collect();
	let mut imports = 0;
	for id in 0..8 {
		let seed = format!("I{id:07}");
		let spec = GameSpec {
			start: Start {
				seed: &seed,
				back: backs[num::pick(id * 2, backs.len())],
				stake: Stake::ALL[num::pick(id, Stake::ALL.len())],
			},
			max_steps: 150,
			policy_seed: id.wrapping_mul(7919),
			inject_rate: 0.1,
		};
		let stats =
			dumpcheck::check_game(&mut o, &spec, 0.25).unwrap_or_else(|e| panic!("{seed}: {e}"));
		imports += stats.imports.values().sum::<usize>();
	}
	assert!(imports > 0, "no decision point was imported");
}
