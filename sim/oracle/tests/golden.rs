//! Golden trajectories: the simulator alone (no game DLL needed) plays a
//! fixed set of seeded games across every deck and stake, with random
//! scenario injections, and every step is fingerprinted: the canonical state
//! (including every RNG stream), the action's timeline (RNG calls, draws,
//! removals, with frame numbers) and the learning observation with its ops.
//! Any behaviour change shows up as a mismatch, so refactors can be checked
//! in seconds; the oracle difftest remains the ground truth for parity.
//!
//! The simulator part (state + timeline, `golden_sim.txt`) and the learning
//! interface part (observation + ops, `golden_obs.txt`) are recorded
//! separately, so an observation change cannot hide a simulator change.
//! Re-record after an intended change with `GOLDEN_BLESS=obs` (observation
//! only; the simulator must still match) or `GOLDEN_BLESS=all`.

use std::env::{self, VarError};
use std::fs;
use std::path::Path;

use oracle::fingerprint::{Fingerprint, fingerprint};
use oracle::policy::Policy;
use sim::data::Back;
use sim::game::Game;
use sim::{Stake, num};

const GAMES: u64 = 200;
const STEPS: usize = 300;

fn play(n: u64) -> Vec<Fingerprint> {
	let backs: Vec<Back> = Back::simulated().collect();
	let back = backs[num::pick(n, backs.len())];
	let stake = Stake::ALL[num::pick(n, Stake::ALL.len())];
	let mut g = Game::new_run(&format!("GOLD{n:04}"), back, stake);
	let mut policy = Policy::for_game(n, 0.1);
	let mut out = vec![fingerprint(&mut g, "start")];
	for _ in 0..STEPS {
		let Some(command) = policy.next(&g) else {
			break;
		};
		command
			.apply(&mut g)
			.unwrap_or_else(|e| panic!("game {n}: {command:?} rejected: {e}"));
		out.push(fingerprint(&mut g, &command.to_string()));
	}
	out
}

/// A golden file.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Part {
	Sim,
	Obs,
}

impl Part {
	fn file(self) -> &'static str {
		match self {
			Part::Sim => "golden_sim.txt",
			Part::Obs => "golden_obs.txt",
		}
	}

	fn of(self, f: Fingerprint) -> u64 {
		match self {
			Part::Sim => f.sim,
			Part::Obs => f.obs,
		}
	}
}

/// What `GOLDEN_BLESS` re-records.
#[derive(Clone, Copy)]
enum Bless {
	Nothing,
	Obs,
	All,
}

impl Bless {
	fn from_env() -> Bless {
		match env::var("GOLDEN_BLESS") {
			Err(VarError::NotPresent) => Bless::Nothing,
			Ok(v) if v.is_empty() => Bless::Nothing,
			Ok(v) if v == "obs" => Bless::Obs,
			Ok(v) if v == "all" => Bless::All,
			other => panic!("GOLDEN_BLESS must be `obs` or `all`, got {other:?}"),
		}
	}

	fn records(self, part: Part) -> bool {
		match self {
			Bless::Nothing => false,
			Bless::Obs => part == Part::Obs,
			Bless::All => true,
		}
	}
}

/// Compares `part` of `games` (one fingerprint list per game) with its
/// recording, or re-records it; returns the diverging games.
fn check(part: Part, games: &[Vec<Fingerprint>], bless: Bless) -> Vec<String> {
	use std::fmt::Write;

	let file = part.file();
	let path = Path::new(env!("CARGO_MANIFEST_DIR"))
		.join("tests")
		.join(file);
	let mut text = String::new();
	for (i, v) in games.iter().enumerate() {
		let hashes: Vec<String> = v.iter().map(|&f| format!("{:016x}", part.of(f))).collect();
		writeln!(text, "{i} {}", hashes.join(" ")).expect("writing to a String cannot fail");
	}
	if bless.records(part) {
		fs::write(&path, &text).expect("the golden file is writable");
		eprintln!(
			"recorded {file}: {} games, {} steps",
			games.len(),
			games.iter().map(Vec::len).sum::<usize>()
		);
		return Vec::new();
	}
	let want = fs::read_to_string(&path)
		.unwrap_or_else(|_| panic!("tests/{file} missing; run with GOLDEN_BLESS=all"));
	let mut bad = Vec::new();
	for (i, (got, exp)) in text.lines().zip(want.lines()).enumerate() {
		if got != exp {
			let a: Vec<&str> = got.split(' ').skip(1).collect();
			let b: Vec<&str> = exp.split(' ').skip(1).collect();
			let at = (0..a.len().max(b.len()))
				.find(|&k| a.get(k) != b.get(k))
				.map_or("its game number".to_string(), |step| format!("step {step}"));
			bad.push(format!(
				"{file}: game {i} at {at} (of {} vs {} recorded)",
				a.len(),
				b.len()
			));
		}
	}
	if text.lines().count() != want.lines().count() {
		bad.push(format!("{file}: number of games changed"));
	}
	bad
}

#[test]
fn golden_trajectories() {
	let bless = Bless::from_env();
	let games: Vec<Vec<Fingerprint>> = (0..GAMES).map(play).collect();
	let mut bad = check(Part::Sim, &games, bless);
	bad.extend(check(Part::Obs, &games, bless));
	assert!(
		bad.is_empty(),
		"{} games diverge from the golden trajectories: {}",
		bad.len(),
		bad.join("; ")
	);
}
