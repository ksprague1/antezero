//! The state dump importer checked against the real game (see
//! `oracle::dumpcheck`).
//!
//! Usage: `dumpcheck [games=30] [first_seed=1] [max_steps=400]
//! [import_rate=0.1] [inject_rate=0.1]`. Game `n` uses seed `D<first_seed +
//! n>`, the `n`-th deck (of all of them, in turn) and the `n`-th stake.
//! Exits with status 1 if any check failed.

use std::collections::BTreeMap;
use std::env;
use std::process::ExitCode;
use std::str::FromStr;

use oracle::command::Start;
use oracle::difftest::GameSpec;
use oracle::{Oracle, dumpcheck};
use sim::data::Back;
use sim::{Stake, num};

const USAGE: &str = "usage: dumpcheck [games] [first_seed] [max_steps] [import_rate] [inject_rate]";

fn arg<T: FromStr>(a: &[String], i: usize, name: &str, default: T) -> Result<T, String> {
	match a.get(i) {
		None => Ok(default),
		Some(s) => s.parse().map_err(|_| format!("bad {name} {s:?}")),
	}
}

fn run() -> Result<bool, String> {
	let a: Vec<String> = env::args().skip(1).collect();
	if a.len() > 5 {
		return Err("too many arguments".into());
	}
	let games: u64 = arg(&a, 0, "games", 30)?;
	let first: u64 = arg(&a, 1, "first_seed", 1)?;
	let max_steps: usize = arg(&a, 2, "max_steps", 400)?;
	let import_rate: f64 = arg(&a, 3, "import_rate", 0.1)?;
	let inject_rate: f64 = arg(&a, 4, "inject_rate", 0.1)?;
	if !(0.0..=1.0).contains(&import_rate) || !(0.0..=1.0).contains(&inject_rate) {
		return Err("import_rate and inject_rate must be in [0, 1]".into());
	}
	let backs: Vec<Back> = Back::simulated().collect();
	let mut o = Oracle::new()?;
	let (mut ok, mut steps, mut followed) = (0, 0, 0);
	let mut imports: BTreeMap<&str, usize> = BTreeMap::new();
	for id in first..first + games {
		let seed = format!("D{id:07}");
		let back = backs[num::pick(id, backs.len())];
		let stake = Stake::ALL[num::pick(id, Stake::ALL.len())];
		let spec = GameSpec {
			start: Start {
				seed: &seed,
				back,
				stake,
			},
			max_steps,
			policy_seed: id.wrapping_mul(7919),
			inject_rate,
		};
		let at = format!("{seed} {} stake {}", back.key(), u8::from(stake));
		match dumpcheck::check_game(&mut o, &spec, import_rate) {
			Ok(s) => {
				ok += 1;
				let n: usize = s.imports.values().sum();
				println!(
					"ok   {at}: {} steps, ante {}, {n} imports, {} followed steps",
					s.steps, s.max_ante, s.followed
				);
				steps += s.steps;
				followed += s.followed;
				for (phase, n) in s.imports {
					*imports.entry(phase).or_default() += n;
				}
			}
			Err(e) => println!("FAIL {at}: {e}"),
		}
	}
	let n: usize = imports.values().sum();
	println!(
		"{ok}/{games} games passed: {steps} steps, {n} imports ({imports:?}), {followed} followed steps"
	);
	Ok(ok == games)
}

fn main() -> ExitCode {
	match run() {
		Ok(true) => ExitCode::SUCCESS,
		Ok(false) => ExitCode::FAILURE,
		Err(e) => {
			eprintln!("dumpcheck: {e}\n{USAGE}");
			ExitCode::from(2)
		}
	}
}
