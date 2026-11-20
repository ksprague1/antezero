//! Randomized differential test against the real game.
//!
//! Usage: `difftest [games=20] [first_seed=1] [max_steps=300] [back=b_red]
//! [stake=1] [inject_rate=0]`. Game `n` uses seed `T<first_seed + n>`. The
//! checks follow `ORACLE_TIMELINE` / `ORACLE_FRAMES` (see
//! `difftest::Checks`). The history of every diverging game is written to
//! `.scratch/fail_<seed>_<back>_<stake>.txt` in the repository (replay it
//! with `replay`).
//! Exits with status 1 if any game diverged.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;
use std::str::FromStr;
use std::{env, fs};

use oracle::Oracle;
use oracle::command::Start;
use oracle::difftest::{self, Checks, GameSpec};
use sim::Stake;
use sim::data::Back;

const USAGE: &str = "usage: difftest [games] [first_seed] [max_steps] [back] [stake] [inject_rate]";

fn arg<T: FromStr>(a: &[String], i: usize, name: &str, default: T) -> Result<T, String> {
	match a.get(i) {
		None => Ok(default),
		Some(s) => s.parse().map_err(|_| format!("bad {name} {s:?}")),
	}
}

fn run() -> Result<bool, String> {
	let a: Vec<String> = env::args().skip(1).collect();
	if a.len() > 6 {
		return Err("too many arguments".into());
	}
	let games: u64 = arg(&a, 0, "games", 20)?;
	let first: u64 = arg(&a, 1, "first_seed", 1)?;
	let max_steps: usize = arg(&a, 2, "max_steps", 300)?;
	let back = match a.get(3) {
		None => Back::Red,
		Some(k) => Back::from_key(k)
			.filter(|b| b.is_simulated())
			.ok_or_else(|| format!("bad back {k:?}"))?,
	};
	let stake = Stake::try_from(arg::<u8>(&a, 4, "stake", 1)?);
	let inject_rate: f64 = arg(&a, 5, "inject_rate", 0.0)?;
	let (Ok(stake), true) = (stake, (0.0..=1.0).contains(&inject_rate)) else {
		return Err(format!(
			"stake must be in {:?} and inject_rate in [0, 1]",
			sim::STAKES
		));
	};
	if first.checked_add(games).is_none() {
		return Err("first_seed + games overflows".into());
	}
	let checks = Checks::from_env()?;
	let mut o = Oracle::new()?;
	let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.scratch");
	println!("checks: {checks}");
	let (mut ok, mut steps, mut timed, mut frame_mismatches) = (0, 0, 0, 0);
	let mut examples = Vec::new();
	for id in first..first + games {
		let seed = format!("T{id:07}");
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
		let stats = match difftest::run_game(&mut o, &spec, checks) {
			Ok(s) => {
				ok += 1;
				println!(
					"ok   {seed} steps={} ante={} won={}",
					s.steps, s.max_ante, s.won
				);
				s
			}
			Err(f) => {
				print!("FAIL {f}");
				let path = out_dir.join(format!(
					"fail_{seed}_{}_{}.txt",
					back.key(),
					u8::from(stake)
				));
				match fs::create_dir_all(&out_dir)
					.and_then(|()| fs::write(&path, f.codes.join("\n")))
				{
					Ok(()) => println!("  history written to {}", path.display()),
					Err(e) => println!("  could not write the history to {}: {e}", path.display()),
				}
				f.stats
			}
		};
		steps += stats.steps;
		timed += stats.timed_steps;
		frame_mismatches += stats.frame_mismatches;
		examples.extend(stats.frame_examples);
	}
	println!("{ok}/{games} games matched, {steps} steps compared");
	println!(
		"frame-exact timelines: {}/{timed} steps",
		timed - frame_mismatches
	);
	let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
	for e in &examples {
		// A code's first word names its command.
		let command = e
			.code
			.split_once(' ')
			.map_or(e.code.as_str(), |(name, _)| name);
		*counts.entry(command).or_default() += 1;
	}
	for (k, n) in counts {
		println!("  frame mismatch {k}: {n}");
	}
	for e in examples.iter().take(12) {
		println!("    {e}");
	}
	Ok(ok == games)
}

fn main() -> ExitCode {
	match run() {
		Ok(true) => ExitCode::SUCCESS,
		Ok(false) => ExitCode::FAILURE,
		Err(e) => {
			eprintln!("difftest: {e}\n{USAGE}");
			ExitCode::from(2)
		}
	}
}
