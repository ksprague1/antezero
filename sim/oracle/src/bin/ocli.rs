//! Tiny CLI for poking the oracle: each argument is a Lua expression
//! evaluated by the harness (e.g. `H.start('SEED','b_red',1)`); results go to
//! stdout and errors to stderr.
//!
//! With no arguments it reads one expression per line from stdin and answers
//! each with its result or with `ERROR: <message>` (newlines flattened), so
//! other programs can drive it over a pipe: one line per answer as long as
//! the results are one line (the harness' JSON dumps are).
//!
//! Exits with status 1 if any expression failed.

use std::io::BufRead;
use std::process::ExitCode;
use std::{env, io};

fn main() -> ExitCode {
	let mut o = match oracle::Oracle::new() {
		Ok(o) => o,
		Err(e) => {
			eprintln!("oracle init failed: {e}");
			return ExitCode::from(2);
		}
	};
	let mut failed = false;
	let args: Vec<String> = env::args().skip(1).collect();
	if args.is_empty() {
		for line in io::stdin().lock().lines() {
			match line {
				Ok(l) if l.trim().is_empty() => {}
				Ok(l) => match o.command(&l) {
					Ok(s) => println!("{s}"),
					Err(e) => {
						println!("ERROR: {}", e.replace(['\r', '\n'], " | "));
						failed = true;
					}
				},
				Err(e) => {
					eprintln!("stdin: {e}");
					return ExitCode::from(2);
				}
			}
		}
	} else {
		for a in &args {
			match o.command(a) {
				Ok(s) => println!("{s}"),
				Err(e) => {
					eprintln!("ERROR: {e}");
					failed = true;
				}
			}
		}
	}
	if failed {
		ExitCode::FAILURE
	} else {
		ExitCode::SUCCESS
	}
}
