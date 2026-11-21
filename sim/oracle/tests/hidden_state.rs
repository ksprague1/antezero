//! The learning observation reads nothing a player cannot know: with its
//! hidden state resampled (`Game::resample_hidden`: deck order, face-down
//! cards, every future random outcome, an unseen voucher) a game gives the
//! same observation and ops, play candidates and build previews included.
//! A game read from the real game's state dump holds the game's true hidden
//! state, so the policy must not read it.

use oracle::fingerprint::obs_fingerprint;
use oracle::policy::Policy;
use sim::data::Back;
use sim::game::Game;
use sim::{Stake, num};

const GAMES: u64 = 30;
const STEPS: usize = 200;

#[test]
fn the_observation_reads_no_hidden_state() {
	let backs: Vec<Back> = Back::simulated().collect();
	for n in 0..GAMES {
		let back = backs[num::pick(n, backs.len())];
		let stake = Stake::ALL[num::pick(n / 2, Stake::ALL.len())];
		let mut g = Game::new_run(&format!("HIDE{n:04}"), back, stake);
		let mut policy = Policy::for_game(n, 0.1);
		for step in 0..STEPS {
			let seen = obs_fingerprint(&mut g);
			for world in 0..2 {
				let mut w = g.clone();
				w.resample_hidden(&format!("{n}_{step}_{world}"));
				assert_eq!(
					obs_fingerprint(&mut w),
					seen,
					"game {n} step {step}: the observation changed with the hidden state"
				);
			}
			let Some(command) = policy.next(&g) else {
				break;
			};
			command
				.apply(&mut g)
				.unwrap_or_else(|e| panic!("game {n}: {command:?} rejected: {e}"));
		}
	}
}
