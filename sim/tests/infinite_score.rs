//! An infinite score against an infinite blind (reachable in endless runs)
//! loses: the game ends the round with
//! `G.GAME.chips - G.GAME.blind.chips >= 0`, which is false for the NaN
//! difference.

use sim::data::Back;
use sim::{Action, Game, Phase, Stake};

#[test]
fn an_infinite_score_against_an_infinite_blind_ends_the_run() {
	let mut g = Game::new_run("INFINITE", Back::Red, Stake::White);
	g.step(Action::SelectBlind).unwrap();
	g.blind.chips = f64::INFINITY;
	g.round.chips = f64::INFINITY;
	g.round.hands_left = 1.0;
	let mut acts = Vec::new();
	g.legal_actions(&mut acts);
	let play = acts
		.into_iter()
		.find(|a| matches!(a, Action::Play(_)))
		.unwrap();
	g.step(play).unwrap();
	assert_eq!(g.round.chips, f64::INFINITY);
	assert_eq!(g.flow.phase, Phase::GameOver);
}
