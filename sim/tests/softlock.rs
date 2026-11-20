//! The real game's pack softlock (reproduced in the game's Lua): hand size 5
//! (Stuntman, Merry Andy), a Mega Spectral pack, Immolate destroys the whole
//! hand, nothing redraws and skipping needs hand cards.

use sim::actions::ActionError;
use sim::data::{Back, Joker, Spectral};
use sim::inject::Inject;
use sim::{Action, Game, Kind, Phase, Stake};

#[test]
fn immolate_emptying_the_pack_hand_is_a_terminal_softlock() {
	use sim::CardLike;

	let mut g = Game::new_run("H28D9", Back::Red, Stake::White);
	g.inject(Inject::joker(Joker::Stuntman));
	g.inject(Inject::joker(Joker::MerryAndy));
	for code in ["select", "play 12", "cash_out", "open 1"] {
		let action: Action = code.parse().unwrap();
		g.step(action).unwrap();
	}
	assert_eq!((g.areas.hand.cards.len(), g.areas.deck.len()), (5, 47));
	assert_eq!(
		g.areas.pack_cards[0].as_card().map(|c| c.kind),
		Some(Kind::Spectral(Spectral::Immolate))
	);
	let pick: Action = "pick 0 0".parse().unwrap();
	g.step(pick).unwrap();
	assert_eq!(g.flow.phase, Phase::Pack);
	assert_eq!(
		g.areas.hand.cards.len(),
		0,
		"Immolate destroys the whole hand"
	);
	assert_eq!(g.pack.choices, Some(1), "one pick of the Mega pack is left");
	assert_eq!(g.check(Action::SkipPack), Err(ActionError::NotAllowed));
	let mut acts = Vec::new();
	g.legal_actions(&mut acts);
	assert_eq!(acts, [Action::SellJoker(0), Action::SellJoker(1)]);
	assert!(!g.is_terminal());
	for _ in 0..2 {
		g.step(Action::SellJoker(0)).unwrap();
	}
	g.legal_actions(&mut acts);
	assert_eq!(
		acts,
		[],
		"nothing to sell, and no card to skip the pack with"
	);
	assert!(g.is_stuck(), "no legal action in a pack is the softlock");
	assert!(g.is_terminal(), "the softlock ends the run");
}
