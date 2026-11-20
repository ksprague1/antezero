use sim::data::{Back, Edition, Enhancement, Joker, Planet, Rank, Seal, Spectral, Suit, Voucher};
use sim::game::{Game, HandSort, Phase};
use sim::inject::{Consumable, Inject};
use sim::{Action, CardMask, Stake};

#[test]
fn every_legal_action_round_trips() {
	let mut g = Game::new_run("CODES", Back::Red, Stake::White);
	let mut acts = Vec::new();
	let mut n = 0;
	for step in 0..400 {
		if g.is_terminal() {
			break;
		}
		g.legal_actions(&mut acts);
		for a in &acts {
			let code = a.to_string();
			assert_eq!(code.parse::<Action>(), Ok(*a), "{code}");
			n += 1;
		}
		let a = if g.flow.phase == Phase::Round {
			*acts.iter().find(|a| matches!(a, Action::Play(_))).unwrap()
		} else {
			acts[step % acts.len()]
		};
		g.step(a).unwrap();
	}
	for a in [
		Action::MoveJoker { from: 0, to: 3 },
		Action::MoveHand { from: 7, to: 1 },
		Action::SortHand(HandSort::Suit),
	] {
		assert_eq!(a.to_string().parse::<Action>(), Ok(a));
	}
	assert!(n > 1000, "only {n} codes checked");
}

#[test]
fn malformed_action_codes_are_rejected() {
	for bad in [
		"",
		"play",
		"play x",
		"select now",
		"play 3 4",
		"use 0",
		"sell_joker 256",
		"move_joker 1 999",
		"sort 2",
		"buy -1",
		"jump",
	] {
		assert!(bad.parse::<Action>().is_err(), "{bad:?} parsed");
	}
	assert_eq!(
		"  play   30 ".parse::<Action>(),
		Ok(Action::Play(CardMask::from_bits(30)))
	);
}

#[test]
fn inject_codes_round_trip_and_reject_malformed_input() {
	let samples = [
		Inject::Joker {
			joker: Joker::Blueprint,
			edition: Some(Edition::Negative),
			eternal: true,
		},
		Inject::Consumable {
			card: Consumable::Planet(Planet::Eris),
			edition: None,
		},
		Inject::Consumable {
			card: Consumable::Spectral(Spectral::Ankh),
			edition: Some(Edition::Polychrome),
		},
		Inject::Card {
			suit: Suit::Clubs,
			rank: Rank::Queen,
			enhancement: Enhancement::Lucky,
			edition: Some(Edition::Holo),
			seal: Some(Seal::Purple),
		},
		Inject::Card {
			suit: Suit::Hearts,
			rank: Rank::Ace,
			enhancement: Enhancement::Base,
			edition: None,
			seal: None,
		},
		Inject::Dollars(17.5),
		Inject::Voucher(Voucher::Palette),
	];
	for inject in samples {
		let code = inject.to_string();
		let back: Inject = code.parse().unwrap();
		assert_eq!(back, inject, "{code}");
	}
	for bad in [
		"inject_joker",
		"inject_joker 150 0 0",
		"inject_joker 1 5 0",
		"inject_joker 1 0 0 9",
		"inject_cons 3 0 0",
		"inject_cons 0 22 0",
		"inject_card 4 2 0 0 0",
		"inject_card 0 15 0 0 0",
		"inject_card 0 2 9 0 0",
		"inject_card 0 2 0 0 5",
		"inject_dollars junk",
		"inject_dollars inf",
		"inject_voucher 32",
		"inject_tag 1",
	] {
		assert!(bad.parse::<Inject>().is_err(), "{bad:?} parsed");
	}
}
