//! Targeted differential scenarios against the real game. They need the
//! game's `lua51.dll` and scripts, so they only run when asked for:
//! `cargo test -p oracle --test scenarios -- --include-ignored`. Each
//! scenario drives a specific mechanic in both engines and compares, after
//! every command, the canonical state, the timeline order and the frame of
//! every timeline entry.

use oracle::Oracle;
use oracle::command::{Command, Start};
use oracle::difftest::{Checks, Comparison, compare};
use oracle::policy::{highest_scoring_play, unused_cards};
use sim::actions::Action;
use sim::data::{Back, BlindKind, Enhancement, Joker, Rank, Suit, TagKind};
use sim::game::{Game, Phase, Scratch};
use sim::inject::Inject;
use sim::{CardMask, Stake, json};

struct Both {
	o: Oracle,
	g: Game,
	log: Vec<String>,
	/// The frame offset this scenario is known to have: how many frames
	/// later the game has the first offset entry of a step. `None`: every
	/// timeline entry must be on the same frame.
	known_frame_offset: Option<i64>,
	/// Steps that had the known frame offset.
	frame_offsets: usize,
	scratch: Scratch,
}

impl Both {
	fn start(start: Start<'_>) -> Both {
		assert!(
			oracle::available(),
			"the oracle needs the game's lua51.dll and scripts (see oracle/src/lib.rs)"
		);
		let mut o = Oracle::new().expect("the harness loads (`oracle::available()` holds)");
		let lua = o
			.command(&start.to_lua())
			.expect("the oracle starts the run");
		let mut b = Both {
			o,
			g: start.new_game(),
			log: vec![start.to_string()],
			known_frame_offset: None,
			frame_offsets: 0,
			scratch: Scratch::default(),
		};
		b.check(&lua);
		b
	}

	fn check(&mut self, lua: &str) {
		let j = json::parse(lua).expect("the oracle returns JSON");
		let checks = if self.known_frame_offset.is_some() {
			Checks::Timeline
		} else {
			Checks::TimelineAndFrames
		};
		let recent = self.log[self.log.len().saturating_sub(6)..].join("\n  ");
		match compare(&mut self.g, &j, checks) {
			Comparison::Match => {}
			Comparison::FrameOffset(offset) => {
				assert_eq!(
					offset.frames(),
					self.known_frame_offset,
					"a frame offset other than the known one after:\n  {recent}\n  {offset}"
				);
				self.frame_offsets += 1;
			}
			Comparison::Diverged(mismatch) => {
				panic!("divergence after:\n  {recent}\n{mismatch}")
			}
		}
	}

	fn run(&mut self, command: Command) {
		self.log.push(command.to_string());
		command
			.apply(&mut self.g)
			.unwrap_or_else(|e| panic!("sim rejected {command:?}: {e}"));
		let lua = self
			.o
			.command(&command.to_lua())
			.expect("the oracle runs the command");
		self.check(&lua);
	}

	fn act(&mut self, a: Action) {
		self.run(Command::Act(a));
	}

	fn inject(&mut self, i: Inject) {
		self.run(Command::Inject(i));
	}

	/// Plays the current round with the best-scoring hands (discarding the
	/// rest when the best hand cannot win); returns whether it was won.
	fn win_round(&mut self) -> bool {
		let mut acts = Vec::new();
		while self.g.flow.phase == Phase::Round {
			self.g.legal_actions(&mut acts);
			let plays = acts.iter().filter_map(|a| match *a {
				Action::Play(m) => Some(m),
				_ => None,
			});
			let Some((best, score)) = highest_scoring_play(&mut self.scratch, &self.g, plays)
				.unwrap_or_else(|m| panic!("the simulator rejected its legal play {m:?}"))
			else {
				return false;
			};
			let need = self.g.blind.chips - self.g.round.chips;
			let mut action = Action::Play(best);
			if score < need && self.g.round.discards_left > 0.0 {
				let d = unused_cards(best, self.g.areas.hand.cards.len());
				if !d.is_empty() && self.g.check(Action::Discard(d)).is_ok() {
					action = Action::Discard(d);
				}
			}
			self.act(action);
		}
		self.g.flow.phase == Phase::RoundEval
	}
}

/// First seed (from a fixed list) whose new run satisfies `ok`.
fn find_seed(ok: impl Fn(&Game) -> bool) -> String {
	(0..20_000)
		.map(|i| format!("SC{i:05}"))
		.find(|s| ok(&Game::new_run(s, Back::Red, Stake::White)))
		.expect("some seed among the first 20,000 has the wanted run")
}

fn red_white(seed: &str) -> Start<'_> {
	Start {
		seed,
		back: Back::Red,
		stake: Stake::White,
	}
}

/// A shop joker created by an Uncommon / Rare tag (`Tag:yep` callback with
/// `start_materialize`).
#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn tag_created_shop_joker() {
	let seed = find_seed(|g| {
		matches!(
			g.blind_select.skip_tags.small,
			Some(TagKind::Uncommon | TagKind::Rare)
		)
	});
	let mut b = Both::start(red_white(&seed));
	for _ in 0..2 {
		b.inject(Inject::joker(Joker::GrosMichel));
	}
	b.act(Action::SkipBlind);
	b.act(Action::SelectBlind);
	assert!(b.win_round(), "big blind not won");
	b.act(Action::CashOut);
	assert_eq!(b.g.flow.phase, Phase::Shop);
	b.act(Action::EndShop);
}

/// An edition tag applied to a tag-created shop joker (`store_joker_modify`
/// on the forced card).
#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn edition_tag_on_tag_created_joker() {
	use sim::CardLike;

	let seed = find_seed(|g| {
		matches!(
			g.blind_select.skip_tags.small,
			Some(TagKind::Uncommon | TagKind::Rare)
		) && matches!(
			g.blind_select.skip_tags.big,
			Some(TagKind::Foil | TagKind::Holo | TagKind::Polychrome | TagKind::Negative)
		) && matches!(
			g.blind_select.boss,
			BlindKind::Club
				| BlindKind::Goad
				| BlindKind::Head
				| BlindKind::Window
				| BlindKind::Pillar
				| BlindKind::Manacle
		)
	});
	let mut b = Both::start(red_white(&seed));
	for _ in 0..3 {
		b.inject(Inject::joker(Joker::GrosMichel));
	}
	b.act(Action::SkipBlind);
	b.act(Action::SkipBlind);
	b.act(Action::SelectBlind);
	assert!(b.win_round(), "boss blind not won");
	b.act(Action::CashOut);
	assert_eq!(b.g.flow.phase, Phase::Shop);
	assert!(
		b.g.areas
			.shop_jokers
			.iter()
			.any(|c| c.core().edition.is_some()),
		"the edition tag did not apply"
	);
}

/// Two Vampires with Midas Mask between them: a card drained by the first
/// Vampire and gilded by Midas is not drained again (`vampired`).
#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn vampire_midas_vampire() {
	let mut b = Both::start(red_white("VAMPIRE1"));
	// The post-scoring draws of the play land 5 frames earlier in the sim
	// than in the game (state and event order match).
	b.known_frame_offset = Some(5);
	for j in [Joker::Vampire, Joker::MidasMask, Joker::Vampire] {
		b.inject(Inject::joker(j));
	}
	for s in Suit::ALL {
		for _ in 0..6 {
			b.inject(Inject::Card {
				suit: s,
				rank: Rank::King,
				enhancement: Enhancement::Steel,
				edition: None,
				seal: None,
			});
		}
	}
	b.act(Action::SelectBlind);
	let kings: CardMask =
		b.g.areas
			.hand
			.cards
			.iter()
			.enumerate()
			.filter(|(_, c)| c.rank == Rank::King)
			.map(|(i, _)| i)
			.take(5)
			.collect();
	assert!(!kings.is_empty(), "no king drawn");
	b.act(Action::Play(kings));
	assert_eq!(
		b.frame_offsets, 1,
		"the play's draws were to be the one step 5 frames early; if they now match, drop `known_frame_offset`"
	);
	let gold =
		b.g.areas
			.discard
			.iter()
			.chain(&b.g.areas.hand.cards)
			.chain(&b.g.areas.deck)
			.filter(|c| c.enhancement == Enhancement::Gold)
			.count();
	assert!(gold > 0, "Midas Mask gilded nothing");
}
