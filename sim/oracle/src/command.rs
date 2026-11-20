//! The commands both engines run, as the simulator's codes and as the Lua
//! the harness evaluates.

use std::fmt;
use std::str::FromStr;

use sim::actions::{Action, ActionError};
use sim::data::{Back, Edition};
use sim::game::{Game, HandSort};
use sim::inject::Inject;
use sim::{CardMask, Kind, MAX_SEED_LEN, Stake};

use crate::lua_str;

/// The command that starts a run; its code is `start <seed> <back> <stake>`,
/// the first line of a recorded history.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Start<'seed> {
	pub seed: &'seed str,
	pub back: Back,
	pub stake: Stake,
}

impl<'seed> Start<'seed> {
	/// Parses a start code.
	///
	/// # Errors
	///
	/// Returns what is malformed.
	pub fn parse(code: &'seed str) -> Result<Start<'seed>, String> {
		let mut words = code.split_whitespace();
		let (Some("start"), Some(seed), Some(back), Some(stake), None) = (
			words.next(),
			words.next(),
			words.next(),
			words.next(),
			words.next(),
		) else {
			return Err(format!(
				"bad start {code:?} (expected `start <seed> <back> <stake>`)"
			));
		};
		if seed.len() > MAX_SEED_LEN {
			return Err(format!("seed {seed:?} is longer than {MAX_SEED_LEN} bytes"));
		}
		let back = Back::from_key(back).ok_or_else(|| format!("unknown back {back:?}"))?;
		let stake: u8 = stake.parse().map_err(|_| format!("bad stake {stake:?}"))?;
		let stake = Stake::try_from(stake).map_err(|e| e.to_string())?;
		Ok(Start { seed, back, stake })
	}

	/// The run in the simulator.
	///
	/// # Panics
	///
	/// If the seed is longer than [`MAX_SEED_LEN`] bytes.
	#[must_use]
	pub fn new_game(&self) -> Game {
		Game::new_run(self.seed, self.back, self.stake)
	}

	/// The run's start for the harness.
	#[must_use]
	pub fn to_lua(&self) -> String {
		format!(
			"H.start({},{},{})",
			lua_str(self.seed),
			lua_str(self.back.key()),
			u8::from(self.stake)
		)
	}
}

impl fmt::Display for Start<'_> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(
			f,
			"start {} {} {}",
			self.seed,
			self.back.key(),
			u8::from(self.stake)
		)
	}
}

/// A command of a run in progress: an action, or a scenario injection. Its
/// code is the action's or the injection's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Command {
	Act(Action),
	Inject(Inject),
}

impl Command {
	/// Performs the command in the simulator.
	///
	/// # Errors
	///
	/// Returns why the simulator rejects the action (an injection always
	/// applies).
	pub fn apply(self, g: &mut Game) -> Result<(), ActionError> {
		match self {
			Command::Act(a) => g.step(a),
			Command::Inject(i) => {
				g.inject(i);
				Ok(())
			}
		}
	}

	/// The command for the harness.
	#[must_use]
	pub fn to_lua(&self) -> String {
		match *self {
			Command::Act(a) => action_to_lua(a),
			Command::Inject(i) => inject_to_lua(i),
		}
	}
}

impl fmt::Display for Command {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Command::Act(a) => fmt::Display::fmt(a, f),
			Command::Inject(i) => fmt::Display::fmt(i, f),
		}
	}
}

impl FromStr for Command {
	type Err = String;

	/// Parses an action code or an injection code.
	fn from_str(code: &str) -> Result<Command, String> {
		match code.parse::<Action>() {
			Ok(a) => Ok(Command::Act(a)),
			Err(action_error) => code.parse::<Inject>().map(Command::Inject).map_err(|e| {
				format!("{code:?} is neither an action ({action_error}) nor an injection ({e})")
			}),
		}
	}
}

fn cards_list(m: CardMask) -> String {
	let v: Vec<String> = m.indices().map(|i| (i + 1).to_string()).collect();
	format!("{{{}}}", v.join(","))
}

fn action_to_lua(a: Action) -> String {
	use Action as A;
	let body = match a {
		A::SelectBlind => "type='select_blind'".to_string(),
		A::SkipBlind => "type='skip_blind'".to_string(),
		A::RerollBoss => "type='reroll_boss'".to_string(),
		A::Play(m) => format!("type='play', cards={}", cards_list(m)),
		A::Discard(m) => format!("type='discard', cards={}", cards_list(m)),
		A::UseConsumable { index, targets } => format!(
			"type='use', index={}, cards={}",
			index + 1,
			cards_list(targets)
		),
		A::SellJoker(i) => format!("type='sell_joker', index={}", i + 1),
		A::SellConsumable(i) => format!("type='sell_consumable', index={}", i + 1),
		A::MoveJoker { from, to } => format!("type='move_joker', from={}, to={}", from + 1, to + 1),
		A::MoveHand { from, to } => format!("type='move_hand', from={}, to={}", from + 1, to + 1),
		A::SortHand(order) => format!("type='sort_hand', by='{}'", match order {
			HandSort::Suit => "suit",
			HandSort::Rank => "value",
		}),
		A::BuyShopCard(i) => format!("type='buy', index={}", i + 1),
		A::BuyAndUse { index, targets } => format!(
			"type='buy_and_use', index={}, cards={}",
			index + 1,
			cards_list(targets)
		),
		A::RedeemVoucher(i) => format!("type='redeem', index={}", i + 1),
		A::OpenBooster(i) => format!("type='open', index={}", i + 1),
		A::Reroll => "type='reroll'".to_string(),
		A::EndShop => "type='end_shop'".to_string(),
		A::CashOut => "type='cash_out'".to_string(),
		A::PickPackCard { index, targets } => format!(
			"type='pick', index={}, cards={}",
			index + 1,
			cards_list(targets)
		),
		A::SkipPack => "type='skip_pack'".to_string(),
	};
	format!("H.act({{{body}}})")
}

fn edition_to_lua(e: Option<Edition>) -> String {
	e.map_or("nil".into(), |e| format!("'{}'", e.name()))
}

fn inject_to_lua(i: Inject) -> String {
	match i {
		Inject::Joker {
			joker,
			edition,
			eternal,
		} => format!(
			"H.inject({{kind='joker', key='{}', edition={}, eternal={eternal}}})",
			joker.key(),
			edition_to_lua(edition),
		),
		Inject::Consumable { card, edition } => format!(
			"H.inject({{kind='consumable', key='{}', edition={}}})",
			Kind::from(card).key(),
			edition_to_lua(edition)
		),
		Inject::Card {
			suit,
			rank,
			enhancement,
			edition,
			seal,
		} => format!(
			"H.inject({{kind='card', key='{}_{}', center='{}', edition={}, seal={}}})",
			char::from(suit.letter()),
			char::from(rank.letter()),
			enhancement.key(),
			edition_to_lua(edition),
			seal.map_or("nil".into(), |x| format!("'{}'", x.name()))
		),
		Inject::Dollars(d) => format!("H.inject({{kind='dollars', amount={d}}})"),
		Inject::Voucher(v) => format!("H.inject({{kind='voucher', key='{}'}})", v.key()),
	}
}
