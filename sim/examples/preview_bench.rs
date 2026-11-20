//! Cost of the build previews on states of real-looking games:
//! `cargo run --release --example preview_bench [games]`.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::env;
use std::hint::black_box;
use std::time::Instant;

use sim::data::Back;
use sim::game::{Game, Phase, Scratch};
use sim::obs::{self, ItemPreview, ObsBuf};
use sim::rng::{self, PseudoRandom, Xorshift};
use sim::{Action, CardMask, Kind, Stake, num};

/// The action the state-collecting policy takes: the best play in rounds,
/// jokers and now and then a consumable or a pack in shops.
fn policy(g: &Game, x: u64, acts: &mut Vec<Action>) -> Option<Action> {
	use sim::CardLike;

	Some(match g.flow.phase {
		Phase::Round => {
			let top = obs::top_plays(g, 1);
			match top.first() {
				Some(c) if !x.is_multiple_of(7) || g.round.discards_left <= 0.0 => {
					Action::Play(c.mask)
				}
				_ => {
					g.legal_actions(acts);
					acts[num::pick(x >> 8, acts.len())]
				}
			}
		}
		Phase::BlindSelect => Action::SelectBlind,
		Phase::RoundEval => Action::CashOut,
		// Build up: buy what is affordable (jokers first), open a pack now and then, else leave.
		Phase::Shop => {
			g.legal_actions(acts);
			let buy = acts.iter().find(|a| {
				matches!(a, Action::BuyShopCard(i)
					if g.areas.shop_jokers[usize::from(*i)].as_card().is_some_and(|c| matches!(c.kind, Kind::Joker(_))))
			});
			let use_ = acts
				.iter()
				.find(|a| matches!(a, Action::BuyAndUse { .. } | Action::RedeemVoucher(_)));
			let open = acts.iter().find(|a| matches!(a, Action::OpenBooster(_)));
			match (buy, use_, open) {
				(Some(a), ..) => *a,
				(_, Some(a), _) if x.is_multiple_of(2) => *a,
				(_, _, Some(a)) if x.is_multiple_of(3) => *a,
				_ => Action::EndShop,
			}
		}
		Phase::Pack => {
			g.legal_actions(acts);
			let pick = acts.iter().find(|a| {
				matches!(a, Action::PickPackCard { index, .. }
					if g.areas.pack_cards[usize::from(*index)].as_card().is_some_and(|c| matches!(c.kind, Kind::Joker(_) | Kind::Planet(_))))
			});
			match pick {
				Some(a) => *a,
				None => acts
					.iter()
					.find(|a| matches!(a, Action::SkipPack))
					.copied()
					.unwrap_or(acts[num::pick(x >> 8, acts.len())]),
			}
		}
		Phase::GameOver => return None,
	})
}

/// States of `games` games played by [`policy`] (at most 20,000).
fn collect_states(games: usize) -> Vec<Game> {
	let mut states: Vec<Game> = Vec::new();
	let mut rng = Xorshift::new(rng::SPLITMIX_GAMMA);
	let mut acts = Vec::new();
	let mut steps_by_phase = Phase::ALL.map(|phase| (phase, 0usize));
	for s in 0..games {
		let mut g = Game::new_run(&format!("BENCH{s}"), Back::Red, Stake::White);
		for _ in 0..1500 {
			if g.is_terminal() {
				break;
			}
			let (_, steps) = steps_by_phase
				.iter_mut()
				.find(|(phase, _)| *phase == g.flow.phase)
				.expect("`Phase::ALL` has every phase");
			*steps += 1;
			if states.len() < 20_000 {
				states.push(g.clone());
			}
			let x = rng.next_u64();
			let Some(action) = policy(&g, x, &mut acts) else {
				break;
			};
			g.step(action).expect("a legal action steps");
		}
	}
	let by_phase: Vec<String> = steps_by_phase
		.iter()
		.map(|(phase, steps)| format!("{} {steps}", phase.name()))
		.collect();
	println!(
		"{} states; steps by phase: {}",
		states.len(),
		by_phase.join(", ")
	);
	let max_ante = states.iter().map(|g| g.run.ante).max().unwrap_or(0);
	println!("max ante reached {max_ante}");
	states
}

/// The cost of an observation on up to 400 states of `phase`: without
/// previews, the build preview alone, and the full observation.
fn time_phase(states: &[Game], phase: Phase, buf: &mut ObsBuf) {
	let sel: Vec<&Game> = states
		.iter()
		.filter(|g| g.flow.phase == phase)
		.take(400)
		.collect();
	if sel.is_empty() {
		return;
	}
	let count = num::float(sel.len());
	let mut ops = Vec::new();
	let mut cache = Vec::new();
	// The env refreshes the derived joker values in place: on copies made
	// before the clock starts.
	let mut copies: Vec<Game> = sel.iter().map(|g| (*g).clone()).collect();
	// Full observation (encode + ops), as the env does it, no cache.
	let start = Instant::now();
	let mut previewed_items = 0usize;
	for g in &mut copies {
		g.update_derived();
		cache.clear();
		let preview = obs::build_preview(g, &mut cache);
		let lay = obs::encode_with(g, &mut buf.as_mut(), /* deck_tokens= */ false, preview);
		obs::ops(
			g,
			&lay,
			obs::OpOptions {
				candidates: 8,
				allow_skip: false,
				..Default::default()
			},
			&mut ops,
		);
		let (tokens, _) = buf.tok_f.as_chunks::<{ obs::TOK_FLOAT }>();
		previewed_items += tokens
			.iter()
			.take(lay.n_tokens)
			.filter(|token| {
				let columns = token[obs::TOK_PREVIEW..]
					.try_into()
					.expect("a token ends with its preview columns");
				ItemPreview::from_columns(columns).is_some()
			})
			.count();
		black_box(&ops);
	}
	let full = start.elapsed().as_secs_f64() / count;
	let start = Instant::now();
	for g in &sel {
		cache.clear();
		obs::build_preview(g, &mut cache);
		black_box(&cache);
	}
	let preview = start.elapsed().as_secs_f64() / count;
	// Observation without any preview.
	let start = Instant::now();
	for g in &mut copies {
		g.update_derived();
		let lay = obs::encode_with(g, &mut buf.as_mut(), /* deck_tokens= */ false, None);
		obs::ops(
			g,
			&lay,
			obs::OpOptions {
				candidates: if phase == Phase::Round { 8 } else { 0 },
				allow_skip: false,
				..Default::default()
			},
			&mut ops,
		);
		black_box(&ops);
	}
	let plain = start.elapsed().as_secs_f64() / count;
	println!(
		"{:>12}: {} states, obs without previews {:7.1} us, baseline preview {:7.1} us, full obs with previews {:7.1} us ({:.1} previewed items/state)",
		phase.name(),
		sel.len(),
		plain * 1e6,
		preview * 1e6,
		full * 1e6,
		num::float(previewed_items) / count
	);
}

/// Virtual round setup against best-play search, on shop states.
fn time_shop_split(states: &[Game]) {
	let shops: Vec<&Game> = states
		.iter()
		.filter(|g| g.flow.phase == Phase::Shop)
		.take(300)
		.collect();
	if shops.is_empty() {
		return;
	}
	let count = num::float(shops.len());
	let start = Instant::now();
	let worlds: Vec<Game> = shops
		.iter()
		.map(|&g| obs::virtual_round(g.clone(), g.blind_select.boss, 1))
		.collect();
	let setup = start.elapsed().as_secs_f64() / count;
	let start = Instant::now();
	for world in &worlds {
		black_box(obs::best_play(world));
	}
	let search = start.elapsed().as_secs_f64() / count;
	let jokers = num::float(shops.iter().map(|g| g.areas.jokers.len()).sum::<usize>()) / count;
	let antes = shops.iter().map(|g| f64::from(g.run.ante)).sum::<f64>() / count;
	println!(
		"shop states: mean ante {antes:.1}, mean jokers {jokers:.1}; virtual_round {:.1} us, best_play {:.1} us (PREVIEW_PLAYS={})",
		setup * 1e6,
		search * 1e6,
		obs::PREVIEW_PLAYS
	);
}

fn main() {
	let games: usize = env::args().nth(1).map_or(40, |s| {
		s.parse().expect("the argument is a number of games")
	});
	let states = collect_states(games);
	let mut buf = ObsBuf::new();
	for phase in [Phase::Round, Phase::Shop, Phase::Pack, Phase::BlindSelect] {
		time_phase(&states, phase, &mut buf);
	}
	time_shop_split(&states);
	// One preview_play, for scale.
	let rounds: Vec<&Game> = states
		.iter()
		.filter(|g| g.flow.phase == Phase::Round)
		.take(200)
		.collect();
	let alt = PseudoRandom::new("X");
	let mut scratch = Scratch::default();
	let start = Instant::now();
	let mut plays = 0usize;
	for g in &rounds {
		for bits in [1, 3, 7, 15, 31] {
			if obs::preview_score(&mut scratch, g, CardMask::from_bits(bits), &alt).is_some() {
				plays += 1;
			}
		}
	}
	println!(
		"preview_play: {:.1} us each ({plays} plays)",
		start.elapsed().as_secs_f64() * 1e6 / num::float(plays.max(1))
	);
}
