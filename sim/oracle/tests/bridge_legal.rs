//! The real-game bridge's legal-action port (`mod/src/legal.lua`)
//! run on the oracle's (vanilla) game, compared with `Game::legal_actions`
//! after every step of differential games. Needs the game's `lua51.dll`:
//! `cargo test -p oracle --test bridge_legal -- --include-ignored`.

use std::path::Path;

use oracle::command::Start;
use oracle::difftest::{Checks, Comparison, compare};
use oracle::policy::Policy;
use oracle::{Oracle, json, lua_path};
use sim::data::Back;
use sim::game::{Game, Phase};
use sim::{Stake, num};

/// Loads legal.lua with stand-ins for the two helpers it takes from the
/// mod's core.lua (vanilla has no SMODS booster state).
fn load_legal(o: &mut Oracle) {
	let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../mod/src/legal.lua");
	o.exec(&format!(
        r"
        AZB = {{}}
        local S = G.STATES
        local KIND = {{[S.TAROT_PACK] = 'Arcana', [S.PLANET_PACK] = 'Celestial', [S.SPECTRAL_PACK] = 'Spectral',
                      [S.STANDARD_PACK] = 'Standard', [S.BUFFOON_PACK] = 'Buffoon'}}
        function AZB.is_pack_state(s) return KIND[s or G.STATE] ~= nil end
        function AZB.opened_booster_kind() return KIND[G.STATE] end
        dofile({})
        function AZ_LEGAL()
          local codes = AZB.legal_codes()
          codes.__array = true
          return JSON.encode(codes)
        end
        ",
        lua_path(&path)
    ))
    .expect("legal.lua loads in the vanilla game");
}

fn sim_codes(g: &Game) -> Vec<String> {
	let mut v = Vec::new();
	g.legal_actions(&mut v);
	v.iter().map(ToString::to_string).collect()
}

fn lua_codes(o: &mut Oracle) -> Vec<String> {
	let codes = o.eval("return AZ_LEGAL()").expect("AZ_LEGAL runs");
	let j = json::parse(&codes).expect("AZ_LEGAL returns JSON");
	j.as_list()
		.iter()
		.map(|c| c.as_str().expect("a legal code is a string").to_string())
		.collect()
}

#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn bridge_legal_actions_match_the_sim() {
	assert!(
		oracle::available(),
		"the oracle needs the game's lua51.dll and scripts"
	);
	let mut o = Oracle::new().unwrap();
	load_legal(&mut o);
	let backs = [
		Back::Red,
		Back::Blue,
		Back::Yellow,
		Back::Magic,
		Back::Ghost,
		Back::Abandoned,
		Back::Plasma,
	];
	let mut checked = 0;
	for n in 0..24u64 {
		let seed = format!("LEGAL{n:03}");
		let start = Start {
			seed: &seed,
			back: backs[num::pick(n, backs.len())],
			stake: Stake::ALL[num::pick(n, Stake::ALL.len())],
		};
		let mut g = start.new_game();
		let mut lua = o.command(&start.to_lua()).unwrap();
		let mut policy = Policy::for_game(n, 0.15);
		let mut history = vec![start.to_string()];
		for _ in 0..250 {
			let j = json::parse(&lua).unwrap();
			if let Comparison::Diverged(m) = compare(&mut g, &j, Checks::StateOnly) {
				panic!("engines diverged after {}: {m}", history.join(" / "));
			}
			if g.flow.phase == Phase::GameOver {
				break;
			}
			let (want, got) = (sim_codes(&g), lua_codes(&mut o));
			if want != got {
				let missing: Vec<_> = want.iter().filter(|c| !got.contains(c)).take(8).collect();
				let extra: Vec<_> = got.iter().filter(|c| !want.contains(c)).take(8).collect();
				let how = if missing.is_empty() && extra.is_empty() {
					"the same codes in another order".to_string()
				} else {
					format!("missing {missing:?}, extra {extra:?}")
				};
				panic!(
					"legal.lua disagrees in {:?} after {}: {how}",
					g.flow.phase,
					history[history.len().saturating_sub(8)..].join(" / "),
				);
			}
			checked += 1;
			let Some(command) = policy.next(&g) else {
				break;
			};
			history.push(command.to_string());
			command.apply(&mut g).unwrap();
			lua = o.command(&command.to_lua()).unwrap();
		}
	}
	assert!(checked > 1000, "only {checked} states checked");
}
