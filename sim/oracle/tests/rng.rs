//! Differential tests of the RNG layer against the game's own `LuaJIT`. They
//! need the game's `lua51.dll` (and scripts), so they only run when asked
//! for: `cargo test -p oracle --test rng -- --include-ignored`.

use oracle::{Oracle, lua_path, lua_str};
use sim::rng::{LuaRandom, PseudoRandom, pseudohash, round13};

/// The comma-separated numbers the oracle printed.
fn parse(s: &str) -> Vec<f64> {
	if s.is_empty() {
		return Vec::new();
	}
	s.split(',')
		.map(|x| x.parse::<f64>().expect("the oracle prints numbers"))
		.collect()
}

fn bare() -> Oracle {
	assert!(
		oracle::dll_available(),
		"the RNG tests need the game's lua51.dll (see oracle/src/lib.rs)"
	);
	Oracle::new_bare().expect("lua51.dll loads")
}

#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn math_random_matches_luajit() {
	const SEEDS: usize = 3000;
	let mut o = bare();
	// Seeds spanning the ranges the game uses (pseudoseed outputs are in [0,1)).
	let out = o
		.eval(&format!(
			r"
            local t = {{}}
            local s = 0.123456789
            for i = 1, {SEEDS} do
                s = (s * 7.31 + 0.137) % 1
                local seed = (i % 7 == 0) and (s * 1000 - 500) or s
                math.randomseed(seed)
                t[#t+1] = string.format('%.17g', seed)
                t[#t+1] = string.format('%.17g', math.random())
                t[#t+1] = string.format('%.17g', math.random(52))
                t[#t+1] = string.format('%.17g', math.random(0, 23))
                t[#t+1] = string.format('%.17g', math.random())
            end
            return table.concat(t, ',')
        ",
		))
		.unwrap();
	let v = parse(&out);
	assert_eq!(v.len(), SEEDS * 5);
	for c in v.chunks(5) {
		let mut r = LuaRandom::new(c[0]);
		assert_eq!(r.random(), c[1], "seed {}", c[0]);
		assert_eq!(r.random_int(52.0), c[2], "seed {}", c[0]);
		assert_eq!(r.random_range(0.0, 23.0), c[3], "seed {}", c[0]);
		assert_eq!(r.random(), c[4], "seed {}", c[0]);
	}
}

#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn round13_matches_string_format() {
	const VALUES: usize = 200_000;
	let mut o = bare();
	let out = o
		.eval(&format!(
			r#"
            local t = {{}}
            math.randomseed(42)
            for i = 1, {VALUES} do
                local x = math.random()
                if i % 3 == 0 then x = (2.134453429141 + x*1.72431234)%1 end
                if i % 11 == 0 then x = math.floor(x*1e13)/1e13 + 0.5e-13 end
                t[#t+1] = string.format('%.17g', x)
                t[#t+1] = string.format('%.17g', math.abs(tonumber(string.format("%.13f", x))))
            end
            return table.concat(t, ',')
        "#,
		))
		.unwrap();
	let v = parse(&out);
	assert_eq!(v.len(), VALUES * 2);
	for c in v.chunks(2) {
		assert_eq!(round13(c[0]), c[1], "x = {:e}", c[0]);
	}
}

#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn pseudohash_and_pseudoseed_match_game_code() {
	const REPS: usize = 40;
	let mut o = bare();
	let src = oracle::default_src().join("functions/misc_functions.lua");
	assert!(
		src.exists(),
		"the game scripts are missing ({})",
		src.display()
	);
	o.exec(&format!("dofile({})", lua_path(&src))).unwrap();
	let keys = [
		"boss",
		"shuffle",
		"Joker1sho1",
		"rarity1sho",
		"cdt1",
		"edisho1",
		"Tag1",
		"Voucher1",
		"soul_Tarot1",
		"front",
		"nr1",
		"cashout2",
		"lucky_mult",
		"Joker2_resample2",
		"misprint",
		"a",
		"",
		"orbital",
		"to_do",
	];
	let seeds = [
		"ABCDEFGH",
		"1",
		"TUTORIAL",
		"ZZZZZZZZ",
		"8NJHJ5M2",
		"seedwith spaces",
	];
	let lua_keys = keys.map(lua_str).join(",");
	for seed in seeds {
		let seed_lua = lua_str(seed);
		let code = format!(
			r"
            G = {{GAME = {{pseudorandom = {{seed = {seed_lua}}}}}}}
            G.GAME.pseudorandom.hashed_seed = pseudohash({seed_lua})
            local keys = {{{lua_keys}}}
            local t = {{string.format('%.17g', G.GAME.pseudorandom.hashed_seed)}}
            for rep = 1, {REPS} do
                for _, k in ipairs(keys) do
                    t[#t+1] = string.format('%.17g', pseudoseed(k))
                end
            end
            for _, k in ipairs(keys) do t[#t+1] = string.format('%.17g', pseudohash(k)) end
            return table.concat(t, ',')
        ",
		);
		let v = parse(&o.eval(&code).unwrap());
		assert_eq!(v.len(), 1 + (REPS + 1) * keys.len(), "seed {seed}");
		let mut pr = PseudoRandom::new(seed);
		assert_eq!(pr.hashed_seed(), v[0], "hashed seed {seed}");
		let mut i = 1;
		for _ in 0..REPS {
			for k in keys {
				assert_eq!(pr.seed_str(k), v[i], "seed {seed} key {k}");
				i += 1;
			}
		}
		for k in keys {
			assert_eq!(pseudohash(k.as_bytes()), v[i], "pseudohash {k}");
			i += 1;
		}
	}
}
