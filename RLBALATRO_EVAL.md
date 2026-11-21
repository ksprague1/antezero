# Evaluating RLBalatro (`model_0095.pt`) in `antezero`

This branch (`expose-extra-obs`) includes `rlbalatro_eval/`, a standalone Python runner that evaluates [RLBalatro](https://github.com/ksprague1/RLBalatro-public)'s released Run 95 policy (`model_0095.pt`) directly in `antezero_sim` without needing `RLBalatro-public` or `libbalatro.so`.

## Quickstart

```bash
uv sync
uv run python -m rlbalatro_eval --decks b_blue --stakes 8 --games 50
```

If `--model` is omitted, `rlbalatro_eval` automatically downloads `model_0095.pt` from the RLBalatro release (`v0.1`) to `rlbalatro_eval/model_0095.pt`.

Useful flags:
- `--decks b_red b_blue ...` (default: `b_blue`)
- `--stakes 1 8 ...` (default: `8` Gold Stake)
- `--games N` (games per `(deck, stake)` pair, default: `50`)
- `--sample` (sample actions instead of greedy argmax)
- `--no-extra-obs` (ablation: disable `Game.extra_obs()` and use only `canonical()` + Python-derivable fields)

## Observation & Stepping Differences vs `BalatroEnv`

`antezero`'s `Game.canonical()` is designed for bit-exact oracle diffing rather than RLBalatro's exact observation layout, so `AntezeroBalatroEnv` bridges the gap as follows:

### Without `Game.extra_obs()` (`--no-extra-obs`)
- `last_hand_chips` and `last_hand_mult` (`round_ctx[1..2]`) are not in `canonical()` and default to `0.0` (`last_hand_score` is tracked via `chips` deltas).
- Per-card `times_played` (`hand`/`deck`/`discard`) is not in `canonical()` and defaults to `0`.
- `hand_levels[:, 4]` (`visible`) is inferred as `1.0` for the 9 standard hands and `0.0` for secret hands (`Flush Five`, `Flush House`, `Five of a Kind`) until `played > 0`.
- Static `extra0`/`extra1` constants on jokers/vouchers/consumables come from a Python lookup table (`_STATIC_EXTRA0`, `_STATIC_EXTRA1`, `_EXTRA_OVERRIDE`); booster pack sizes (`extra0`) are `0.0`.

### With `Game.extra_obs()` (default on `expose-extra-obs`)
- `Game.extra_obs()` exposes `last_hand_score`, `last_hand_chips`, `last_hand_mult`, `hand_visible`, `hand_times_played`, `deck_times_played`, `discard_times_played`, and dynamic `ab.extra`/`ab.chips`/`ab.dollars`/booster `def().extra` without modifying `canon.rs`.
- **Remaining differences vs native `BalatroEnv`:**
  - `scalars[44]`: `BalatroEnv` writes cumulative `run.hands_played`; `obs_builder.py` originally wrote `shop_joker_max`, which `AntezeroBalatroEnv` patches to `run.hands_played` (and `scalars[34]` to `hands_played_round`).
  - `idol_rank`: `antezero` always rolls `idol.rank`, whereas `BalatroEnv` keeps `idol_rank = 11` (and `ancient_suit = ""`, `castle_suit = ""`, `mail_rank = 0`) unless the corresponding joker is owned; `AntezeroBalatroEnv` gates these on owning the joker.
  - `extra0` / `extra1`: `antezero` stores dynamic `ab.extra` (e.g. Seltzer, Turtle Bean) and `ab.chips` (e.g. Runner, Square, Ice Cream), whereas `BalatroEnv` also encodes static `config.extra` constants for jokers/vouchers/consumables (handled via `_STATIC_EXTRA0`/`_STATIC_EXTRA1`) and encodes `j_todo_list` `extra1` as the target hand index (`1..12`).
  - `is_face` edge cases, `BalatroEnv` auto-exiting empty/unaffordable shops (or `BLIND_SELECT` / `ROUND_EVAL` auto-stepping — `AntezeroBalatroEnv` auto-steps `ROUND_EVAL` and Boss `BLIND_SELECT` when `reroll_boss` is unavailable), and `obs_builder` action masks intersected with `legal_codes()` (with `REORDER_JOKERS` capped at 1 per state).
  - **Face-down cards & jokers:** Run 95 (`model_0095.pt`) was trained with `obs_hide_face_down=False` (face-down cards and jokers visible in the observation), whereas `AntezeroBalatroEnv` defaults to `obs_hide_face_down=True` (hiding face-down card/joker identity and stats), so evaluation in `antezero` can be slightly weaker on face-down bosses (The House, The Wheel, The Mark, The Fish, Amber Acorn).

## Run 95 (`model_0095.pt`) Training Setup

Run 95 was trained in `RLBalatro` (`BalatroEnv`) with:
- **Reward scheme:** `reward_mode="blinds"` (`+1.0` per blind beaten, `reward_skip_deferred=True`, `win_log=True` paying `log10(final_round_chips)` at ante 8, `endless_ante=16`, partial loss credit `min(1.0, chips / blind_chips)` on defeat, `gamma=1.0`).
- **Curriculum / training distribution:** all 15 decks at Gold Stake (`stake=8`), with 20% human start states from `pools/human_gold_v2_ante6` and `disable_softlocks=True`.
