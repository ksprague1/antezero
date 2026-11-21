"""Evaluate RLBalatro's released model (model_0095.pt) directly in antezero_sim.

Usage:
    uv run python -m rlbalatro_eval --decks b_blue --stakes 8 --games 50
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .antezero_adapter import AntezeroBalatroEnv
from .policy import Policy

DEFAULT_MODEL = Path(__file__).resolve().parent / "model_0095.pt"


def fetch_model(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"model file not found: {p}")
    return p


def load_policy(path: Path, groups, dev: str = "cpu") -> tuple[Policy, dict]:
    ck = torch.load(path, map_location=dev, weights_only=True)
    mc = ck["config"]
    model = Policy(groups, mc["d_model"], mc["layers"], mc["heads"], deck_emb=mc["deck_emb"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, mc


def run_episode(
    env: AntezeroBalatroEnv,
    model: Policy,
    seed: str,
    *,
    device: str = "cpu",
) -> dict:
    o, info = env.reset(seed=seed)
    illegal_steps = 0
    dead_ends = 0
    while True:
        x = torch.from_numpy(np.concatenate([env.obs_buffer, o["selection"]])).to(device)[None]
        mask = torch.from_numpy(info["action_mask"]).to(device)[None]
        spec = torch.from_numpy(env.sel_specs()).to(device=device, dtype=torch.long)[None]
        hctx_np = env.hand_ctx()
        hctx = torch.from_numpy(hctx_np).to(device)[None]
        with torch.no_grad():
            act, _, _, pk, _, _ = model.act(x, mask, spec, hctx, hctx_np[None])
        picks = [int(q) for q in pk[0] if q >= 0]
        o, _, term, trunc, info = env.step((int(act[0]), picks))
        if info["illegal"]:
            illegal_steps += 1
        if info["dead_end"]:
            dead_ends += 1
        if term or trunc:
            break
    return {
        "seed": seed,
        "won": bool(info["won"]),
        "ante": int(info["ante"]),
        "steps": int(info["steps"]),
        "truncated": bool(trunc),
        "illegal_steps": illegal_steps,
        "dead_ends": dead_ends,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate RLBalatro model_0095.pt in antezero_sim.")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="path to model_0095.pt (auto-downloaded if omitted)")
    ap.add_argument("--decks", nargs="+", default=["b_blue"])
    ap.add_argument("--stakes", nargs="+", type=int, default=[8])
    ap.add_argument("--games", type=int, default=50, help="games per (deck, stake) pair")
    ap.add_argument("--seed-prefix", default="EVAL")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--sample", action="store_true", help="sample actions instead of greedy argmax")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--no-extra-obs", action="store_true", help="disable Game.extra_obs() (ablation)")
    a = ap.parse_args(argv)

    if not a.sample:
        torch.multinomial = lambda probs, n, replacement=False, **kw: probs.argmax(-1, keepdim=True)

    model_path = fetch_model(a.model)
    probe = AntezeroBalatroEnv(deck=a.decks[0], stake=a.stakes[0], max_steps=a.max_steps)
    model, _ = load_policy(model_path, probe.groups, dev=a.device)
    probe.close()

    total_games = 0
    total_wins = 0
    total_steps = 0
    total_illegal = 0
    total_trunc = 0
    antes: list[int] = []

    t0 = time.perf_counter()
    for deck in a.decks:
        for stake in a.stakes:
            env = AntezeroBalatroEnv(
                deck=deck,
                stake=stake,
                max_steps=a.max_steps,
                use_extra_obs=not a.no_extra_obs,
            )
            wins, d_antes, d_steps, d_illegal, d_trunc = 0, [], 0, 0, 0
            for i in range(a.games):
                seed = f"{a.seed_prefix}{a.seed_start + i:02d}"
                res = run_episode(env, model, seed, device=a.device)
                wins += int(res["won"])
                d_antes.append(res["ante"])
                d_steps += res["steps"]
                d_illegal += res["illegal_steps"]
                d_trunc += int(res["truncated"])
            env.close()
            total_games += a.games
            total_wins += wins
            total_steps += d_steps
            total_illegal += d_illegal
            total_trunc += d_trunc
            antes.extend(d_antes)
            print(
                f"{deck} stake={stake}: {wins}/{a.games} wins ({100.0 * wins / a.games:.1f}%), "
                f"mean_ante={np.mean(d_antes):.2f}, max_ante={max(d_antes)}, "
                f"mean_steps={d_steps / a.games:.1f}, illegal={d_illegal}, trunc={d_trunc}",
                flush=True,
            )
    dt = time.perf_counter() - t0
    print(
        f"\nTOTAL: {total_wins}/{total_games} wins ({100.0 * total_wins / total_games:.1f}%), "
        f"mean_ante={np.mean(antes):.2f}, mean_steps={total_steps / total_games:.1f} "
        f"({total_steps / dt:.1f} steps/s), illegal={total_illegal}, trunc={total_trunc}, elapsed={dt:.1f}s"
    )


if __name__ == "__main__":
    main()
