"""Worker and episode runner helpers for parallel evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .antezero_adapter import AntezeroBalatroEnv
from .policy import Policy

_WORKER_STATE: dict = {}


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


def init_worker(
    model_path: str,
    deck: str,
    stake: int,
    max_steps: int,
    use_extra_obs: bool,
    sample: bool,
    device: str,
) -> None:
    torch.set_num_threads(1)
    if not sample:
        torch.multinomial = lambda probs, n, replacement=False, **kw: probs.argmax(-1, keepdim=True)
    env = AntezeroBalatroEnv(deck=deck, stake=stake, max_steps=max_steps, use_extra_obs=use_extra_obs)
    model, _ = load_policy(Path(model_path), env.groups, dev=device)
    _WORKER_STATE["env"] = env
    _WORKER_STATE["model"] = model
    _WORKER_STATE["device"] = device


def run_worker_seed(seed: str) -> dict:
    return run_episode(_WORKER_STATE["env"], _WORKER_STATE["model"], seed, device=_WORKER_STATE["device"])
