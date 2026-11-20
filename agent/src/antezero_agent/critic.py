"""A decoupled critic for PPO (``ppo.critic``): its own copy of :class:`~.model.Net`'s token encoder with an
HL-Gauss value head (a categorical over return bins; Farebrother et al. 2024) and the P(win) head its checkpoints
were trained with (unused by PPO). ``value`` is the expected bin value, in the policy value head's units. Trained
offline (the training scripts are not part of this repository); PPO reads it frozen, and :class:`Learner` can keep a
copy fitted during a run (``ppo.critic_learn``)."""

from __future__ import annotations

import copy
import math

import antezero_sim as az
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import effects
from .config import ModelConfig
from .model import GLOB_VOCAB, TOK_VOCAB, Obs, mlp

V_MIN, V_MAX, N_BINS = -0.25, 2.0, 64  # returns span about -0.13..1.86
SIGMA = 0.75 * (V_MAX - V_MIN) / N_BINS  # the width of the HL-Gauss targets


class Critic(nn.Module):
    def __init__(self, cfg: ModelConfig, fx: torch.Tensor | None = None):
        super().__init__()
        d = cfg.d_model
        self.tok_emb = nn.ModuleList(nn.Embedding(v, d) for v in TOK_VOCAB)
        self.tok_lin = nn.Linear(az.TOK_FLOAT, d)
        self.glob_emb = nn.ModuleList(nn.Embedding(v, d) for v in GLOB_VOCAB)
        self.glob_lin = nn.Linear(az.GLOB_FLOAT, d)
        self.in_norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.ff_mult * d, 0.0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)
        self.v_head = mlp(d, d, N_BINS)
        self.w_head = mlp(d, d, 1)
        edges = torch.linspace(V_MIN, V_MAX, N_BINS + 1)
        self.register_buffer("edges", edges, persistent=False)
        self.register_buffer("centers", (edges[:-1] + edges[1:]) / 2, persistent=False)
        # As Net.fx and Net.fx_lin.
        self.register_buffer("fx", effects.table(cfg.effects) if fx is None else fx)
        self.fx_lin = nn.Linear(self.fx.shape[1], d, bias=False)
        nn.init.zeros_(self.fx_lin.weight)

    def _cls(self, o: Obs, T: int | None = None) -> torch.Tensor:
        """[B, d] state encodings. Float inputs are clipped to +-6e4 and rounded to float16, as the critic's
        training data was."""
        T = T or max(int(o.n_tokens.max().item()), 1)
        ti, tf = o.tok_i[:, :T], o.tok_f[:, :T].clamp(-6e4, 6e4).half().float()
        x = self.tok_lin(tf) + self.fx_lin(self.fx[ti[..., 1].clamp(0, az.N_CENTERS - 1)])
        for k, emb in enumerate(self.tok_emb):
            x = x + emb(ti[..., k].clamp(0, TOK_VOCAB[k] - 1))
        g = self.glob_lin(o.glob_f.clamp(-6e4, 6e4).half().float())
        for k, emb in enumerate(self.glob_emb):
            g = g + emb(o.glob_i[:, k].clamp(0, GLOB_VOCAB[k] - 1))
        h = self.in_norm(torch.cat([g.unsqueeze(1), x], 1))
        pad = torch.cat([torch.zeros_like(o.n_tokens[:, None], dtype=torch.bool), torch.arange(T, device=ti.device)[None] >= o.n_tokens[:, None]], 1)
        # The encoder may run in float16 (train.py on CUDA); the rest stays float32.
        enc = self.encoder(h.to(self.encoder.layers[0].linear1.weight.dtype), src_key_padding_mask=pad).float()
        return self.out_norm(enc)[:, 0]

    def value(self, o: Obs, T: int | None = None) -> torch.Tensor:
        return (F.softmax(self.v_head(self._cls(o, T)), -1) * self.centers).sum(-1)

    def win_prob(self, o: Obs) -> torch.Tensor:
        return torch.sigmoid(self.w_head(self._cls(o))).squeeze(-1)

    def losses(self, o: Obs, G: torch.Tensor, won: torch.Tensor):
        """What the offline fit minimises, on returns ``G`` and outcomes ``won`` -> (the value head's cross-entropy
        against HL-Gauss targets: the mass of N(G, SIGMA^2) in each bin; the win head's BCE; the values [B])."""
        cls = self._cls(o)
        logits = self.v_head(cls)
        cdf = torch.special.ndtr((self.edges[None] - G[:, None]) / SIGMA)
        p = cdf[:, 1:] - cdf[:, :-1]
        ce = -(p / p.sum(-1, keepdim=True).clamp(min=1e-12) * F.log_softmax(logits, -1)).sum(-1).mean()
        bce = F.binary_cross_entropy_with_logits(self.w_head(cls).squeeze(-1), won.float())
        return ce, bce, (F.softmax(logits, -1) * self.centers).sum(-1).detach()


class Learner:
    """Keeps a copy of the critic fitted while PPO trains (``ppo.critic_learn``). A sample of each rollout's states waits
    until its episode ends and is then trained on once, against the realised discounted return and outcome, with the
    offline fit's loss. No value function enters a target, so the copy never learns from its own predictions, and a
    state is only trained on after the rollout it came from has been used. Every state is scored before it is trained
    on: :meth:`step` reports those scores next to the values the rollout recorded for the same states."""

    # States sampled per rollout; the offline fine-tune's batch size and about its mean learning rate.
    FRAC, BATCH, LR = 1 / 32, 512, 5e-5
    FIELDS = ["learn_states", "learn_ev", "learn_bias", "learn_critic_ev", "learn_critic_bias", "learn_value_ev", "learn_value_bias"]

    def __init__(self, critic: Critic):
        self.net = copy.deepcopy(critic).float().train().requires_grad_(True)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=self.LR, weight_decay=0.01)
        self.waiting: list[dict] = []  # per earlier rollout: its sampled states whose episode has not ended

    def state(self) -> dict:
        return {"model": self.net.state_dict(), "opt": self.opt.state_dict()}

    def load(self, state: dict):
        self.net.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])

    def collect(self, roll, ends: torch.Tensor, part: torch.Tensor, refs, gamma: float, r_win: float) -> dict:
        """Takes a rollout (train.Rollout, with train.mc_returns' ``ends`` and ``part``) and returns the sampled states
        whose episode has now ended, of this rollout and of earlier ones: their observation, ``G`` (the discounted
        return to the episode's end), ``won`` and ``ref`` (the values of ``refs`` [T, N] recorded for them)."""
        T, N, dev = roll.T, roll.N, part.device
        # The step at which each step's episode ends inside the rollout (T: not inside it), and whether it was won.
        # A win is read off the final reward: r_win against shaping steps of a few hundredths. A truncated episode's
        # bootstrapped reward can pass for one.
        first = torch.where(roll.done > 0, torch.arange(T, device=dev)[:, None], T).flip(0).cummin(0).values.flip(0)
        won = ends & (roll.reward.gather(0, first.clamp(max=T - 1)) > r_win / 2)
        done, waiting = [], []
        for w in self.waiting:
            n = w["env"]
            w = {**w, "G": w["G"] + w["disc"] * part[0, n], "won": won[0, n], "disc": w["disc"] * gamma**T}
            fin = ends[0, n]
            done.append({k: v[fin] for k, v in w.items()})
            if not fin.all():
                waiting.append({k: v[~fin] for k, v in w.items()})
        idx = torch.randperm(T * N, device=dev)[: max(1, int(T * N * self.FRAC))]
        t, n = idx // N, idx % N
        W = max(int(roll.n_tokens.flatten()[idx].max()), 1)
        o = roll.obs(idx, None, W, 0)
        new = {
            "env": n,
            "G": part[t, n],
            "disc": gamma ** (T - t).float(),
            "won": won[t, n],
            "ref": torch.stack([r[t, n] for r in refs], -1),
            # As compact as the rollout keeps them; float16 is what the critic rounds its inputs to anyway.
            "tok_i": o.tok_i[:, :W].to(torch.int16),
            "tok_f": o.tok_f[:, :W].clamp(-6e4, 6e4).half(),
            "glob_i": o.glob_i.to(torch.int16),
            "glob_f": o.glob_f.clamp(-6e4, 6e4).half(),
            "n_tokens": o.n_tokens,
        }
        fin = ends[t, n]
        done.append({k: v[fin] for k, v in new.items()})
        waiting.append({k: v[~fin] for k, v in new.items()})
        self.waiting = waiting
        W = max(d["tok_i"].shape[1] for d in done)
        wide = lambda k, x: F.pad(x, (0, 0, 0, W - x.shape[1])) if k in ("tok_i", "tok_f") else x  # noqa: E731
        return {k: torch.cat([wide(k, d[k]) for d in done]) for k in new}

    def step(self, roll, ends: torch.Tensor, part: torch.Tensor, refs, gamma: float, r_win: float) -> dict:
        """One update's learning: :meth:`collect`, then one pass over the states it returns -> ``FIELDS`` (the states
        trained on; the explained variance and mean error, on those states, of the copy before it saw them, and of
        ``refs``: the acting critic's values and the policy head's)."""
        d = self.collect(roll, ends, part, refs, gamma, r_win)
        G = d["G"]
        vals = torch.empty_like(G)
        for idx in torch.randperm(len(G), device=G.device).split(self.BATCH) if len(G) else ():
            T = max(int(d["n_tokens"][idx].max()), 1)
            o = Obs(d["tok_i"][idx, :T].long(), d["tok_f"][idx, :T].float(), d["glob_i"][idx].long(), d["glob_f"][idx].float(), None, None, None, d["n_tokens"][idx], None)
            ce, bce, vals[idx] = self.net.losses(o, G[idx], d["won"][idx])
            self.opt.zero_grad(set_to_none=True)
            (ce + bce).backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
        out = {"learn_states": len(G)}
        for k, v in (("learn", vals), ("learn_critic", d["ref"][:, 0]), ("learn_value", d["ref"][:, 1])):
            e = v - G
            out[f"{k}_ev"], out[f"{k}_bias"] = (float(1 - e.var() / G.var()), float(e.mean())) if len(G) > 1 else (math.nan, math.nan)
        return out


def critic_state(path) -> dict:
    """The critic weights of a checkpoint: a critic's own ({"model": ...}) or a training
    checkpoint that holds one ("critic")."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "critic" in ck:
        return ck["critic"]
    sd = ck["model"]
    if "v_head.0.weight" not in sd:
        raise ValueError(f"{path} holds no critic")
    return sd
