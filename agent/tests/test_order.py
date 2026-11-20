"""The joker-order gain input, the labels at played hands and the ordering loss (``ppo.aux_order``)."""

import csv
import dataclasses
import functools
import math

import numpy as np
import pytest
import torch

import antezero_sim as az
from antezero_agent import collect as collect_mod
from antezero_agent import train as train_mod
from antezero_agent.config import ModelConfig, preset
from antezero_agent.model import GAIN, MAX_CARDS, ORDER_BAND, Net, Obs, obs_from_numpy
from antezero_agent.runtime import make_env, read_checkpoint
from antezero_agent.train import LOG_FIELDS, ORDER_CLIP, ORDER_FIELDS, add_order_grad, train

# Jokers that only score (no money, no created cards, no random draws): their place in the row changes the score alone.
SCORING = [0, 5, 6, 15, 16, 21, 22, 30, 32, 33, 38, 39, 40, 60, 71, 77, 100, 103, 107, 114, 122, 130, 131, 137]
PROGRESS = (0, 14, 13, 16)  # select blind, cash out, end shop, skip pack
# What the ordering loss may move. dst_k.bias is left out of comparisons: it shifts every destination key alike, which
# the pointer's softmax ignores, so its gradient is zero up to rounding.
ORDER_PARAMS = {"op_feat.weight", "dst_q.0.weight", "dst_q.0.bias", "dst_q.2.weight", "dst_q.2.bias", "dst_k.weight", "dst_pos.weight"}


def tiny(gain_weights: bool = False):
    """A small policy. Like a migrated checkpoint, its gain input has no weights yet unless asked for."""
    torch.manual_seed(0)
    m = Net(ModelConfig(d_model=32, n_layers=1, n_heads=2)).eval()
    if not gain_weights:
        m.op_feat.weight.data[:, GAIN] = 0
    return m


def joker_game(k: int) -> az.Game:
    """A Green-stake game at its first hand, with 3 to 6 of ``SCORING`` (some with an edition). It starts in the
    round because a random policy at the blind select mostly sells or moves its jokers."""
    rng = np.random.default_rng(k)
    g = az.Game(f"ORD{k}", "b_red", 3)
    for j in rng.choice(SCORING, size=rng.integers(3, 7), replace=False):
        g.inject(f"inject_joker {j} {rng.choice([0, 0, 2, 3])} 0")
    g.step("select")
    return g


def joker_env(cfg, seed=None, num_envs=None, **over):
    """``make_env`` with every env on a :func:`joker_game` (the games an env resets to start without jokers)."""
    env = make_env(cfg, seed=seed, num_envs=num_envs, **over)
    for i in range(env.num_envs):
        env.set_game(i, joker_game(1000 * (seed or 0) + i))
    return env


def choose(env, rng, weak=True):
    """A scored play in rounds (``weak``: a random one, which more often stays below the blind; else the top
    candidate, as a trained policy mostly plays), otherwise progress."""
    ops, n_ops = np.asarray(env.ops), np.asarray(env.n_ops)
    op, cards = np.zeros(env.num_envs, np.int32), np.zeros(env.num_envs, np.uint32)
    for i in range(env.num_envs):
        kinds = ops[i, : n_ops[i], 0]
        top = np.flatnonzero(kinds == az.OP_PLAY_TOP)
        if len(top):
            op[i] = rng.choice(top) if weak else top[0]
            cards[i] = np.int32(ops[i, op[i], 5]).view(np.uint32)
        else:
            op[i] = next(int(np.flatnonzero(kinds == k)[0]) for k in PROGRESS if (kinds == k).any())
    return op, cards, np.zeros(env.num_envs, np.int32)


def brute_label(g: az.Game, mask: int):
    """The label of playing ``mask`` in ``g`` from full steps on clones whose hidden state is resampled:
    (kind, destination bits per joker), or None for a play whose scores depend on the resample (chance)."""
    c = g.canonical()
    n = int(c["jokers.len"])
    if g.phase != "round" or not 2 <= n <= az.ORDER_JOKERS or any(v == "back" for k, v in c.items() if k.startswith(("hand.", "jokers.")) and k.endswith(".facing")):
        return -1, [0] * az.ORDER_JOKERS
    need = g.blind_chips - g.chips

    def score(move, world):
        w = g.clone()
        w.resample_hidden(f"W{world}")
        if move:
            w.step(f"move_joker {move[0]} {move[1]}")
        w.step(f"play {mask}")
        return w.chips - g.chips

    moves = [(i, j) for i in range(n) for j in range(n) if i != j]
    runs = [[score(m, world) for m in [None] + moves] for world in range(2)]
    if runs[0] != runs[1]:
        return None
    base, capped = min(runs[0][0], need), [min(s, need) for s in runs[0][1:]]
    if base >= need:
        return -1, [0] * az.ORDER_JOKERS
    gaining = [s for s in capped if (s + 1) / (base + 1) >= 1.01]
    dest = [0] * az.ORDER_JOKERS
    for (i, j), s in zip(moves, capped):
        if gaining and (s + 1) / (base + 1) >= 1.01 and s >= 0.999 * max(gaining):
            dest[i] |= 1 << j
    return int(bool(gaining)), dest


def test_labels_are_a_brute_force_of_the_state_the_action_was_taken_in():
    """Every label equals the one made from full steps on resampled clones of the state the play was chosen in,
    also when that play ended the episode (the env is on its next game when the label is read)."""
    cfg, rng = preset("smoke"), np.random.default_rng(0)
    env = joker_env(cfg, seed=7, num_envs=32, order_label_rate=1.0)
    assert (np.asarray(env.aux_kind) == -1).all()
    counts, games = dict(labels=0, positive=0, tied=0, ended=0, plays=0), 32
    for _ in range(260):
        op, cards, dest = choose(env, rng)
        before = [env.game(i) if cards[i] else None for i in range(env.num_envs)]
        _rew, done, _trunc = env.step(op, cards, dest)
        kind, bits = np.asarray(env.aux_kind).copy(), np.asarray(env.aux_dest).copy()
        for i, g in enumerate(before):
            if g is None:
                assert kind[i] == -1 and not bits[i].any()
                continue
            want = brute_label(g, int(cards[i]))
            counts["plays"] += 1
            if want is None:
                continue
            assert (int(kind[i]), bits[i].tolist()) == want, (g, int(cards[i]))
            counts["labels"] += kind[i] >= 0
            counts["positive"] += kind[i] == 1
            counts["tied"] += (bits[i] != 0).sum() > 1
            counts["ended"] += bool(kind[i] >= 0 and done[i])
        for i in np.flatnonzero(done):  # keep the envs on games with jokers
            env.set_game(int(i), joker_game(games))
            games += 1
    assert counts["labels"] > 300 and counts["positive"] > 50 and counts["tied"] > 20 and counts["ended"] > 5, counts


def test_label_sampling_draws_no_random_numbers_and_follows_its_rate():
    cfg, out = preset("smoke"), []
    for rate in (0.0, 0.25, 1.0):
        env, rng, kinds, obs = joker_env(cfg, seed=3, num_envs=16, order_label_rate=rate), np.random.default_rng(1), [], []
        for _ in range(120):
            env.step(*choose(env, rng))
            kinds.append(np.asarray(env.aux_kind).copy())
            obs.append(np.asarray(env.tok_f).sum())
        out.append((np.stack(kinds), obs))
    (k0, o0), (k1, o1), (k2, o2) = out
    assert o0 == o1 == o2  # the same games whatever is labelled
    n1, n2 = int((k1 >= 0).sum()), int((k2 >= 0).sum())
    assert (k0 == -1).all() and n2 > 100 and 0.1 * n2 < n1 < 0.5 * n2
    assert (k1[k1 >= 0] == k2[k1 >= 0]).all()  # a sampled label is the label
    with pytest.raises(ValueError):
        make_env(cfg, order_label_rate=1.5)


def test_order_gain_can_be_switched_off():
    cfg = preset("smoke")
    on, off = (joker_env(cfg, seed=5, num_envs=16, order_gain=x) for x in (True, False))
    rng_on, rng_off, seen = np.random.default_rng(2), np.random.default_rng(2), 0
    for _ in range(100):
        a, b = np.asarray(on.ops_f), np.asarray(off.ops_f)
        moves = np.asarray(on.ops)[..., 0] == az.OP_MOVE_JOKER
        assert not b[..., az.OP_ORDER_GAIN].any() and not a[..., az.OP_ORDER_GAIN][~moves].any()
        assert np.array_equal(a[..., : az.OP_ORDER_GAIN], b[..., : az.OP_ORDER_GAIN]) and np.array_equal(np.asarray(on.tok_f), np.asarray(off.tok_f))
        gain = a[..., az.OP_ORDER_GAIN]
        assert ((gain == 0) | ((gain > 2.9) & (gain <= 600))).all()
        seen += int((gain > 0).any(-1).sum())
        on.step(*choose(on, rng_on))
        off.step(*choose(off, rng_off))
    assert seen > 30
    # A single game shows the same column, and none when asked not to.
    g = joker_game(4)
    d_on, d_off = g.encode(8), g.encode(8, False, True, False)
    assert np.array_equal(d_on["ops_f"][:, : az.OP_ORDER_GAIN], d_off["ops_f"][:, : az.OP_ORDER_GAIN]) and not d_off["ops_f"][:, az.OP_ORDER_GAIN].any()


@functools.cache
def positives(n: int = 48):
    """Observations of ``n`` played hands whose label has a gaining move, with their destination bits."""
    cfg, rng = preset("smoke"), np.random.default_rng(3)
    env = joker_env(cfg, seed=11, num_envs=32, order_label_rate=1.0)
    rows, bits, games = [], [], 32
    while sum(len(b) for b in bits) < n:
        o = obs_from_numpy(env)
        o = Obs(**{f: getattr(o, f).clone() for f in Obs.__dataclass_fields__})
        _rew, done, _trunc = env.step(*choose(env, rng, weak=False))
        pos = torch.from_numpy(np.asarray(env.aux_kind) == 1)
        rows.append(Obs(**{f: getattr(o, f)[pos] for f in Obs.__dataclass_fields__}))
        bits.append(torch.from_numpy(np.asarray(env.aux_dest).copy())[pos])
        for i in np.flatnonzero(done):
            env.set_game(int(i), joker_game(games))
            games += 1
    return Obs(**{f: torch.cat([getattr(r, f) for r in rows])[:n] for f in Obs.__dataclass_fields__}), torch.cat(bits)[:n]


def test_destinations_do_not_read_the_gain():
    """The gain reaches the op logits only: the destination pointer of a joker move sees the op embedding without
    it, and a policy whose gain weights are zero (a migrated checkpoint) acts the same with the input on or off."""
    m, (o, _bits) = tiny(gain_weights=True), positives()
    blank = dataclasses.replace(o, ops_f=o.ops_f.clone())
    blank.ops_f[..., az.OP_ORDER_GAIN] = 0
    gain = o.ops_f[..., az.OP_ORDER_GAIN]
    ar = (gain > 0).any(-1).nonzero().squeeze(1)  # the states whose top candidate has a gaining move
    move, seq = gain[ar].argmax(-1), torch.full((len(ar), MAX_CARDS), -1)
    assert len(ar) > 5
    with torch.no_grad():
        (cls, tok, e_on, lg_on), (_, _, e_off, lg_off) = m.encode(o), m.encode(blank)
        d_on = m._dest_logits(cls[ar], tok[ar], e_on[ar, move], gain[ar, move], o.ops[ar, move], seq)[0]
        d_off = m._dest_logits(cls[ar], tok[ar], e_off[ar, move], blank.ops_f[ar, move, az.OP_ORDER_GAIN], o.ops[ar, move], seq)[0]
        assert (lg_on[ar, move] - lg_off[ar, move]).abs().min() > 1e-3 and torch.allclose(d_on, d_off, atol=1e-4)
        m.op_feat.weight[:, GAIN] = 0
        a, b = m.act(o, deterministic=True), m.act(blank, deterministic=True)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_order_loss_reaches_only_the_gain_column_and_the_destination_head():
    """Added to a PPO gradient, the loss changes those entries only (its graph runs through the op scorer, which
    gets none of it), by a step of at most ORDER_CLIP in norm."""
    m, (o, bits) = tiny(), positives()
    for coef in (1e-3, 100.0):
        for p in m.parameters():
            p.grad = torch.ones_like(p)
        add_order_grad(m, [(o, bits, None)], coef)
        added = {n: p.grad - 1 for n, p in m.named_parameters()}
        assert {n for n, d in added.items() if d.abs().sum() > 0} - {"dst_k.bias"} == ORDER_PARAMS
        assert added["op_feat.weight"][:, GAIN].any() and not added["op_feat.weight"][:, :GAIN].any()
        norm = math.sqrt(sum(float((d * d).sum()) for d in added.values()))
        assert norm <= ORDER_CLIP * 1.001 and (coef < 1 or norm > ORDER_CLIP * 0.99) and (coef > 1 or norm < 0.1 * ORDER_CLIP)

    def step(parts):
        for p in m.parameters():
            p.grad = torch.zeros_like(p)
        add_order_grad(m, parts, 1e-3)
        return torch.cat([p.grad.flatten() for p in m.parameters()])

    # The same step from encodings the caller already has (the PPO pass's), whole or in the chunks of a minibatch.
    enc = m.encode(o)
    chunk = lambda s: (Obs(**{f: getattr(o, f)[s] for f in Obs.__dataclass_fields__}), bits[s], tuple(x[s] for x in enc[:3]))  # noqa: E731
    whole = step([(o, bits, None)])
    assert whole.abs().sum() > 0 and torch.allclose(whole, step([(o, bits, enc)]), rtol=1e-4, atol=1e-9)
    assert torch.allclose(whole, step([chunk(slice(0, 20)), chunk(slice(20, 20)), chunk(slice(20, None))]), rtol=1e-4, atol=1e-9)


def test_order_loss_keeps_its_gradient_far_below_the_play():
    """A labelled move 500 nats below the best other op still gets a gradient of full size: the logit's pull is
    1 - sigmoid(margin), which is 1 there (a loss on the move's probability would have none left)."""
    (o, bits), pulls = positives(), []
    with torch.no_grad():  # a state with one labelled joker, below the best other op (one logit to scale)
        k = int((((bits != 0).sum(1) == 1) & (tiny().order_aux(o, bits)[1] < 0)).nonzero()[0])
    one, bits = Obs(**{f: getattr(o, f)[k : k + 1] for f in Obs.__dataclass_fields__}), bits[k : k + 1]
    for margin in (-2.0, -500.0):
        m = tiny()
        with torch.no_grad():
            scale = float(margin / m.order_aux(one, bits)[1])
            m.op_score[-1].weight *= scale  # scales every logit difference (and the logit's slope with it)
        loss, lead, _hit, _p = m.order_aux(one, bits)
        assert abs(float(lead) - margin) < 0.05 * abs(margin) and torch.isfinite(loss).all() and float(loss.detach()) > -margin
        (g,) = torch.autograd.grad(loss.sum(), m.op_feat.weight)
        assert torch.isfinite(g).all()
        pulls.append(float(g[:, GAIN].norm()) / abs(scale))
    assert 1.0 < pulls[1] / pulls[0] < 1.3  # 1 / (1 - sigmoid(-2)) = 1.14


def test_order_loss_teaches_the_labelled_moves():
    """Aux steps alone, through the gain column and the destination head: greedy play comes to make a labelled move
    to an allowed slot, and the moves stop rising once they lead by the band."""
    m, (o, bits) = tiny(), positives()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    hit0 = float(m.order_aux(o, bits)[2].float().mean())
    for _ in range(300):
        for p in m.parameters():
            p.grad = torch.zeros_like(p)
        add_order_grad(m, [(o, bits, None)], 1.0)
        opt.step()
    _loss, lead, hit, _p = m.order_aux(o, bits)
    assert hit0 < 0.2 and float(hit.float().mean()) > 0.8, (hit0, float(hit.float().mean()))
    assert float(lead.median()) < 2 * ORDER_BAND + 1
    moved = {n for n, p in m.named_parameters() if not torch.equal(p, before[n])}
    assert moved - {"dst_k.bias"} == ORDER_PARAMS and torch.equal(m.op_feat.weight[:, :GAIN], before["op_feat.weight"][:, :GAIN])


def order_cfg(tmp_path, name, monkeypatch, **ppo):
    """A smoke run on games with jokers that labels every played hand."""
    monkeypatch.setattr(train_mod, "make_env", joker_env)
    monkeypatch.setattr(collect_mod, "make_env", joker_env)
    cfg = preset("smoke")
    cfg.run_dir, cfg.name, cfg.device = str(tmp_path), name, "cpu"
    # Many envs, few steps: an env that loses its game goes on with a fresh one, which has no jokers.
    cfg.env.num_envs, cfg.ppo.rollout_len, cfg.ppo.total_steps, cfg.ppo.amp_cpu = 128, 16, 128 * 16 * 4, False
    cfg.env.order_label_rate = 1.0
    for k, v in ppo.items():
        setattr(cfg.ppo, k, v)
    return cfg


def spy_rollout(monkeypatch):
    """The fields of the last rollout ``train`` ran its advantages on."""
    real, seen = train_mod.gae, {}

    def spy(r, last_value, gamma, lam, value=None, cut=None):
        seen.update({k: v.cpu().clone() for k, v in vars(r).items() if isinstance(v, torch.Tensor) and k not in ("tok_i", "tok_f")})
        return real(r, last_value, gamma, lam, value, cut)

    monkeypatch.setattr(train_mod, "gae", spy)
    return seen


MODES = [("cpu", 0, 1, False)] + ([("cuda", 0, 1, False), ("cuda", 1, 1, False), ("cuda", 2, 1, False), ("cuda", 2, 2, True)] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device,fast,chunks,host", MODES)
def test_rollout_labels_are_those_of_the_stored_actions(tmp_path, monkeypatch, device, fast, chunks, host):
    """In every rollout path (the plain loop; the Collector with one env or two halves, act chunks, host storage)
    row [t, n] holds the label of the action stored at [t, n]: a twin env given the stored actions reports them."""
    cfg = order_cfg(tmp_path, "r", monkeypatch, fast_rollout=fast, act_chunks=chunks, rollout_obs_on_cpu=host)
    cfg.device = device
    seen = spy_rollout(monkeypatch)
    row = train(cfg, max_updates=1)
    halves = 2 if fast == 2 else 1
    nb, seed = cfg.env.num_envs // halves, cfg.env.seed
    twins = [joker_env(cfg, seed=seed + 7_000_001 * h if fast else seed, num_envs=nb) for h in range(halves)]
    cards = ((seen["card_seq"] >= 0) * (1 << seen["card_seq"].clamp(min=0))).sum(-1)
    for t in range(cfg.ppo.rollout_len):
        for h, twin in enumerate(twins):
            sl = slice(h * nb, (h + 1) * nb)
            assert np.array_equal(np.asarray(twin.ops), seen["ops"][t, sl].numpy()), t
            _rew, done, _trunc = twin.step(seen["op"][t, sl].numpy().astype(np.int32), cards[t, sl].numpy().astype(np.uint32), seen["dest"][t, sl].numpy().astype(np.int32))
            assert np.array_equal(done, seen["done"][t, sl].numpy() > 0)
            assert np.array_equal(np.asarray(twin.aux_kind), seen["aux_kind"][t, sl].numpy()), t
            assert np.array_equal(np.asarray(twin.aux_dest), seen["aux_dest"][t, sl].numpy()), t
    kind = seen["aux_kind"]
    assert row["order_pos"] == int((kind == 1).sum()) and row["order_zero"] == int((kind == 0).sum()) and row["order_pos"] + row["order_zero"] > 10


def test_labels_alone_change_nothing_about_the_update(tmp_path, monkeypatch):
    """With the loss off, a run that labels its played hands takes the same steps, bitwise, as one that does not
    (no random number is drawn for them), and a run with both off logs today's columns."""
    out = []
    for rate in (0.0, 1.0):
        cfg = order_cfg(tmp_path, f"l{rate}", monkeypatch)
        cfg.env.order_label_rate = rate
        train(cfg, max_updates=2)
        with open(tmp_path / cfg.name / "log.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        out.append((rows, read_checkpoint(tmp_path / cfg.name / "last.pt")["model"]))
    (r0, w0), (r1, w1) = out
    assert list(r0[0]) == LOG_FIELDS and list(r1[0]) == LOG_FIELDS + ORDER_FIELDS
    assert int(r1[0]["order_pos"]) > 0 and int(r1[0]["order_zero"]) > 0 and 0 <= float(r1[0]["order_stray"]) <= 1
    assert all(torch.equal(w0[k], w1[k]) for k in w0)
    assert all(a[k] == b[k] for a, b in zip(r0, r1) for k in LOG_FIELDS if k not in ("fps", "rollout_s"))


def test_a_rollout_without_labels_trains(tmp_path):
    """Fresh games have no jokers to order: nothing is labelled, the loss has nothing to add, the counts are zero."""
    cfg = preset("smoke")
    cfg.run_dir, cfg.device, cfg.env.order_label_rate, cfg.ppo.aux_order = str(tmp_path), "cpu", 1.0, 0.05
    row = train(cfg, max_updates=1)
    assert row["order_pos"] == row["order_zero"] == 0 and math.isnan(row["order_lead"]) and math.isnan(row["order_stray"])


def test_ordering_loss_in_training_moves_only_its_parameters_and_resumes(tmp_path, monkeypatch):
    """One optimizer step with and without the loss: PPO's step is the same, and the loss adds its own to the gain
    column and the destination head only. The run then resumes, and positives survive ``ppo.learn_frac``."""
    weights, seen = [], spy_rollout(monkeypatch)
    for coef in (0.0, 0.05):
        cfg = order_cfg(tmp_path, f"a{coef}", monkeypatch, aux_order=coef, minibatches=1)
        row = train(cfg, max_updates=1)
        assert row["order_pos"] > 0 and math.isfinite(row["order_lead"])
        weights.append(read_checkpoint(tmp_path / cfg.name / "last.pt")["model"])
    off, on = weights
    changed = {k for k in off if not torch.equal(off[k], on[k])} - {"dst_k.bias"}
    # The gain column only learns from positives whose labelled move carries a gain (the label is on the hand
    # played, the input on the top candidate).
    ops, gain = seen["ops"].long(), seen["ops_f"][..., az.OP_ORDER_GAIN]
    labelled = (ops[..., 0] == az.OP_MOVE_JOKER) & (seen["aux_dest"].long().gather(2, ops[..., 7].clamp(0, az.ORDER_JOKERS - 1)) != 0)
    fed = bool(((seen["aux_kind"] == 1)[..., None] & labelled & (gain > 0)).any())
    assert changed == ORDER_PARAMS - ({"op_feat.weight"} if not fed else set())
    assert torch.equal(off["op_feat.weight"][:, :GAIN], on["op_feat.weight"][:, :GAIN])
    assert train(cfg, resume=str(tmp_path / cfg.name / "last.pt"), max_updates=1)["update"] == 2
    # A quarter of each minibatch is trained on: every positive must be among it.
    cfg = order_cfg(tmp_path, "frac", monkeypatch, aux_order=0.05, learn_frac=0.25)
    real, fed = train_mod.add_order_grad, []
    monkeypatch.setattr(train_mod, "add_order_grad", lambda m, parts, coef: fed.append(sum(len(p[1]) for p in parts)) or real(m, parts, coef))
    train(cfg, max_updates=1)
    assert sum(fed) == int((seen["aux_kind"] == 1).sum()) * cfg.ppo.epochs > 0


def test_order_settings_are_validated():
    cfg = preset("smoke")
    cfg.ppo.aux_order = 0.05
    with pytest.raises(ValueError):
        cfg.validate()  # no labels to learn from
    cfg.env.order_label_rate = 1 / 16
    cfg.validate()
    cfg.env.order_label_rate, cfg.ppo.aux_order = 2.0, 0.0
    with pytest.raises(ValueError):
        cfg.validate()
    cfg.env.order_label_rate, cfg.ppo.aux_order = 0.0, -1.0
    with pytest.raises(ValueError):
        cfg.validate()
