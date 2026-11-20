import itertools

import antezero_sim as az
import torch

from antezero_agent import effects
from antezero_agent.config import preset
from antezero_agent.critic import Critic
from antezero_agent.model import Net

TRIGGERS = ["on_scored", "on_held", "on_hand", "on_discard", "on_round_end", "on_blind", "on_shop", "on_sell", "on_consumable", "on_boss", "passive", "sell_to_use"]
# The basic hands and the bigger hands that contain them (the simulator's names).
HOLDS = {
    "pair": {"Pair", "Two Pair", "Three of a Kind", "Full House", "Four of a Kind"},
    "two_pair": {"Two Pair", "Full House"},
    "three": {"Three of a Kind", "Full House", "Four of a Kind"},
    "straight": {"Straight", "Straight Flush"},
    "flush": {"Flush", "Straight Flush"},
}
HAND_JOKERS = ["j_jolly", "j_zany", "j_mad", "j_crazy", "j_droll", "j_sly", "j_wily", "j_clever", "j_devious", "j_crafty", "j_duo", "j_trio", "j_order", "j_tribe"]


def col(name):
    return effects.matrix()[:, effects.COLS.index(name)]


def test_every_card_has_a_row():
    m = effects.matrix()  # asserts the keys against the simulator's
    kinds = {"is_joker": 150, "is_tarot": 22, "is_planet": 12, "is_spectral": 18, "is_voucher": 32, "is_pack": 32}
    assert {k: int(col(k).sum()) for k in kinds} == kinds
    assert int((m.abs().sum(1) > 0).sum()) == sum(kinds.values())
    rarity = torch.stack([col(r) for r in effects.RARITIES], 1)
    assert bool((rarity[col("is_joker") > 0].sum(1) >= 1).all()), "a joker without a rarity"
    when = torch.stack([col(t) for t in TRIGGERS], 1).sum(1)
    assert bool((when[col("is_joker") > 0] > 0).all()), "a joker without a trigger"


def played(g, code):
    """-> (hand type played, chips scored) of a play that leaves the blind standing, else None."""
    w = g.clone()
    before = w.canonical()
    w.step(code)
    if w.phase != "round":
        return None
    after = w.canonical()
    (hand,) = [k.split(".")[1] for k in after if k.endswith(".played") and after[k] != before[k]]
    return hand, w.chips


def test_hand_type_rows_match_the_simulator():
    """A hand-type joker raises the score of exactly the hands that hold its hand."""
    ids, m = {k: i for k, i, _ in az.centers()}, effects.matrix()
    seen = {key: set() for key in HAND_JOKERS}
    for seed in range(60):
        g = az.Game(f"FX{seed}", "b_red", 1)
        g.step("select")
        n = int(g.canonical()["hand.len"])
        by_hand = {}
        for k in (5, 4, 2):
            for cards in itertools.combinations(range(n), k):
                code = f"play {sum(1 << i for i in cards)}"
                if (r := played(g, code)) is not None:
                    by_hand.setdefault(r[0], (code, r[1]))
        for key in HAND_JOKERS:
            (named,) = [h for h in HOLDS if m[ids[key], effects.COLS.index(f"h_{h}")] > 0]
            w = g.clone()
            w.inject(f"inject_joker {ids[key] - ids['j_joker']} 0 0")
            for hand, (code, chips) in by_hand.items():
                with_joker = played(w, code)
                gained = with_joker is None or with_joker[1] > chips
                assert gained == (hand in HOLDS[named]), (key, hand, seed, code)
                seen[key].add(hand)
    for key in HAND_JOKERS:
        (named,) = [h for h in HOLDS if m[ids[key], effects.COLS.index(f"h_{h}")] > 0]
        assert seen[key] & HOLDS[named], f"no hand for {key} came up"


def test_effect_weights_start_at_zero_and_come_last():
    cfg = preset("smoke")
    for net in (Net(cfg.model), Critic(cfg.model)):
        assert float(net.fx_lin.weight.detach().abs().sum()) == 0.0
        assert list(net.state_dict())[-1] == "fx_lin.weight"


def test_a_checkpoint_keeps_the_table_it_was_trained_with():
    cfg = preset("smoke")
    old = torch.ones(az.N_CENTERS, 7)
    for kind in (Net, Critic):
        state = kind(cfg.model, old).state_dict()
        net = kind(cfg.model, state["fx"])
        net.load_state_dict(state)
        assert torch.equal(net.fx, old) and net.fx_lin.weight.shape[1] == 7


def test_the_table_can_be_switched_off():
    cfg = preset("smoke")
    cfg.model.effects = False
    for net in (Net(cfg.model), Critic(cfg.model)):
        assert net.fx.shape == effects.matrix().shape and not bool(net.fx.any())
