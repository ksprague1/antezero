"""Card-pick masks of the plays, on the policy's device.

A play declares a poker hand (PLAY_<hand>; PLAY = "Any") and then picks its cards one at a time. The engine owns the
rule, "core-then-free": a declared hand h (or Any = HS_ANY = 12) has a core (Pair: two cards of one rank; Two Pair: two
ranks with two, or four of one; Full House: three + two; Flush: 5 (Four Fingers: 4) of one suit; Straight: the straight
table on the ranks; Straight Flush / Flush House / Flush Five: one suit's cards alone form a Straight / Full House /
Five of a Kind; ...). With picks P (+ forced cards) and r = 5 - |P| (the boss's max for a play), card c may be picked
iff the core is still reachable from P + c within r - 1 more cards; the picks may stop iff P contains the core (and
|P| >= the boss's min, The Psychic's 5). Once the core is in hand with room to spare every card is a free kicker;
upgrades are allowed (a Pair declared, three Kings played: the game scores Three of a Kind). The engine checks every
pick with this rule (BalatroEnv.pick_mask).

The policy samples the picks one at a time on its device, so it computes the same masks itself (HandMasker), from the
env's hand context (infos["hand_ctx"]: per card the rank, the flush suits after wild / Smeared / debuff, stone, forced;
Four Fingers, Shortcut; the boss's card-count range): per pick one scatter-add counts the picks per (channel, rank)
(channel 0: every card; 1 + q: suit q) and per suit, then a few batched integer-valued ops give the cost (cards still
needed) of the row's declared hand and the ranks / suits that lower it. No subset enumeration, no host
synchronisation (every row is computed; rows without a declared hand are ignored).

Generic per-pick masks (every selection action): Policy.pick_cards takes a mask provider at rollout and returns each
pick step's allow mask (pack_allow: 24 card bits + bit 24 stop), stored with the picks; the teacher-forced pass replays
the stored masks (unpack_allow), so rollout and training use identical masks by construction. DISCARD, targeting
consumables and PICK produce the spec mask (eligible - picked while count < hi; stop once count >= lo), the plays
HandMasker's.
"""
import numpy as np
import torch

from .constants import PLAY_HAND_ACTIONS

MAX_HAND = 24
HS_ANY = 12
STOP_BIT = 1 << MAX_HAND
INF = 64.0
_CONST = {}


def _const(dev):
    """Per-device constant tables: card bits, pack weights, rank powers, the straight minimal sets."""
    t = _CONST.get(dev)
    if t is None:
        cb = torch.ones(MAX_HAND, dtype=torch.long) << torch.arange(MAX_HAND)
        t = {"cb": cb.to(dev), "w": torch.cat([cb, torch.tensor([STOP_BIT])]).to(dev),
             "p2": (torch.ones(13, dtype=torch.long) << torch.arange(13)).to(dev),
             "r13": torch.arange(13, device=dev)}
        _CONST[dev] = t
    return t


_MTAB = {}


def straight_table(dev, modes=(0, 1, 2, 3)):
    """The minimal straight sets of these modes (Four Fingers * 2 + Shortcut) in one table: (sets (M,) long rank bits,
    valid (4, M) bool: the sets of each mode, bits (M, 13) float, bits.T (13, M)). Mode 0 alone: 10 sets; all: 204."""
    key = (dev, tuple(sorted(set(modes))))
    t = _MTAB.get(key)
    if t is None:
        mins = straight_mins()
        mu = sorted(set(x for m in key[1] for x in mins[m]))
        mt = torch.tensor(mu, dtype=torch.long)
        valid = torch.tensor([[x in set(m) for x in mu] for m in mins], dtype=torch.bool).reshape(4, len(mu))
        bits = ((mt[:, None] >> torch.arange(13)) & 1).float()
        t = (mt.to(dev), valid.to(dev), bits.to(dev), bits.T.contiguous().to(dev))
        _MTAB[key] = t
    return t


def card_bits(dev):
    return _const(dev)["cb"]


def pack_allow(allow):
    """(B, 25) bool allow mask of one pick step (24 cards, stop) -> (B,) int32 bits (bit 24: stop)."""
    return (allow.long() * _const(allow.device)["w"]).sum(-1).int()


def unpack_allow(bits):
    """(..., ) int32 packed allow masks -> (..., 25) bool."""
    sh = torch.arange(MAX_HAND + 1, device=bits.device)
    return ((bits.long()[..., None] >> sh) & 1).bool()


_LUT = None


def straight_lut():
    """Pure-NumPy straight table: (4, 8192) bool, mode = Four Fingers * 2 + Shortcut."""
    global _LUT
    if _LUT is None:
        out = np.zeros((4, 8192), dtype=np.bool_)
        line_bits = [12] + list(range(13))  # A-low (bit 12), 2..A (bits 0..12)
        x = np.arange(8192, dtype=np.int32)
        for mode in range(4):
            need = 4 if (mode & 2) else 5
            max_step = 2 if (mode & 1) else 1
            mins = []

            def dfs(last_idx, count, mask):
                if count == need:
                    mins.append(mask)
                    return
                for nxt in range(last_idx + 1, min(len(line_bits), last_idx + 1 + max_step)):
                    dfs(nxt, count + 1, mask | (1 << line_bits[nxt]))

            for start in range(len(line_bits)):
                dfs(start, 1, 1 << line_bits[start])
            for m in mins:
                out[mode] |= (x & m) == m
        _LUT = out
    return _LUT


_MINS = None


def straight_mins():
    """Per mode, the minimal rank sets (bit r - 2) the straight table accepts (it is monotone: a rank set is a straight
    iff it contains one), in the engine's order."""
    global _MINS
    if _MINS is None:
        L = straight_lut()
        _MINS = []
        for m in range(4):
            s = L[m]
            _MINS.append([x for x in range(8192) if s[x] and not any(s[x & ~(1 << b)] for b in range(13) if x >> b & 1)])
    return _MINS


# per declared hand (handlist index 0..11, HS_ANY): (family, suited channels, k / ka, kb)
#   family 0 kind (k of a rank), 1 groups (ka of a rank + kb of another; Two Pair also kind 4), 2 flush, 3 straight,
#   4 any card (High Card, Any)
_HAND_SPEC = {0: (0, True, 5, 0), 1: (1, True, 3, 2), 2: (0, False, 5, 0), 3: (3, True, 0, 0), 4: (0, False, 4, 0),
              5: (1, False, 3, 2), 6: (2, False, 0, 0), 7: (3, False, 0, 0), 8: (0, False, 3, 0), 9: (1, False, 2, 2),
              10: (0, False, 2, 0), 11: (4, False, 0, 0), 12: (4, False, 0, 0)}
_SPEC_T = {}


def _spec(dev):
    t = _SPEC_T.get(dev)
    if t is None:
        rows = [_HAND_SPEC[h] for h in range(13)] + [(4, False, 0, 0)]   # (index 13: no declared hand)
        f = torch.tensor([r[0] for r in rows])
        suited = torch.tensor([r[1] for r in rows])
        k = torch.tensor([float(r[2] if r[0] == 0 else 4 if r[0] == 1 and r[2] == 2 else 0) for r in rows])
        ka = torch.tensor([float(r[2] if r[0] == 1 else 0) for r in rows])
        kb = torch.tensor([float(r[3] if r[0] == 1 else 0) for r in rows])
        kind_on = (f == 0) | ((f == 1) & (ka == 2))   # (Two Pair: also four of one rank)
        t = tuple(x.to(dev) for x in (f, suited, k, ka, kb, kind_on))
        _SPEC_T[dev] = t
    return t


class HandMasker:
    """Per-pick legality masks of declared-hand verbs for a batch (rows without a declared hand are not used).

    ctx: (B, HS_CTX) int hand contexts; target: (B,) declared hand (0..11, HS_ANY) or -1 (not a hand verb); boss: the
    boss's card-count range applies (a play); ctx_host: optional, the same contexts as a host array (numpy), read
    for the Four Fingers / Shortcut modes in the batch (a smaller straight table; same masks). card_stop(chosen) -> (card_ok (B, 24), stop_ok (B,)) for picks
    chosen (B, 24) bool (forced cards excluded, as pick_cards keeps them); `active` (B,) bool: the rows with a
    declared hand. No host synchronisation."""

    def __init__(self, ctx, target, boss, ctx_host=None):
        dev = ctx.device
        C = _const(dev)
        c = ctx.long()
        Bn = c.shape[0]
        cards = c[:, :MAX_HAND]
        self.active = target >= 0
        tg = torch.where(self.active, target.long(), 13)
        f, suited, k, ka, kb, kind_on = (x[tg] for x in _spec(dev))
        self.k, self.ka, self.kb = k[:, None, None], ka[:, None, None], kb[:, None, None]
        self.any_on = f == 4
        present = ((cards >> 10) & 1).bool()
        rank = cards & 15
        has = present & (rank >= 2)
        self.ridx = torch.where(has, rank - 2, 13)                                  # (B, 24) 0..12, 13: none
        sb = ((cards[..., None] >> torch.arange(4, device=dev) + 4) & 1).bool() & present[..., None]   # (B, 24, 4)
        self.suit = sb
        self.inch = torch.cat([has[:, None, :], (sb & has[..., None]).transpose(1, 2)], 1)   # (B, 5, 24)
        self.present = present
        self.forced = ((cards >> 9) & 1).bool() & present
        # each card's count slots: (channel, rank) = channel * 13 + rank (65 + q: suit q; 69: none), so one
        # scatter-add of the picks counts them per channel and rank and per suit
        q4 = torch.arange(4, device=dev)
        slot = torch.cat([torch.where(self.inch, torch.arange(5, device=dev)[:, None] * 13 + self.ridx[:, None, :].clamp(max=12),
                                      69).transpose(1, 2), torch.where(sb, 65 + q4, 69)], 2)   # (B, 24, 9)
        self.slot = slot.reshape(Bn, MAX_HAND * 9)
        tot = torch.zeros(Bn, 70, device=dev).scatter_add_(1, self.slot, present.float()[..., None].expand(-1, -1, 9)
                                                             .reshape(Bn, -1))
        self.tot, self.stot = tot[:, :65].view(Bn, 5, 13), tot[:, 65:69]
        chan = torch.zeros(Bn, 5, dtype=torch.bool, device=dev)
        chan[:, 0] = True
        chan = torch.where(suited[:, None], ~chan, chan)[..., None]                # (B, 5, 1) the hand's channels
        flags = c[:, 25]
        ff, sc = flags & 1, (flags >> 1) & 1
        self.need = (5 - ff).float()[:, None]
        if boss:
            self.lo, self.budget = c[:, 28].float(), c[:, 29].float()
        else:
            self.lo, self.budget = torch.ones(Bn, device=dev), torch.full((Bn,), 5.0, device=dev)
        # what does not depend on the picks: per family, 0 where a (channel, rank) / suit / straight set can be
        # completed from the hand, else INF (added to the deficits)
        pen = lambda ok: torch.where(ok, 0.0, INF)   # noqa: E731
        tot = self.tot
        self.kpen = pen((tot >= self.k) & chan & kind_on[:, None, None])           # (B, 5, 13)
        g_on = chan & (f == 1)[:, None, None]
        self.apen, self.bpen = pen((tot >= self.ka) & g_on), pen((tot >= self.kb) & g_on)
        self.spen = pen((self.stot >= self.need) & (f == 2)[:, None])              # (B, 4)
        self.anypen = pen(self.any_on)                                              # (B,)
        # straights: the minimal rank sets of the row's mode, feasible per channel (every rank in hand there). One
        # table for the batch: the modes in ctx_host (the same contexts on the host, numpy: no device sync), or all
        fl = None if ctx_host is None else np.asarray(ctx_host)[:, 25]
        modes = (0, 1, 2, 3) if fl is None else np.unique((fl & 1) * 2 + ((fl >> 1) & 1)).tolist()
        mt, mvalid, self.mbits, self.mbT = straight_table(dev, modes)
        have = ((self.tot > 0).long() * C["p2"]).sum(-1)                            # (B, 5)
        mfeas = ((mt & ~have[..., None]) == 0) & mvalid[ff * 2 + sc][:, None, :] & chan & (f == 3)[:, None, None]
        self.mbase = self.mbits.sum(-1) + pen(mfeas)                                # (B, 5, M) ranks per set (+ INF)
        self.zero1 = torch.zeros(Bn, 1, dtype=torch.bool, device=dev)

    def card_stop(self, chosen):
        """(card_ok (B, 24), stop_ok (B,)) after picks chosen (B, 24) bool (forced cards excluded)."""
        dev = chosen.device
        Bn = chosen.shape[0]
        with torch.autocast(dev.type, enabled=False):   # (exact small integers in fp32, also inside a bf16 autocast)
            P = chosen | self.forced
            Pf = P.float()
            cnt_all = torch.zeros(Bn, 70, device=dev).scatter_add_(1, self.slot, Pf[..., None].expand(-1, -1, 9)
                                                                   .reshape(Bn, -1))   # (B, 70) the picks per slot
            cnt = cnt_all[:, :65].view(Bn, 5, 13)
            n = Pf.sum(1)
            r1 = (self.budget - n)[:, None]                                         # (B, 1) cards left in the budget
            r = r1[..., None]
            # cost = the fewest cards still needed; use = the (channel, rank) / suit slots that lower it to <= r - 1
            # k of a kind (and Two Pair's four of one rank): deficit per (channel, rank)
            d = (self.k - cnt).clamp_(min=0).add_(self.kpen)
            cost = d.amin((1, 2))
            use = d <= r
            # groups: ka of one rank + kb of another (Full House, Flush House, Two Pair)
            da = (self.ka - cnt).clamp_(min=0).add_(self.apen)
            db = (self.kb - cnt).clamp_(min=0).add_(self.bpen)
            va, ia = da.topk(2, -1, largest=False)
            vb, ib = db.topk(2, -1, largest=False)
            r13 = _const(dev)["r13"]
            oa = torch.where(r13 == ia[..., :1], va[..., 1:], va[..., :1])          # the best other rank's deficit
            ob = torch.where(r13 == ib[..., :1], vb[..., 1:], vb[..., :1])
            cost = torch.minimum(cost, (da + ob).amin((1, 2)))
            use |= torch.minimum((da - 1).clamp_(min=0) + ob, (db - 1).clamp_(min=0) + oa) <= r - 1
            # straights: the feasible minimal rank set missing the fewest ranks
            dm = self.mbase - ((cnt > 0).float() @ self.mbT)                       # (B, 5, M) ranks missing
            cost = torch.minimum(cost, dm.amin((1, 2)))
            use |= (((dm <= r).float() @ self.mbits) > 0) & (cnt == 0)
            # flush: per suit
            ds = (self.need - cnt_all[:, 65:69]).clamp_(min=0).add_(self.spen)
            cost = torch.minimum(cost, ds.amin(1))
            # any card (High Card, Any)
            cost = torch.minimum(cost, (n == 0).float() + self.anypen)
            # cards: free while cost <= r - 1 (or any card), else the ones in a slot that lowers the cost
            us = torch.cat([use.view(Bn, 65), ds <= r1, self.zero1], 1)            # (B, 70)
            hit = us.gather(1, self.slot).view(Bn, MAX_HAND, 9).any(-1)
            rr = r1[:, 0]
            free = (cost <= rr - 1) | self.any_on
            card_ok = self.present & ~P & (rr >= 1)[:, None] & (free[:, None] | ((cost <= rr)[:, None] & hit))
            stop_ok = (cost == 0) & (n >= self.lo) & (rr >= 0)
        return card_ok, stop_ok


def verb_targets(sel_verbs):
    """Per selection verb (sel_verbs order): the declared hand (0..11, HS_ANY) or -1; and the verbs whose picks the
    picker is told the hand formed so far of (the plays and DISCARD)."""
    ht, he = [], []
    for name in sel_verbs:
        h = HS_ANY if name == "PLAY" else PLAY_HAND_ACTIONS.index(name) if name in PLAY_HAND_ACTIONS else -1
        ht.append(h)
        he.append(h >= 0 or name == "DISCARD")
    return ht, he
