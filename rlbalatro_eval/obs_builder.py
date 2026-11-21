"""Build RLBalatro observations, action masks, selection specs, and hand contexts from Balatro Lua game state."""
import math
import numpy as np

from .constants import (
    ACTIONS,
    CARD_FEATURES,
    COUNT_NAMES,
    HANDS,
    HS_CTX,
    IDS,
    N_HAND,
    PLAY_HAND_ACTIONS,
    ROUND_CTX_FEATURES,
    SEL_FEATURES,
    SEL_VERBS,
    VOUCHER_KEYS,
    layout as _layout,
)

CENTER_IDS = IDS["center"]
BLIND_IDS = IDS["blind"]
TAG_IDS = IDS["tag"]

SUIT_MAP = {"Spades": 1, "Hearts": 2, "Clubs": 3, "Diamonds": 4}
EDITION_MAP = {"foil": 1, "holo": 2, "polychrome": 3, "negative": 4}
SEAL_MAP = {"Gold": 1, "Red": 2, "Blue": 3, "Purple": 4}
STATE_INDEX = {
    "BLIND_SELECT": 0,
    "SELECTING_HAND": 1,
    "ROUND_EVAL": 2,
    "SHOP": 3,
    "TAROT_PACK": 4,
    "PLANET_PACK": 5,
    "SPECTRAL_PACK": 6,
    "STANDARD_PACK": 7,
    "BUFFOON_PACK": 8,
    "GAME_OVER": 9,
}

# Consumables that require hand-card selection: key -> (min_cards, max_cards)
CONS_HAND_SELECT = {
    "c_magician": (1, 2),
    "c_empress": (1, 2),
    "c_heirophant": (1, 2),
    "c_lovers": (1, 1),
    "c_chariot": (1, 1),
    "c_justice": (1, 1),
    "c_devil": (1, 1),
    "c_tower": (1, 1),
    "c_strength": (1, 2),
    "c_hanged_man": (1, 2),
    "c_death": (2, 2),
    "c_star": (1, 3),
    "c_moon": (1, 3),
    "c_sun": (1, 3),
    "c_world": (1, 3),
    "c_aura": (1, 1),
    "c_talisman": (1, 1),
    "c_deja_vu": (1, 1),
    "c_trance": (1, 1),
    "c_medium": (1, 1),
    "c_cryptid": (1, 1),
}

# Consumables that need at least 1 card in hand (no selection, random targets)
CONS_NEED_HAND = {"c_familiar", "c_grim", "c_incantation", "c_sigil", "c_ouija", "c_immolate"}
# Consumables that need a joker
CONS_NEED_JOKER = {"c_wheel_of_fortune", "c_ankh", "c_hex"}
# Consumables that need a free joker slot
CONS_NEED_JOKER_SLOT = {"c_judgement", "c_wraith", "c_soul"}

GROUPS, OBS_SIZE = _layout()
GROUP_MAP = {name: (off, r, c) for name, off, r, c in GROUPS}
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
SEL_VERB_IDX = {v: i for i, v in enumerate(SEL_VERBS)}


def _suit_mask(card, smeared=False):
    """4-bit suit mask (bit 0=S, 1=H, 2=C, 3=D) matching Balatro's suit rules."""
    if card.get("debuff") or card.get("center_key") == "m_stone":
        return 0
    if card.get("center_key") == "m_wild":
        return 0b1111
    s = SUIT_MAP.get(card.get("suit", ""), 0)
    if s == 0:
        return 0
    if smeared:
        return 0b0101 if s in (1, 3) else 0b1010
    return 1 << (s - 1)


def _encode_card(card, state_ctx):
    """Encode one card dict into a 39-float CARD_FEATURES row."""
    row = np.zeros(len(CARD_FEATURES), dtype=np.float32)
    if not card:
        return row
    ckey = card.get("center_key") or "c_base"
    is_stone = ckey == "m_stone"
    rank = 0 if is_stone else int(card.get("rank") or 0)
    suit = 0 if is_stone else SUIT_MAP.get(card.get("suit", ""), 0)
    ed = EDITION_MAP.get(card.get("edition", ""), 0)
    seal = SEAL_MAP.get(card.get("seal", ""), 0)
    debuff = 1.0 if card.get("debuff") else 0.0
    sm = _suit_mask(card, smeared=state_ctx.get("smeared", False))

    row[0] = 1.0
    row[1] = float(CENTER_IDS[ckey] + 1) if ckey in CENTER_IDS else 0.0
    row[2] = float(rank)
    row[3] = float(suit)
    row[4] = float(ed)
    row[5] = float(seal)
    row[6] = debuff
    row[7] = 1.0 if card.get("face_down") else 0.0
    row[8] = 1.0 if card.get("selected") else 0.0
    row[9] = float(card.get("cost") or 0.0)
    row[10] = float(card.get("sell_cost") or 0.0)
    row[11] = 1.0 if card.get("eternal") else 0.0
    row[12] = float(card.get("perishable") or 0.0)
    row[13] = 1.0 if card.get("rental") else 0.0
    row[14] = float(card.get("mult") or 0.0)
    row[15] = float(card.get("x_mult") if card.get("x_mult") is not None else 1.0)
    row[16] = float(card.get("chips") or 0.0)
    row[17] = float(card.get("extra0") or 0.0)
    row[18] = float(card.get("extra1") or 0.0)
    row[19] = float(card.get("times_played") or 0.0)
    row[22] = float(card.get("joker_value") or 0.0)
    row[23] = float(card.get("joker_count") or 0.0)

    has_rank = rank >= 2 and not is_stone
    pareidolia = state_ctx.get("pareidolia", False)
    row[24] = 1.0 if (not is_stone and (rank in (11, 12, 13) or (pareidolia and has_rank))) else 0.0
    row[25] = 1.0 if (sm & 1) else 0.0
    row[26] = 1.0 if (sm & 2) else 0.0
    row[27] = 1.0 if (sm & 4) else 0.0
    row[28] = 1.0 if (sm & 8) else 0.0
    row[29] = 1.0 if has_rank else 0.0
    row[30] = 1.0 if card.get("forced") else 0.0
    if has_rank:
        row[31] = 1.0 if rank in (2, 3, 5, 8, 14) else 0.0
        row[32] = 1.0 if rank in (2, 4, 6, 8, 10) else 0.0
        row[33] = 1.0 if rank in (3, 5, 7, 9, 14) else 0.0
        row[34] = 1.0 if rank in (2, 3, 4, 5) else 0.0
    idol_r, idol_s = state_ctx.get("idol_rank", 0), state_ctx.get("idol_suit", 0)
    mail_r = state_ctx.get("mail_rank", 0)
    anc_s = state_ctx.get("ancient_suit", 0)
    cas_s = state_ctx.get("castle_suit", 0)
    row[35] = 1.0 if (has_rank and rank == idol_r and (sm & (1 << (idol_s - 1)) if idol_s else False)) else 0.0
    row[36] = 1.0 if (has_rank and rank == mail_r) else 0.0
    row[37] = 1.0 if (anc_s and (sm & (1 << (anc_s - 1)))) else 0.0
    row[38] = 1.0 if (cas_s and (sm & (1 << (cas_s - 1)))) else 0.0
    return row


def _can_use_consumable(card, state, n_hand, n_jokers, joker_limit, n_cons):
    """Check whether a consumable card is currently usable under Balatro's rules."""
    if not card:
        return False
    ckey = card.get("center_key", "")
    in_round = state == "SELECTING_HAND"
    if ckey in CONS_HAND_SELECT:
        lo, _ = CONS_HAND_SELECT[ckey]
        return n_hand >= lo
    if ckey in CONS_NEED_HAND:
        return n_hand >= 1
    if ckey in CONS_NEED_JOKER:
        return n_jokers >= 1
    if ckey in CONS_NEED_JOKER_SLOT:
        return n_jokers < joker_limit
    if ckey == "c_fool":
        return bool(card.get("can_use", False))
    if ckey in ("c_emperor", "c_high_priestess"):
        return card.get("can_use", True)
    if ckey == "c_ectoplasm":
        return n_jokers >= 1 and bool(card.get("can_use", True))
    # Planets, Hermit, Temperance, Black Hole, etc.
    if in_round or state in ("SHOP", "BLIND_SELECT", "ROUND_EVAL", "TAROT_PACK", "PLANET_PACK", "SPECTRAL_PACK"):
        return bool(card.get("can_use", True))
    return False


def build_observation(gs):
    """Convert a serialized Balatro Lua game state dict `gs` into:
    (x_flat (obs_size + 44,), action_mask (152,), sel_specs (32, 3), hand_ctx (32,)).
    """
    buf = np.zeros(OBS_SIZE, dtype=np.float32)

    def set_group(name, arr):
        off, r, c = GROUP_MAP[name]
        flat = arr.reshape(-1)
        buf[off:off + len(flat)] = flat

    state_name = gs.get("state", "BLIND_SELECT")
    st_idx = STATE_INDEX.get(state_name, 0)

    jokers = gs.get("jokers") or []
    hand = gs.get("hand") or []
    consumables = gs.get("consumables") or []
    shop = gs.get("shop") or []
    vouchers = gs.get("vouchers") or []
    boosters = gs.get("boosters") or []
    pack = gs.get("pack") or []
    deck = gs.get("deck") or []
    tags = gs.get("tags") or []

    joker_keys = {j.get("center_key") for j in jokers if not j.get("debuff")}
    smeared = "j_smeared" in joker_keys
    four_fingers = "j_four_fingers" in joker_keys
    shortcut = "j_shortcut" in joker_keys
    pareidolia = "j_pareidolia" in joker_keys

    state_ctx = {
        "smeared": smeared,
        "pareidolia": pareidolia,
        "idol_rank": int(gs.get("idol_rank") or 0),
        "idol_suit": SUIT_MAP.get(gs.get("idol_suit", ""), 0),
        "mail_rank": int(gs.get("mail_rank") or 0),
        "ancient_suit": SUIT_MAP.get(gs.get("ancient_suit", ""), 0),
        "castle_suit": SUIT_MAP.get(gs.get("castle_suit", ""), 0),
    }

    # 1. scalars (48,)
    sc = np.zeros(48, dtype=np.float32)
    sc[st_idx] = 1.0
    sc[10] = float(gs.get("ante") or 1)
    sc[11] = float(gs.get("round") or 0)
    sc[12] = float(gs.get("dollars") or 0)
    sc[13] = float(gs.get("hands_left") or 0)
    sc[14] = float(gs.get("discards_left") or 0)
    sc[15] = float(gs.get("chips") or 0)
    sc[16] = float(gs.get("blind_chips") or 0)
    blind_key = gs.get("blind_key") or ""
    sc[17] = float(BLIND_IDS[blind_key] + 1) if blind_key in BLIND_IDS else 0.0
    sc[18] = 1.0 if gs.get("blind_disabled") else 0.0
    sc[20] = float(gs.get("hand_limit") or 8)
    sc[21] = float(gs.get("joker_limit") or 5)
    sc[22] = float(gs.get("consumable_limit") or 2)
    sc[23] = float(gs.get("reroll_cost") or 5)
    sc[24] = float(gs.get("interest_cap") or 25)
    sc[25] = float(gs.get("bankrupt_at") or 0)
    blind_on_deck = {"Small": 1.0, "Big": 2.0, "Boss": 3.0}.get(gs.get("blind_on_deck", ""), 1.0)
    sc[26] = blind_on_deck
    boss_key = gs.get("boss_key") or ""
    sc[27] = float(BLIND_IDS[boss_key] + 1) if boss_key in BLIND_IDS else 0.0
    sc[28] = float(len(deck))
    sc[29] = float(gs.get("playing_cards_count") or len(deck) + len(hand))
    sc[30] = float(gs.get("pack_choices") or 0)
    sc[33] = float(gs.get("skips") or 0)
    sc[34] = float(gs.get("hands_played") or 0)
    sc[35] = float(gs.get("discards_used") or 0)
    sc[36] = 1.0
    sc[37] = float(gs.get("stake") or 1)
    deck_key = gs.get("deck_key") or "b_red"
    sc[38] = float(CENTER_IDS[deck_key] + 1) if deck_key in CENTER_IDS else 14.0
    sc[41] = 1.0
    sc[42] = float(gs.get("idol_rank") or 11)
    voucher_key = gs.get("ante_voucher_key") or ""
    sc[43] = float(CENTER_IDS[voucher_key] + 1) if voucher_key in CENTER_IDS else 0.0
    sc[44] = float(gs.get("shop_joker_max") or 2)
    sc[47] = float(gs.get("blind_mult") or (1.0 if state_name == "SELECTING_HAND" else 0.0))
    set_group("scalars", sc)

    # 2. hand_levels (12, 5)
    hl = np.zeros((12, 5), dtype=np.float32)
    hl_in = gs.get("hand_levels") or {}
    for i, hname in enumerate(HANDS):
        info = hl_in.get(hname) or {}
        hl[i, 0] = float(info.get("level") or 1)
        hl[i, 1] = float(info.get("chips") or 0)
        hl[i, 2] = float(info.get("mult") or 0)
        hl[i, 3] = float(info.get("played") or 0)
        hl[i, 4] = float(info.get("l_chips") or 0)
    set_group("hand_levels", hl)

    # 3. Card groups
    for gname, cards_list in (
        ("hand", hand),
        ("jokers", jokers),
        ("consumables", consumables),
        ("shop", shop),
        ("vouchers", vouchers),
        ("boosters", boosters),
        ("pack", pack),
        ("deck", deck),
    ):
        _, rows, cols = GROUP_MAP[gname]
        mat = np.zeros((rows, cols), dtype=np.float32)
        for i, c in enumerate(cards_list[:rows]):
            mat[i] = _encode_card(c, state_ctx)
        set_group(gname, mat)

    # 4. tags (12, 2)
    tmat = np.zeros((12, 2), dtype=np.float32)
    for i, t in enumerate(tags[:12]):
        tkey = t.get("key") if isinstance(t, dict) else t
        if tkey in TAG_IDS:
            tmat[i, 0] = float(TAG_IDS[tkey] + 1)
            tmat[i, 1] = float(t.get("tally", 0) if isinstance(t, dict) else 0)
    set_group("tags", tmat)

    # 5. counts (10,)
    cnt = np.zeros(len(COUNT_NAMES), dtype=np.float32)
    cnt[0] = float(len(hand))
    cnt[1] = float(len(jokers))
    cnt[2] = float(len(consumables))
    cnt[3] = float(len(shop))
    cnt[4] = float(len(vouchers))
    cnt[5] = float(len(boosters))
    cnt[6] = float(len(pack))
    cnt[7] = float(len(deck))
    cnt[8] = float(len(tags))
    cnt[9] = float(gs.get("discard_count") or 0)
    set_group("counts", cnt)

    # 6. blind_tags (3,)
    bt = np.zeros(3, dtype=np.float32)
    stag = gs.get("small_tag") or ""
    btag = gs.get("big_tag") or ""
    bt[0] = float(TAG_IDS[stag] + 1) if stag in TAG_IDS else 0.0
    bt[1] = float(TAG_IDS[btag] + 1) if btag in TAG_IDS else 0.0
    bt[2] = sc[27]
    set_group("blind_tags", bt)

    # 7. round_ctx (78,)
    rctx = np.zeros(len(ROUND_CTX_FEATURES), dtype=np.float32)
    last_score = float(gs.get("last_hand_score") or 0.0)
    last_chips = float(gs.get("last_hand_chips") or 0.0)
    last_mult = float(gs.get("last_hand_mult") or 0.0)
    rctx[0] = math.log1p(max(0.0, last_score))
    rctx[1] = math.log1p(max(0.0, last_chips))
    rctx[2] = math.log1p(max(0.0, last_mult))
    rctx[3] = (last_score / sc[16]) if sc[16] > 0 else 0.0
    last_hand = gs.get("last_hand_name") or ""
    hands_this_round = gs.get("hands_played_round") or {}
    forbidden = set(gs.get("forbidden_hands") or [])
    for i, hname in enumerate(HANDS):
        if hname == last_hand:
            rctx[4 + i] = 1.0
        rctx[16 + i] = float(hands_this_round.get(hname) or 0)
        if hname in forbidden:
            rctx[28 + i] = 1.0
    rctx[40] = math.log1p(max(0.0, float(gs.get("chips_small") or 0.0)))
    rctx[41] = math.log1p(max(0.0, float(gs.get("chips_big") or 0.0)))
    rctx[42] = math.log1p(max(0.0, float(gs.get("chips_boss") or 0.0)))
    rctx[43] = float(gs.get("dollars_small") or 0.0)
    rctx[44] = float(gs.get("dollars_big") or 0.0)
    rctx[45] = float(gs.get("dollars_boss") or 0.0)
    owned_vouchers = set(gs.get("owned_vouchers") or [])
    for i, vkey in enumerate(VOUCHER_KEYS):
        if vkey in owned_vouchers:
            rctx[46 + i] = 1.0
    set_group("round_ctx", rctx)

    # 8. selection (44,) - zero at decision time (before picks)
    sel = np.zeros(len(SEL_FEATURES), dtype=np.float32)
    x_flat = np.concatenate([buf, sel])

    # 9. hand_ctx (32,)
    hctx = np.zeros(HS_CTX, dtype=np.int32)
    n_hand = min(len(hand), N_HAND)
    for i in range(n_hand):
        c = hand[i]
        is_stone = c.get("center_key") == "m_stone"
        rank = 0 if is_stone else int(c.get("rank") or 0)
        sm = _suit_mask(c, smeared=smeared)
        forced = 1 if c.get("forced") else 0
        hctx[i] = (rank & 15) | ((sm & 15) << 4) | ((1 if is_stone else 0) << 8) | (forced << 9) | (1 << 10)
    hctx[24] = n_hand
    hctx[25] = (1 if four_fingers else 0) | (2 if shortcut else 0)
    play_lo = 5 if blind_key == "bl_psychic" else 1
    play_hi = min(5, n_hand)
    hctx[28] = play_lo
    hctx[29] = play_hi

    # 10. action_mask (152,) and sel_specs (32, 3)
    mask = np.zeros(len(ACTIONS), dtype=bool)
    specs = np.zeros((len(SEL_VERBS), 3), dtype=np.int64)
    all_hand_bits = (1 << n_hand) - 1

    dollars = float(gs.get("dollars") or 0.0)
    bankrupt_at = float(gs.get("bankrupt_at") or 0.0)
    spendable = dollars - bankrupt_at
    joker_limit = int(gs.get("joker_limit") or 5)
    cons_limit = int(gs.get("consumable_limit") or 2)

    # Consumables & Jokers (available in most decision states except GAME_OVER)
    if state_name != "GAME_OVER":
        for i, jk in enumerate(jokers[:20]):
            if not jk.get("eternal"):
                mask[ACTION_IDX[f"SELL_JOKER_{i}"]] = True
        for i, cs in enumerate(consumables[:10]):
            mask[ACTION_IDX[f"SELL_CONS_{i}"]] = True
            if _can_use_consumable(cs, state_name, n_hand, len(jokers), joker_limit, len(consumables)):
                ckey = cs.get("center_key", "")
                if ckey in CONS_HAND_SELECT:
                    lo, hi = CONS_HAND_SELECT[ckey]
                    if n_hand >= lo:
                        mask[ACTION_IDX[f"USE_CONS_{i}"]] = True
                        specs[SEL_VERB_IDX[f"USE_CONS_{i}"]] = [lo, hi, all_hand_bits]
                else:
                    mask[ACTION_IDX[f"USE_CONS_{i}"]] = True
                    specs[SEL_VERB_IDX[f"USE_CONS_{i}"]] = [0, 0, 0]
        if len(jokers) >= 2:
            mask[ACTION_IDX["REORDER_JOKERS"]] = True
            specs[SEL_VERB_IDX["REORDER_JOKERS"]] = [len(jokers), len(jokers), (1 << len(jokers)) - 1]

    if state_name == "BLIND_SELECT":
        mask[ACTION_IDX["SELECT_BLIND"]] = True
        if gs.get("blind_on_deck") in ("Small", "Big"):
            mask[ACTION_IDX["SKIP_BLIND"]] = True
        if gs.get("can_reroll_boss"):
            mask[ACTION_IDX["REROLL_BOSS"]] = True

    elif state_name == "SELECTING_HAND":
        if int(gs.get("hands_left") or 0) > 0 and n_hand >= play_lo:
            mask[ACTION_IDX["PLAY"]] = True
            specs[SEL_VERB_IDX["PLAY"]] = [play_lo, play_hi, all_hand_bits]
            import torch
            from . import hand_select as HS
            hctx_t = torch.from_numpy(hctx)[None].repeat(len(PLAY_HAND_ACTIONS), 1)
            targets_t = torch.arange(len(PLAY_HAND_ACTIONS), dtype=torch.long)
            hm = HS.HandMasker(hctx_t, targets_t, boss=True, ctx_host=hctx_t.numpy())
            co, so = hm.card_stop(torch.zeros(len(PLAY_HAND_ACTIONS), N_HAND, dtype=torch.bool))
            reachable = (co.any(dim=1) | so).tolist()
            for pa, ok in zip(PLAY_HAND_ACTIONS, reachable):
                if ok:
                    mask[ACTION_IDX[pa]] = True
                    specs[SEL_VERB_IDX[pa]] = [play_lo, play_hi, all_hand_bits]
        if int(gs.get("discards_left") or 0) > 0 and n_hand >= 1:
            mask[ACTION_IDX["DISCARD"]] = True
            specs[SEL_VERB_IDX["DISCARD"]] = [1, min(5, n_hand), all_hand_bits]

    elif state_name == "ROUND_EVAL":
        mask[ACTION_IDX["CASH_OUT"]] = True

    elif state_name == "SHOP":
        mask[ACTION_IDX["END_SHOP"]] = True
        reroll_cost = float(gs.get("reroll_cost") or 5)
        if spendable >= reroll_cost:
            mask[ACTION_IDX["REROLL"]] = True
        for i, scard in enumerate(shop[:8]):
            cost = float(scard.get("cost") or 0)
            if spendable < cost:
                continue
            kind = scard.get("area_kind") or scard.get("ability_set") or ""
            is_joker = kind == "Joker" or (scard.get("center_key") or "").startswith("j_")
            is_cons = not is_joker and kind in ("Tarot", "Planet", "Spectral")
            is_playing = not is_joker and not is_cons
            has_slot = (len(jokers) < joker_limit) if is_joker else ((len(consumables) < cons_limit) if is_cons else True)
            if has_slot or scard.get("edition") == "negative":
                mask[ACTION_IDX[f"BUY_{i}"]] = True
            if is_cons and _can_use_consumable(scard, state_name, n_hand, len(jokers), joker_limit, len(consumables)):
                mask[ACTION_IDX[f"BUY_USE_{i}"]] = True
        for i, vcard in enumerate(vouchers[:4]):
            if spendable >= float(vcard.get("cost") or 0):
                mask[ACTION_IDX[f"REDEEM_{i}"]] = True
        for i, bcard in enumerate(boosters[:4]):
            if spendable >= float(bcard.get("cost") or 0):
                mask[ACTION_IDX[f"OPEN_{i}"]] = True

    elif state_name in ("TAROT_PACK", "PLANET_PACK", "SPECTRAL_PACK", "STANDARD_PACK", "BUFFOON_PACK"):
        mask[ACTION_IDX["SKIP_PACK"]] = True
        for i, pcard in enumerate(pack[:8]):
            ckey = pcard.get("center_key", "")
            kind = pcard.get("ability_set") or ""
            if state_name == "BUFFOON_PACK" or kind == "Joker" or ckey.startswith("j_"):
                if len(jokers) < joker_limit or pcard.get("edition") == "negative":
                    mask[ACTION_IDX[f"PICK_{i}"]] = True
                    specs[SEL_VERB_IDX[f"PICK_{i}"]] = [0, 0, 0]
            elif state_name == "STANDARD_PACK" or ckey == "c_base" or ckey.startswith("m_"):
                mask[ACTION_IDX[f"PICK_{i}"]] = True
                specs[SEL_VERB_IDX[f"PICK_{i}"]] = [0, 0, 0]
            else:
                if _can_use_consumable(pcard, state_name, n_hand, len(jokers), joker_limit, len(consumables)):
                    mask[ACTION_IDX[f"PICK_{i}"]] = True
                    if ckey in CONS_HAND_SELECT:
                        lo, hi = CONS_HAND_SELECT[ckey]
                        specs[SEL_VERB_IDX[f"PICK_{i}"]] = [lo, hi, all_hand_bits]
                    else:
                        specs[SEL_VERB_IDX[f"PICK_{i}"]] = [0, 0, 0]

    return x_flat, mask, specs, hctx
