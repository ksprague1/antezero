"""Adapter wrapping antezero's `antezero_sim.Game` (or `antezero_bridge` Env)
into a `BalatroEnv`-compatible interface for evaluating RLBalatro models.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .obs_builder import ACTION_IDX, ACTIONS, OBS_SIZE, SEL_VERB_IDX, build_observation
from .constants import (
    DECKS,
    HANDS,
    HS_CTX,
    SEL_FEATURES,
    SEL_VERBS,
    layout as _layout,
    split_obs as _split,
)

try:
    import gymnasium as gym
except ImportError:
    gym = None

SUIT_LETTER = {"S": "Spades", "H": "Hearts", "C": "Clubs", "D": "Diamonds"}
SUIT_SORT_ORDER = {"Spades": 1, "Hearts": 2, "Clubs": 3, "Diamonds": 4}
SECRET_HANDS = {"Flush Five", "Flush House", "Five of a Kind"}
RANK_LETTER = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
BASE_RANK_CHIPS = {2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 10, 12: 10, 13: 10, 14: 11}

BLIND_MULT_OVERRIDE = {
    "bl_needle": 1.0,
    "bl_wall": 4.0,
    "bl_violet_vessel": 6.0,
}
FINISHER_BOSSES = {
    "bl_amber_acorn",
    "bl_verdant_leaf",
    "bl_violet_vessel",
    "bl_crimson_heart",
    "bl_cerulean_bell",
}

BLIND_AMOUNTS_BASE = (300.0, 800.0, 2000.0, 5000.0, 11000.0, 20000.0, 35000.0, 50000.0)
BLIND_AMOUNTS_FASTER = (300.0, 900.0, 2600.0, 8000.0, 20000.0, 36000.0, 60000.0, 100000.0)
BLIND_AMOUNTS_FASTEST = (300.0, 1000.0, 3200.0, 9000.0, 25000.0, 60000.0, 110000.0, 200000.0)

# Consumables by key prefix/set for obs_builder ability_set inference
_PLANETS = {
    "c_mercury", "c_venus", "c_earth", "c_mars", "c_jupiter", "c_saturn",
    "c_uranus", "c_neptune", "c_pluto", "c_planet_x", "c_ceres", "c_eris",
}
_SPECTRALS = {
    "c_familiar", "c_grim", "c_incantation", "c_talisman", "c_aura", "c_wraith",
    "c_sigil", "c_ouija", "c_ectoplasm", "c_immolate", "c_ankh", "c_deja_vu",
    "c_hex", "c_trance", "c_medium", "c_cryptid", "c_soul", "c_black_hole",
}


# Static config.extra values for vouchers/jokers/consumables where antezero stores
# the constant in code rather than Card.ab.extra (used as fallback when extra_obs gives 0.0).
_STATIC_EXTRA0: dict[str, float] = {
    "v_overstock_plus": 1.0, "v_clearance_sale": 25.0,
    "v_liquidation": 50.0, "v_hone": 2.0, "v_glow_up": 4.0, "v_reroll_surplus": 2.0,
    "v_reroll_glut": 2.0, "v_crystal_ball": 3.0, "v_omen_globe": 1.0,
    "v_telescope": 3.0, "v_observatory": 1.5, "v_grabber": 1.0, "v_nacho_tong": 1.0,
    "v_wasteful": 1.0, "v_recyclomancy": 1.0, "v_tarot_merchant": 2.0,
    "v_tarot_tycoon": 4.0, "v_planet_merchant": 2.4, "v_planet_tycoon": 4.0,
    "v_seed_money": 50.0, "v_money_tree": 20.0, "v_magic_trick": 1.0,
    "v_illusion": 1.0, "v_hieroglyph": 1.0, "v_petroglyph": 1.0,
    "v_directors_cut": 10.0, "v_retcon": 10.0, "v_paint_brush": 1.0,
    "v_palette": 1.0, "v_antimatter": 1.0, "v_blank": 5.0,
    "c_hex": 2.0, "c_temperance": 50.0, "c_wheel_of_fortune": 4.0,
    "j_8_ball": 4.0, "j_acrobat": 3.0, "j_rough_gem": 1.0, "j_bloodstone": 2.0,
    "j_arrowhead": 50.0, "j_onyx_agate": 7.0, "j_space": 4.0, "j_gros_michel": 15.0,
    "j_cavendish": 3.0, "j_business": 2.0, "j_reserved_parking": 2.0,
    "j_hallucination": 2.0, "j_egg": 3.0, "j_golden": 4.0, "j_cloud_9": 1.0,
    "j_faceless": 5.0, "j_todo_list": 4.0, "j_mail": 5.0, "j_scholar": 20.0,
    "j_walkie_talkie": 10.0, "j_even_steven": 4.0, "j_odd_todd": 31.0,
    "j_fibonacci": 8.0, "j_smiley": 5.0, "j_scary_face": 30.0,
    "j_photograph": 2.0, "j_ancient": 1.5, "j_triboulet": 2.0, "j_baron": 1.5,
    "j_hanging_chad": 2.0, "j_blackboard": 3.0, "j_card_sharp": 3.0,
    "j_seeing_double": 2.0, "j_flower_pot": 3.0, "j_drivers_license": 3.0,
    "j_idol": 2.0, "j_baseball": 1.5, "j_stuntman": 250.0, "j_blue_joker": 2.0,
    "j_bull": 2.0, "j_bootstraps": 2.0, "j_abstract": 3.0, "j_misprint": 23.0,
    "j_shoot_the_moon": 13.0, "j_steel_joker": 0.2,
    "j_stone": 25.0, "j_erosion": 4.0, "j_banner": 30.0,
    "j_half": 20.0, "j_delayed_grat": 2.0, "j_troubadour": 2.0, "j_merry_andy": 3.0,
    "j_burglar": 3.0, "j_trading": 3.0, "j_matador": 8.0, "j_vagabond": 4.0,
    "j_credit_card": 20.0, "j_gift": 1.0, "j_satellite": 1.0, "j_to_the_moon": 1.0,
    "j_oops": 2.0, "j_riff_raff": 2.0, "j_flash": 2.0, "j_fortune_teller": 1.0,
    "j_greedy_joker": 3.0, "j_lusty_joker": 3.0, "j_wrathful_joker": 3.0,
    "j_gluttenous_joker": 3.0, "j_green_joker": 1.0, "j_invisible": 2.0,
    "j_ramen": 0.01, "j_ride_the_bus": 1.0, "j_sock_and_buskin": 1.0, "j_trousers": 2.0,
}

_STATIC_EXTRA1: dict[str, float] = {
    "j_faceless": 3.0,
    "j_green_joker": 1.0,
    "j_half": 3.0,
    "j_mystic_summit": 15.0,
    "j_scholar": 4.0,
    "j_todo_list": 4.0,
    "j_walkie_talkie": 4.0,
}

# Explicit overrides (extra0, extra1) where BalatroEnv differs from extra_obs
_EXTRA_OVERRIDE: dict[str, tuple[float | None, float | None]] = {
    "j_raised_fist": (0.0, None),
    "j_mystic_summit": (0.0, 15.0),
    "v_overstock_norm": (0.0, None),
}


def _infer_ability_set(key: str) -> str:
    if key.startswith("j_"):
        return "Joker"
    if key.startswith("v_"):
        return "Voucher"
    if key.startswith("p_"):
        return "Booster"
    if key == "c_base" or key.startswith("m_"):
        return "Default" if key == "c_base" else "Enhanced"
    if key in _PLANETS:
        return "Planet"
    if key in _SPECTRALS:
        return "Spectral"
    if key.startswith("c_"):
        return "Tarot"
    return ""


def _parse_card_code(card_code: str) -> tuple[int, str]:
    """Convert antezero 'H_A' / 'C_T' / 'S_7' into (rank_int, suit_name)."""
    if not card_code or "_" not in card_code:
        return 0, ""
    s_ch, r_ch = card_code.split("_", 1)
    suit = SUIT_LETTER.get(s_ch, "")
    if r_ch in RANK_LETTER:
        rank = RANK_LETTER[r_ch]
    elif r_ch.isdigit():
        rank = int(r_ch)
    else:
        rank = 0
    return rank, suit


def _blind_base_amount(ante: int, stake: int) -> float:
    if stake >= 6:
        table = BLIND_AMOUNTS_FASTEST
    elif stake >= 3:
        table = BLIND_AMOUNTS_FASTER
    else:
        table = BLIND_AMOUNTS_BASE
    if ante < 1:
        return 100.0
    if ante <= len(table):
        return table[ante - 1]
    past = float(ante - len(table))
    k = 0.75
    a = table[-1]
    b = 1.6
    c = past
    d = 1.0 + 0.2 * past
    return round(a * ((b + (k * c) ** d) ** c))


def _canon_bool(val: str | None) -> bool:
    return val == "true"


def _canon_float(val: str | None, default: float = 0.0) -> float:
    if val is None or val in ("nil", "", "nan"):
        return default
    return float(val)


def _canon_int(val: str | None, default: int = 0) -> int:
    if val is None or val in ("nil", "", "nan"):
        return default
    return int(float(val))


def _parse_area_cards(
    canon: dict[str, str],
    prefix: str,
    *,
    last_tarot_planet: str = "nil",
    hide_face_down: bool = True,
    times_played_list: list[int] | None = None,
    default_cost: float | None = None,
    extra_list: list[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    length = _canon_int(canon.get(f"{prefix}.len"), 0)
    out: list[dict[str, Any]] = []
    for i in range(length):
        p = f"{prefix}.{i}"
        key = canon.get(f"{p}.key", "c_base")
        if key == "nil":
            key = "c_base"
        facing = canon.get(f"{p}.facing", "front")
        face_down = facing == "back"

        debuff = _canon_bool(canon.get(f"{p}.debuff"))
        forced = _canon_bool(canon.get(f"{p}.forced"))

        if hide_face_down and face_down:
            out.append({
                "center_key": "",
                "ability_set": "",
                "rank": 0,
                "suit": "",
                "edition": "",
                "seal": "",
                "debuff": debuff,
                "face_down": True,
                "selected": False,
                "cost": 0.0,
                "sell_cost": 0.0,
                "eternal": False,
                "perishable": 0,
                "rental": False,
                "mult": 0.0,
                "x_mult": 1.0,
                "chips": 0.0,
                "extra0": 0.0,
                "extra1": 0.0,
                "times_played": 0,
                "joker_value": 0.0,
                "joker_count": 0.0,
                "forced": forced,
                "can_use": False,
            })
            continue

        card_code = canon.get(f"{p}.card", "")
        rank, suit = _parse_card_code(card_code)
        ed = canon.get(f"{p}.edition", "nil")
        edition = "" if ed in ("nil", "") else ed
        sl = canon.get(f"{p}.seal", "nil")
        seal = "" if sl in ("nil", "") else sl

        ab_set = _infer_ability_set(key)
        is_joker = ab_set == "Joker"
        is_playing = bool(card_code) or key == "c_base" or key.startswith("m_")

        perma_bonus = _canon_float(canon.get(f"{p}.perma_bonus"), 0.0)
        chips = 0.0
        if is_playing:
            if key == "m_stone":
                chips = 50.0 + perma_bonus
            else:
                chips = float(BASE_RANK_CHIPS.get(rank, 0)) + (30.0 if key == "m_bonus" else 0.0) + perma_bonus
        elif is_joker:
            chips = _canon_float(canon.get(f"{p}.ab.extra.chips"), 0.0)

        mult = _canon_float(canon.get(f"{p}.ab.mult"), 0.0) if is_joker else (4.0 if key == "m_mult" else 0.0)
        x_mult = _canon_float(canon.get(f"{p}.ab.x_mult"), 1.0) if is_joker else (2.0 if key == "m_glass" else 1.0)
        extra_value = _canon_float(canon.get(f"{p}.ab.extra_value"), 0.0)

        extra0 = 0.0
        extra1 = 0.0
        joker_count = 0.0
        if extra_list is not None and i < len(extra_list):
            extra0 = float(extra_list[i][0])
            extra1 = float(extra_list[i][1])
        if is_joker:
            if extra0 == 0.0:
                if f"{p}.ab.extra" in canon:
                    extra0 = _canon_float(canon.get(f"{p}.ab.extra"), 0.0)
                elif f"{p}.ab.extra.chips" in canon:
                    extra0 = _canon_float(canon.get(f"{p}.ab.extra.chips"), 0.0)
                elif f"{p}.ab.extra.h_size" in canon:
                    extra0 = _canon_float(canon.get(f"{p}.ab.extra.h_size"), 0.0)
            if extra1 == 0.0 and f"{p}.ab.extra.dollars" in canon:
                extra1 = _canon_float(canon.get(f"{p}.ab.extra.dollars"), 0.0)

            for tally_key in ("caino_xmult", "steel_tally", "stone_tally", "driver_tally", "nine_tally"):
                if f"{p}.ab.{tally_key}" in canon:
                    joker_count = _canon_float(canon.get(f"{p}.ab.{tally_key}"), 0.0)
                    break
        if extra0 == 0.0 and key in _STATIC_EXTRA0:
            extra0 = _STATIC_EXTRA0[key]
        if extra1 == 0.0 and key in _STATIC_EXTRA1:
            extra1 = _STATIC_EXTRA1[key]
        if key == "j_ice_cream":
            extra1 = extra0 if extra0 != 0.0 else chips
            extra0 = 5.0
            chips = 0.0
        if key in _EXTRA_OVERRIDE:
            ov0, ov1 = _EXTRA_OVERRIDE[key]
            if ov0 is not None:
                extra0 = ov0
            if ov1 is not None:
                extra1 = ov1

        can_use = True
        if key == "c_fool":
            can_use = last_tarot_planet not in ("nil", "", "c_fool")

        cost = _canon_float(canon.get(f"{p}.cost"), default_cost if default_cost is not None else 0.0)
        sell_cost = _canon_float(canon.get(f"{p}.sell_cost"), default_cost if default_cost is not None else 0.0)
        tp = int(times_played_list[i]) if times_played_list is not None and i < len(times_played_list) else 0
        perish = (
            _canon_int(canon.get(f"{p}.ab.perish_tally"), 0) + 1
            if f"{p}.ab.perish_tally" in canon
            else 0
        )

        out.append({
            "center_key": key,
            "ability_set": ab_set,
            "rank": rank,
            "suit": suit,
            "edition": edition,
            "seal": seal,
            "debuff": debuff,
            "face_down": face_down,
            "selected": False,
            "cost": cost,
            "sell_cost": sell_cost,
            "eternal": _canon_bool(canon.get(f"{p}.ab.eternal")),
            "perishable": perish,
            "rental": _canon_bool(canon.get(f"{p}.ab.rental")),
            "mult": mult,
            "x_mult": x_mult,
            "chips": chips,
            "extra0": extra0,
            "extra1": extra1,
            "times_played": tp,
            "joker_value": extra_value,
            "joker_count": joker_count,
            "forced": forced,
            "can_use": can_use,
        })
    return out


def _populate_joker_values(
    areas: tuple[list[dict[str, Any]], ...],
    *,
    jokers: list[dict[str, Any]],
    deck: list[dict[str, Any]],
    discards_left: int,
    dollars: float,
    joker_limit: int,
    starting_deck_size: int,
    playing_cards_count: int,
) -> None:
    """Populate `joker_value` (col 22) for dynamic/scaling jokers to match BalatroEnv."""
    n_jokers = len(jokers)
    n_stencil = sum(1 for j in jokers if j.get("center_key") == "j_stencil")
    total_sell = sum(float(j.get("sell_cost") or 0.0) for j in jokers)
    for area in areas:
        is_owned = area is jokers
        for j in area:
            if j.get("face_down"):
                continue
            k = j.get("center_key", "")
            if not k.startswith("j_"):
                continue
            if k == "j_banner":
                j["joker_value"] = 30.0 * discards_left
            elif k == "j_delayed_grat":
                j["joker_value"] = 2.0 * discards_left
            elif k == "j_mystic_summit":
                j["joker_value"] = 15.0 if discards_left == 0 else 0.0
            elif k == "j_abstract":
                j["joker_value"] = 3.0 * n_jokers
            elif k == "j_stencil":
                j["joker_value"] = float(max(1, joker_limit - n_jokers + n_stencil))
            elif k == "j_bull":
                j["joker_value"] = 2.0 * max(0.0, dollars)
            elif k == "j_bootstraps":
                j["joker_value"] = 2.0 * max(0, int(dollars // 5))
            elif k == "j_swashbuckler":
                j["joker_value"] = (total_sell - float(j.get("sell_cost") or 0.0)) if is_owned else total_sell
            elif k == "j_blue_joker":
                j["joker_value"] = 2.0 * len(deck)
            elif k == "j_erosion":
                j["joker_value"] = 4.0 * max(0, starting_deck_size - playing_cards_count)
            elif k == "j_ice_cream":
                j["joker_value"] = float(j.get("extra1") or 0.0)
            elif k in ("j_runner", "j_square", "j_castle", "j_wee"):
                j["joker_value"] = float(j.get("chips") or 0.0)
            elif k in (
                "j_green_joker", "j_popcorn", "j_ride_the_bus", "j_red_card",
                "j_flash", "j_fortune_teller", "j_trousers", "j_ceremonial",
            ):
                j["joker_value"] = float(j.get("mult") or 0.0)
            elif k in (
                "j_ramen", "j_constellation", "j_vampire", "j_hologram", "j_obelisk",
                "j_lucky_cat", "j_glass", "j_madness", "j_campfire", "j_hit_the_road",
                "j_throwback", "j_yorick",
            ):
                j["joker_value"] = float(j.get("x_mult") or 1.0)
            elif k == "j_caino":
                j["joker_value"] = float(j.get("joker_count") or 1.0)
            elif k == "j_steel_joker":
                j["joker_value"] = 1.0 + 0.2 * float(j.get("joker_count") or 0.0)
            elif k == "j_stone":
                j["joker_value"] = 25.0 * float(j.get("joker_count") or 0.0)
            elif k == "j_cloud_9":
                j["joker_value"] = float(j.get("joker_count") or 0.0)
            elif k == "j_drivers_license":
                j["joker_value"] = 3.0 if float(j.get("joker_count") or 0.0) >= 16 else 1.0
            elif k == "j_rocket":
                j["joker_value"] = float(j.get("extra1") or 0.0)
            else:
                j["joker_value"] = 0.0


def canonical_to_gs(
    canon: dict[str, str],
    *,
    deck_key: str = "b_red",
    stake: int = 1,
    last_hand_score: float = 0.0,
    hide_face_down: bool = True,
    extra_obs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert an antezero `canonical()` dict into an `obs_builder` `gs` dict."""
    state_name = canon.get("state", "BLIND_SELECT")
    ante = _canon_int(canon.get("ante"), 1)
    dollars = _canon_float(canon.get("dollars"), 0.0)
    bankrupt_at = _canon_float(canon.get("bankrupt_at"), 0.0)

    blind_on_deck = canon.get("blind_on_deck", "Small")
    if blind_on_deck == "nil":
        blind_on_deck = "Small"
    boss_key = canon.get("blind_choices.Boss", "")
    if boss_key == "nil":
        boss_key = ""

    blind_name = canon.get("blind.name", "")
    if state_name == "SELECTING_HAND" or blind_name:
        if blind_on_deck == "Small":
            blind_key = "bl_small"
            blind_mult = 1.0
        elif blind_on_deck == "Big":
            blind_key = "bl_big"
            blind_mult = 1.5
        else:
            blind_key = boss_key
            blind_mult = BLIND_MULT_OVERRIDE.get(boss_key, 2.0)
    else:
        blind_key = ""
        blind_mult = 0.0

    last_tarot = canon.get("last_tarot_planet", "nil")
    hand_tp = extra_obs.get("hand_times_played") if extra_obs else None
    deck_tp = extra_obs.get("deck_times_played") if extra_obs else None
    hand = _parse_area_cards(
        canon, "hand", last_tarot_planet=last_tarot, hide_face_down=hide_face_down, times_played_list=hand_tp
    )
    jokers = _parse_area_cards(
        canon,
        "jokers",
        last_tarot_planet=last_tarot,
        hide_face_down=hide_face_down,
        extra_list=extra_obs.get("jokers_extra") if extra_obs else None,
    )
    consumables = _parse_area_cards(
        canon,
        "consumables",
        last_tarot_planet=last_tarot,
        hide_face_down=False,
        extra_list=extra_obs.get("consumables_extra") if extra_obs else None,
    )
    shop = _parse_area_cards(
        canon,
        "shop_jokers",
        last_tarot_planet=last_tarot,
        hide_face_down=False,
        extra_list=extra_obs.get("shop_extra") if extra_obs else None,
    )
    vouchers = _parse_area_cards(
        canon,
        "shop_vouchers",
        last_tarot_planet=last_tarot,
        hide_face_down=False,
        extra_list=extra_obs.get("vouchers_extra") if extra_obs else None,
    )
    boosters = _parse_area_cards(
        canon,
        "shop_booster",
        last_tarot_planet=last_tarot,
        hide_face_down=False,
        extra_list=extra_obs.get("boosters_extra") if extra_obs else None,
    )
    pack = _parse_area_cards(
        canon,
        "pack",
        last_tarot_planet=last_tarot,
        hide_face_down=False,
        extra_list=extra_obs.get("pack_extra") if extra_obs else None,
    )
    deck = _parse_area_cards(
        canon,
        "deck",
        last_tarot_planet=last_tarot,
        hide_face_down=False,
        times_played_list=deck_tp,
        default_cost=1.0,
    )
    # Sort deck so draw-pile order is never leaked to the agent (matching BalatroEnv suit order).
    deck.sort(
        key=lambda c: (
            c["rank"],
            SUIT_SORT_ORDER.get(c["suit"], 0),
            c["center_key"],
            c["edition"],
            c["seal"],
        )
    )

    discard_count = _canon_int(canon.get("discard.len"), 0)
    playing_cards_count = len(deck) + len(hand) + discard_count
    discards_left = _canon_int(canon.get("discards_left"), 0)
    joker_limit = _canon_int(canon.get("joker_limit"), 5)
    starting_deck_size = _canon_int(canon.get("starting_deck_size"), 52)
    _populate_joker_values(
        (jokers, shop, pack),
        jokers=jokers,
        deck=deck,
        discards_left=discards_left,
        dollars=dollars,
        joker_limit=joker_limit,
        starting_deck_size=starting_deck_size,
        playing_cards_count=playing_cards_count,
    )

    owned_vouchers = [k for k in canon.get("used_vouchers", "").split(",") if k]
    boss_rerolled = _canon_bool(canon.get("boss_rerolled"))
    can_reroll_boss = (
        ("v_retcon" in owned_vouchers or ("v_directors_cut" in owned_vouchers and not boss_rerolled))
        and (dollars - bankrupt_at >= 10.0)
    )

    hand_visible = extra_obs.get("hand_visible") if extra_obs else None
    hand_levels: dict[str, dict[str, float]] = {}
    hands_played_round: dict[str, int] = {}
    for h in HANDS:
        played_cnt = _canon_float(canon.get(f"hands.{h}.played"), 0.0)
        if hand_visible is not None:
            vis = bool(hand_visible.get(h, False))
        else:
            vis = (h not in SECRET_HANDS) or (played_cnt > 0.0)
        hand_levels[h] = {
            "level": _canon_float(canon.get(f"hands.{h}.level"), 1.0),
            "chips": _canon_float(canon.get(f"hands.{h}.chips"), 0.0),
            "mult": _canon_float(canon.get(f"hands.{h}.mult"), 0.0),
            "played": played_cnt,
            "l_chips": 1.0 if vis else 0.0,
        }
        hands_played_round[h] = _canon_int(canon.get(f"hands.{h}.played_this_round"), 0)

    forbidden_hands: list[str] = []
    if blind_key == "bl_eye":
        forbidden_hands = [h for h, cnt in hands_played_round.items() if cnt > 0]
    elif blind_key == "bl_mouth" and sum(hands_played_round.values()) > 0:
        forbidden_hands = [h for h, cnt in hands_played_round.items() if cnt == 0]

    tags: list[dict[str, Any]] = []
    t_idx = 0
    while f"tags.{t_idx}.key" in canon:
        tags.append({"key": canon[f"tags.{t_idx}.key"], "tally": 0})
        t_idx += 1

    base_chips = _blind_base_amount(ante, stake)
    ante_scaling = 2.0 if deck_key == "b_plasma" else 1.0
    boss_mult = BLIND_MULT_OVERRIDE.get(boss_key, 2.0)

    last_hand = canon.get("last_hand_played", "nil")
    small_tag = canon.get("blind_tags.Small", "nil")
    big_tag = canon.get("blind_tags.Big", "nil")
    voucher_key = canon.get("voucher", "nil")

    owned_joker_keys = {j["center_key"] for j in jokers if j["center_key"]}
    has_idol = "j_idol" in owned_joker_keys
    has_mail = "j_mail" in owned_joker_keys
    has_ancient = "j_ancient" in owned_joker_keys
    has_castle = "j_castle" in owned_joker_keys

    lhs_score = float(extra_obs["last_hand_score"]) if extra_obs else last_hand_score
    lhs_chips = float(extra_obs["last_hand_chips"]) if extra_obs else 0.0
    lhs_mult = float(extra_obs["last_hand_mult"]) if extra_obs else 0.0

    return {
        "state": state_name,
        "deck_key": deck_key,
        "stake": stake,
        "ante": ante,
        "round": _canon_int(canon.get("round"), 0),
        "dollars": dollars,
        "hands_left": _canon_int(canon.get("hands_left"), 0),
        "discards_left": _canon_int(canon.get("discards_left"), 0),
        "chips": _canon_float(canon.get("chips"), 0.0),
        "blind_chips": _canon_float(canon.get("blind.chips"), 0.0),
        "blind_key": blind_key,
        "blind_disabled": _canon_bool(canon.get("blind.disabled")),
        "hand_limit": _canon_int(canon.get("hand_limit"), 8),
        "joker_limit": _canon_int(canon.get("joker_limit"), 5),
        "consumable_limit": _canon_int(canon.get("consumable_limit"), 2),
        "reroll_cost": _canon_float(canon.get("reroll_cost"), 5.0),
        "interest_cap": _canon_float(canon.get("interest_cap"), 25.0),
        "bankrupt_at": bankrupt_at,
        "blind_on_deck": blind_on_deck,
        "boss_key": boss_key,
        "skips": _canon_int(canon.get("skips"), 0),
        "hands_played": _canon_int(canon.get("hands_played"), 0),
        "discards_used": _canon_int(canon.get("discards_used"), 0),
        "pack_choices": _canon_int(canon.get("pack_choices"), 0),
        "ante_voucher_key": "" if voucher_key == "nil" else voucher_key,
        "shop_joker_max": _canon_int(canon.get("shop_joker_max"), 2) if state_name == "SHOP" else 0,
        "blind_mult": blind_mult,
        "small_tag": "" if small_tag == "nil" else small_tag,
        "big_tag": "" if big_tag == "nil" else big_tag,
        "discard_count": discard_count,
        "playing_cards_count": len(deck) + len(hand) + discard_count,
        "idol_rank": _canon_int(canon.get("idol.rank"), 11) if has_idol else 11,
        "idol_suit": canon.get("idol.suit", "") if has_idol else "",
        "mail_rank": _canon_int(canon.get("mail"), 0) if has_mail else 0,
        "ancient_suit": canon.get("ancient", "") if has_ancient else "",
        "castle_suit": canon.get("castle", "") if has_castle else "",
        "owned_vouchers": owned_vouchers,
        "can_reroll_boss": can_reroll_boss,
        "last_hand_name": "" if last_hand == "nil" else last_hand,
        "last_hand_score": lhs_score,
        "last_hand_chips": lhs_chips,
        "last_hand_mult": lhs_mult,
        "hands_played_round": hands_played_round,
        "forbidden_hands": forbidden_hands,
        "hand_levels": hand_levels,
        "chips_small": base_chips * 1.0 * ante_scaling,
        "chips_big": base_chips * 1.5 * ante_scaling,
        "chips_boss": base_chips * boss_mult * ante_scaling,
        "dollars_small": 0.0 if stake >= 2 else 3.0,
        "dollars_big": 4.0,
        "dollars_boss": 8.0 if boss_key in FINISHER_BOSSES else 5.0,
        "tags": tags,
        "hand": hand,
        "jokers": jokers,
        "consumables": consumables,
        "shop": shop,
        "vouchers": vouchers,
        "boosters": boosters,
        "pack": pack,
        "deck": deck,
    }


def decompose_joker_permutation(picks: list[int], n_jokers: int) -> list[str]:
    """Decompose a target joker permutation `picks` into `move_joker from to` commands.

    If `picks` has fewer than `n_jokers` elements (e.g. policy selection head caps at 5),
    any omitted joker indices are appended in their relative order.
    """
    seen = set(picks)
    if len(seen) != len(picks) or any(p < 0 or p >= n_jokers for p in picks):
        raise ValueError(f"invalid joker permutation {picks!r} for {n_jokers} jokers")
    full_picks = list(picks) + [i for i in range(n_jokers) if i not in seen]
    cur = list(range(n_jokers))
    cmds: list[str] = []
    for target_idx, desired in enumerate(full_picks):
        from_idx = cur.index(desired)
        if from_idx != target_idx:
            cmds.append(f"move_joker {from_idx} {target_idx}")
            item = cur.pop(from_idx)
            cur.insert(target_idx, item)
    return cmds


def action_to_antezero_codes(
    act_name: str,
    picks: list[int],
    n_jokers: int,
    forced_bits: int = 0,
) -> list[str]:
    """Translate an RLBalatro `(action_name, picks)` into 0 or more antezero action codes."""
    if act_name == "PLAY" or act_name.startswith("PLAY_") or act_name == "DISCARD":
        forced_idx = [i for i in range(24) if (forced_bits >> i) & 1]
        merged = forced_idx + [int(q) for q in picks if int(q) not in forced_idx]
        merged = merged[:5]
        mask_bits = 0
        for idx in merged:
            mask_bits |= 1 << idx
        op = "discard" if act_name == "DISCARD" else "play"
        return [f"{op} {mask_bits}"]
    if act_name == "SELECT_BLIND":
        return ["select"]
    if act_name == "SKIP_BLIND":
        return ["skip"]
    if act_name == "REROLL_BOSS":
        return ["reroll_boss"]
    if act_name == "CASH_OUT":
        return ["cash_out"]
    if act_name == "END_SHOP":
        return ["end_shop"]
    if act_name == "REROLL":
        return ["reroll"]
    if act_name == "SKIP_PACK":
        return ["skip_pack"]
    if act_name.startswith("BUY_USE_"):
        idx = int(act_name.rsplit("_", 1)[1])
        return [f"buy_use {idx} 0"]
    if act_name.startswith("BUY_"):
        idx = int(act_name.rsplit("_", 1)[1])
        return [f"buy {idx}"]
    if act_name.startswith("REDEEM_"):
        idx = int(act_name.rsplit("_", 1)[1])
        return [f"redeem {idx}"]
    if act_name.startswith("OPEN_"):
        idx = int(act_name.rsplit("_", 1)[1])
        return [f"open {idx}"]
    if act_name.startswith("SELL_JOKER_"):
        idx = int(act_name.rsplit("_", 1)[1])
        return [f"sell_joker {idx}"]
    if act_name.startswith("SELL_CONS_"):
        idx = int(act_name.rsplit("_", 1)[1])
        return [f"sell_consumable {idx}"]
    if act_name.startswith("USE_CONS_"):
        idx = int(act_name.rsplit("_", 1)[1])
        mask_bits = 0
        for q in picks:
            mask_bits |= 1 << int(q)
        return [f"use {idx} {mask_bits}"]
    if act_name.startswith("PICK_"):
        idx = int(act_name.rsplit("_", 1)[1])
        mask_bits = 0
        for q in picks:
            mask_bits |= 1 << int(q)
        return [f"pick {idx} {mask_bits}"]
    if act_name == "REORDER_JOKERS":
        return decompose_joker_permutation([int(q) for q in picks], n_jokers)
    if act_name.startswith("JOKER_RIGHT_"):
        idx = int(act_name.rsplit("_", 1)[1])
        return [f"move_joker {idx} {idx + 1}"]
    if act_name.startswith("JOKER_FRONT_"):
        idx = int(act_name.rsplit("_", 1)[1])
        return [] if idx == 0 else [f"move_joker {idx} 0"]
    if act_name.startswith("JOKER_BACK_"):
        idx = int(act_name.rsplit("_", 1)[1])
        return [] if idx >= n_jokers - 1 else [f"move_joker {idx} {n_jokers - 1}"]
    raise ValueError(f"unsupported RLBalatro action {act_name!r}")


def _intersect_mask(obs_mask: np.ndarray, legal_codes: set[str], state_name: str, n_jokers: int) -> np.ndarray:
    """Intersect `obs_builder`'s 152-dim mask with antezero's `legal_codes`."""
    if state_name == "GAME_OVER" or not legal_codes:
        return np.zeros_like(obs_mask, dtype=bool)

    has_play = any(c.startswith("play ") for c in legal_codes)
    has_discard = any(c.startswith("discard ") for c in legal_codes)

    out = np.zeros_like(obs_mask, dtype=bool)
    for idx, name in enumerate(ACTIONS):
        if not obs_mask[idx]:
            continue
        if name == "PLAY" or name.startswith("PLAY_"):
            out[idx] = has_play
        elif name == "DISCARD":
            out[idx] = has_discard
        elif name == "SELECT_BLIND":
            out[idx] = "select" in legal_codes
        elif name == "SKIP_BLIND":
            out[idx] = "skip" in legal_codes
        elif name == "REROLL_BOSS":
            out[idx] = "reroll_boss" in legal_codes
        elif name == "CASH_OUT":
            out[idx] = "cash_out" in legal_codes
        elif name == "END_SHOP":
            out[idx] = "end_shop" in legal_codes
        elif name == "REROLL":
            out[idx] = "reroll" in legal_codes
        elif name == "SKIP_PACK":
            out[idx] = "skip_pack" in legal_codes
        elif name.startswith("BUY_USE_"):
            i = int(name.rsplit("_", 1)[1])
            out[idx] = f"buy_use {i} 0" in legal_codes
        elif name.startswith("BUY_"):
            i = int(name.rsplit("_", 1)[1])
            out[idx] = f"buy {i}" in legal_codes
        elif name.startswith("REDEEM_"):
            i = int(name.rsplit("_", 1)[1])
            out[idx] = f"redeem {i}" in legal_codes
        elif name.startswith("OPEN_"):
            i = int(name.rsplit("_", 1)[1])
            out[idx] = f"open {i}" in legal_codes
        elif name.startswith("SELL_JOKER_"):
            i = int(name.rsplit("_", 1)[1])
            out[idx] = f"sell_joker {i}" in legal_codes
        elif name.startswith("SELL_CONS_"):
            i = int(name.rsplit("_", 1)[1])
            out[idx] = f"sell_consumable {i}" in legal_codes
        elif name.startswith("USE_CONS_"):
            i = int(name.rsplit("_", 1)[1])
            prefix = f"use {i} "
            out[idx] = any(c.startswith(prefix) for c in legal_codes)
        elif name.startswith("PICK_"):
            i = int(name.rsplit("_", 1)[1])
            prefix = f"pick {i} "
            out[idx] = any(c.startswith(prefix) for c in legal_codes)
        elif name == "REORDER_JOKERS":
            out[idx] = state_name != "GAME_OVER" and n_jokers >= 2
        elif name.startswith("JOKER_RIGHT_"):
            i = int(name.rsplit("_", 1)[1])
            out[idx] = state_name != "GAME_OVER" and 0 <= i < n_jokers - 1
        elif name.startswith("JOKER_FRONT_"):
            i = int(name.rsplit("_", 1)[1])
            out[idx] = state_name != "GAME_OVER" and 1 <= i < n_jokers
        elif name.startswith("JOKER_BACK_"):
            i = int(name.rsplit("_", 1)[1])
            out[idx] = state_name != "GAME_OVER" and 0 <= i < n_jokers - 1

    # Fallback: if intersection cleared everything even though antezero has legal actions,
    # enable any action whose code is directly in legal_codes.
    if not out.any() and legal_codes:
        for idx, name in enumerate(ACTIONS):
            if name == "PLAY" and has_play:
                out[idx] = True
            elif name == "DISCARD" and has_discard:
                out[idx] = True
            elif name == "SELECT_BLIND" and "select" in legal_codes:
                out[idx] = True
            elif name == "SKIP_BLIND" and "skip" in legal_codes:
                out[idx] = True
            elif name == "REROLL_BOSS" and "reroll_boss" in legal_codes:
                out[idx] = True
            elif name == "CASH_OUT" and "cash_out" in legal_codes:
                out[idx] = True
            elif name == "END_SHOP" and "end_shop" in legal_codes:
                out[idx] = True
            elif name == "REROLL" and "reroll" in legal_codes:
                out[idx] = True
            elif name == "SKIP_PACK" and "skip_pack" in legal_codes:
                out[idx] = True
            elif name.startswith("PICK_") and any(c.startswith(f"pick {int(name.rsplit('_', 1)[1])} ") for c in legal_codes):
                out[idx] = True
            elif name.startswith("BUY_") and not name.startswith("BUY_USE_") and f"buy {int(name.rsplit('_', 1)[1])}" in legal_codes:
                out[idx] = True
    return out


class AntezeroBalatroEnv(gym.Env if gym else object):
    """`BalatroEnv`-compatible wrapper around `antezero_sim.Game` or `antezero_bridge` `Env`."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        deck: str = "b_red",
        stake: int = 1,
        max_steps: int = 5000,
        endless: bool = False,
        obs_hide_face_down: bool = True,
        backend: Any | None = None,
        max_reorders_per_decision: int = 1,
        use_extra_obs: bool = True,
    ):
        if deck not in DECKS:
            raise ValueError(f"unknown deck {deck!r}")
        self.deck = deck
        self.stake = int(stake)
        self.max_steps = int(max_steps)
        self.endless = bool(endless)
        self.obs_hide_face_down = bool(obs_hide_face_down)
        self.max_reorders_per_decision = int(max_reorders_per_decision)
        self.use_extra_obs = bool(use_extra_obs)
        self._backend = backend
        self._game: Any = None

        self.actions = ACTIONS
        self.sel_verbs = SEL_VERBS
        self.groups, self.obs_size = _layout()
        self.n_actions = len(ACTIONS)

        if gym:
            self.action_space = gym.spaces.Discrete(self.n_actions)
            spaces = {
                name: gym.spaces.Box(-np.inf, np.inf, (c,) if r == 1 else (r, c), np.float32)
                for name, _, r, c in self.groups
            }
            spaces["selection"] = gym.spaces.Box(-np.inf, np.inf, (len(SEL_FEATURES),), np.float32)
            self.observation_space = gym.spaces.Dict(spaces)

        self._seed_counter = 0
        self._buf = np.zeros(self.obs_size, dtype=np.float32)
        self._selbuf = np.zeros(len(SEL_FEATURES), dtype=np.float32)
        self._mask = np.zeros(self.n_actions, dtype=bool)
        self._specs = np.zeros((len(SEL_VERBS), 3), dtype=np.int32)
        self._hctx = np.zeros(HS_CTX, dtype=np.int32)
        self._gs: dict[str, Any] = {}
        self._canon: dict[str, str] = {}
        self._legal_codes: set[str] = set()
        self._steps = 0
        self._last_hand_score = 0.0
        self._last_blind_chips = 0.0
        self._reorders_here = 0

    @property
    def obs_buffer(self) -> np.ndarray:
        return self._buf

    def set_deck(self, deck: str, stake: int) -> None:
        if deck not in DECKS:
            raise ValueError(f"unknown deck {deck!r}")
        self.deck = deck
        self.stake = int(stake)

    def action_masks(self) -> np.ndarray:
        return self._mask

    def sel_specs(self) -> np.ndarray:
        return self._specs

    def hand_ctx(self) -> np.ndarray:
        return self._hctx

    def forced_cards(self) -> np.ndarray:
        out = np.zeros(24, dtype=bool)
        for i, c in enumerate(self._gs.get("hand", [])[:24]):
            if c.get("forced"):
                out[i] = True
        return out

    def _forced_bits(self) -> int:
        bits = 0
        for i, c in enumerate(self._gs.get("hand", [])[:24]):
            if c.get("forced"):
                bits |= 1 << i
        return bits

    def _backend_legal_codes(self) -> set[str]:
        if self._backend is not None:
            if hasattr(self._backend, "legal_codes"):
                return set(self._backend.legal_codes())
            return {a.code() for a in self._backend.legal_actions()}
        return set(self._game.legal_codes())

    def _backend_canonical(self) -> dict[str, str]:
        if self._backend is not None:
            return dict(self._backend.canonical())
        return dict(self._game.canonical())

    def _backend_is_terminal(self) -> bool:
        if self._backend is not None:
            return bool(self._backend.is_terminal())
        return bool(self._game.is_terminal())

    def _backend_step(self, code: str) -> None:
        if self._backend is not None:
            self._backend.step(code)
        else:
            self._game.step(code)

    def _backend_extra_obs(self) -> dict[str, Any] | None:
        if not self.use_extra_obs:
            return None
        if self._backend is not None:
            if hasattr(self._backend, "extra_obs"):
                return dict(self._backend.extra_obs())
            return None
        if hasattr(self._game, "extra_obs"):
            return dict(self._game.extra_obs())
        return None

    def _refresh(self) -> dict[str, np.ndarray]:
        self._canon = self._backend_canonical()
        self._legal_codes = self._backend_legal_codes()
        extra_obs = self._backend_extra_obs()
        self._gs = canonical_to_gs(
            self._canon,
            deck_key=self.deck,
            stake=self.stake,
            last_hand_score=self._last_hand_score,
            hide_face_down=self.obs_hide_face_down,
            extra_obs=extra_obs,
        )
        x_flat, raw_mask, specs, hctx = build_observation(self._gs)
        self._buf = x_flat[: self.obs_size].copy()
        self._selbuf = x_flat[self.obs_size :].copy()

        # Patch scalar / round_ctx / selection fields where obs_builder differs from BalatroEnv:
        # - scalars[23]: reroll_cost (0 when free_rerolls > 0; obs_builder turns 0 into 5)
        # - scalars[34]: hands_played_round (obs_builder reads gs["hands_played"])
        # - scalars[44]: run hands_played (obs_builder puts shop_joker_max here)
        # - round_ctx[3]: last_blind_share (preserve across ROUND_EVAL / SHOP / BLIND_SELECT)
        # - selection[7]: chips_needed = max(0, blind_chips - chips) in SELECTING_HAND
        free_rerolls = _canon_float(self._canon.get("free_rerolls"), 0.0)
        self._buf[23] = 0.0 if free_rerolls > 0.0 else float(self._gs["reroll_cost"])
        self._buf[34] = float(_canon_int(self._canon.get("hands_played_round"), 0))
        self._buf[44] = float(self._gs["hands_played"])

        if self._gs["state"] == "SELECTING_HAND" and self._gs["blind_chips"] > 0:
            self._last_blind_chips = float(self._gs["blind_chips"])
        rctx_off = next(off for name, off, _, _ in self.groups if name == "round_ctx")
        denom = float(self._gs["blind_chips"]) if self._gs["blind_chips"] > 0 else self._last_blind_chips
        self._buf[rctx_off + 3] = (float(self._gs["last_hand_score"]) / denom) if denom > 0 else 0.0

        if self._gs["state"] == "SELECTING_HAND":
            self._selbuf[7] = max(0.0, float(self._gs["blind_chips"]) - float(self._gs["chips"]))
        else:
            self._selbuf[7] = 0.0
        self._specs = specs.astype(np.int32, copy=True)
        self._hctx = hctx.astype(np.int32, copy=True)

        # Tighten USE_CONS_* and PICK_* selection specs from antezero's legal_codes()
        # (e.g. Cerulean Bell forced card or edition-ineligible cards for Aura).
        for verb_prefix, code_prefix in (("USE_CONS_", "use "), ("PICK_", "pick ")):
            for slot in range(10 if verb_prefix == "USE_CONS_" else 8):
                v_name = f"{verb_prefix}{slot}"
                v_idx = SEL_VERB_IDX[v_name]
                c_pref = f"{code_prefix}{slot} "
                masks = [int(c.split()[2]) for c in self._legal_codes if c.startswith(c_pref)]
                if masks:
                    lo = min(m.bit_count() for m in masks)
                    hi = max(m.bit_count() for m in masks)
                    elig = 0
                    for m in masks:
                        elig |= m
                    self._specs[v_idx] = [lo, hi, elig]

        mask = _intersect_mask(
            raw_mask,
            self._legal_codes,
            self._gs["state"],
            len(self._gs.get("jokers", [])),
        )
        if self._reorders_here >= self.max_reorders_per_decision:
            mask[ACTION_IDX["REORDER_JOKERS"]] = False
            for idx, name in enumerate(ACTIONS):
                if name.startswith(("JOKER_RIGHT_", "JOKER_FRONT_", "JOKER_BACK_")):
                    mask[idx] = False
        self._mask = mask

        obs = _split(self._buf.copy(), self.groups)
        obs["selection"] = self._selbuf.copy()
        return obs

    def reset(self, seed: int | str | None = None, options: dict[str, Any] | None = None):
        if options and "seed_str" in options:
            seed_str = str(options["seed_str"])
        elif isinstance(seed, str):
            seed_str = seed
        elif seed is not None:
            seed_str = f"S{int(seed)}"
        else:
            self._seed_counter += 1
            seed_str = f"S{self._seed_counter}"

        if self._backend is not None:
            self._backend.reset(seed_str, deck=self.deck, stake=self.stake)
        else:
            import antezero_sim

            self._game = antezero_sim.Game(seed_str, self.deck, self.stake, endless=self.endless)

        self._steps = 0
        self._last_hand_score = 0.0
        self._last_blind_chips = 0.0
        self._reorders_here = 0
        obs = self._refresh()
        return obs, {"action_mask": self.action_masks()}

    def step(self, action):
        act_idx, picks = action if isinstance(action, tuple) else (int(action), [])
        act_name = ACTIONS[int(act_idx)]
        prev_chips = _canon_float(self._canon.get("chips"), 0.0)
        prev_state = self._canon.get("state", "")
        n_jokers = len(self._gs.get("jokers", []))

        is_reorder = act_name == "REORDER_JOKERS" or act_name.startswith(
            ("JOKER_RIGHT_", "JOKER_FRONT_", "JOKER_BACK_")
        )
        forced_bits = self._forced_bits()
        codes = action_to_antezero_codes(
            act_name,
            [int(q) for q in picks],
            n_jokers,
            forced_bits=forced_bits,
        )
        if (
            (act_name.startswith("USE_CONS_") or act_name.startswith("PICK_"))
            and len(codes) == 1
            and codes[0] not in self._legal_codes
            and forced_bits
        ):
            hi = int(self._specs[SEL_VERB_IDX[act_name], 1])
            forced_idx = [i for i in range(24) if (forced_bits >> i) & 1]
            merged = forced_idx + [int(q) for q in picks if int(q) not in forced_idx]
            if hi > 0:
                merged = merged[:hi]
            op, slot_s, _ = codes[0].split()
            cand = f"{op} {slot_s} {sum(1 << i for i in merged)}"
            if cand in self._legal_codes:
                codes = [cand]

        illegal = False
        try:
            for code in codes:
                self._backend_step(code)
        except Exception:
            illegal = True

        self._steps += 1
        if is_reorder:
            self._reorders_here += 1
        else:
            self._reorders_here = 0

        new_canon = self._backend_canonical()
        new_chips = _canon_float(new_canon.get("chips"), 0.0)
        hand_score = 0.0
        if prev_state == "SELECTING_HAND" and (act_name == "PLAY" or act_name.startswith("PLAY_")):
            hand_score = max(0.0, new_chips - prev_chips)
            self._last_hand_score = hand_score

        # Auto-step single-legal-action ROUND_EVAL (cash_out) and BLIND_SELECT (Boss blind
        # when reroll_boss is not available), matching BalatroEnv's decision points.
        for _ in range(8):
            if self._backend_is_terminal():
                break
            cur_canon = self._backend_canonical()
            cur_state = cur_canon.get("state", "")
            cur_legal = self._backend_legal_codes()
            if cur_state == "ROUND_EVAL" and "cash_out" in cur_legal:
                self._backend_step("cash_out")
            elif (
                cur_state == "BLIND_SELECT"
                and cur_canon.get("blind_on_deck") == "Boss"
                and "reroll_boss" not in cur_legal
                and "select" in cur_legal
            ):
                self._backend_step("select")
            else:
                break

        obs = self._refresh()
        term = self._backend_is_terminal() or self._gs.get("state") == "GAME_OVER"
        trunc = (not term) and (self._steps >= self.max_steps)
        won = _canon_bool(self._canon.get("won"))
        ante = _canon_int(self._canon.get("ante"), 1)

        info = {
            "action_mask": self.action_masks(),
            "won": won,
            "ante": ante,
            "steps": self._steps,
            "hand_score": hand_score,
            "illegal": illegal,
            "dead_end": not term and not self._legal_codes,
        }
        return obs, 0.0, term, trunc, info

    def close(self) -> None:
        if self._backend is not None and hasattr(self._backend, "close"):
            self._backend.close()
        self._game = None
