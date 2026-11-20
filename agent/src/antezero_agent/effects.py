"""What each joker, consumable, voucher and booster pack does, as fixed features the networks read next to an item's id.

An id alone has to be learned from games in which the item was owned; items the policy never keeps stay unknown.
These columns say what the card text says (what it gives, when, and what it needs), so items with similar effects
look similar. ``matrix()`` is indexed by center id; rows of other centers are zero. A checkpoint stores the table
it was trained with, so the rows can be corrected without changing what older checkpoints read.

Sizes are the printed ones, scaled to order 1: chips / 100, mult / 10, xmult - 1, dollars / 5; growth per event
chips / 10, mult / 2, xmult x 4, dollars / 2, value / 2; hand size, hands and discards / 2; a payout's cap / 50. A joker paid per unit of
something (Bull per dollar, Blue Joker per deck card, Steel Joker per Steel card) carries a typical total and the
flag of the unit.
"""

import functools

import antezero_sim as az
import torch

HANDS = ["flush_five", "flush_house", "five", "straight_flush", "four", "full_house", "flush", "straight", "three", "two_pair", "pair", "high_card"]
SUITS = ["spades", "hearts", "clubs", "diamonds"]
RANKS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
ENHANCEMENTS = ["bonus", "mult", "wild", "glass", "steel", "stone", "gold", "lucky", "any"]
SEALS = ["gold", "red", "blue", "purple"]
RARITIES = ["common", "uncommon", "rare", "legendary"]
COLS = [
    # what it gives, per trigger
    "chips", "mult", "xmult", "dollars", "cap",
    # what it gains per growth event; whether it shrinks with use, resets, or can destroy itself (risk: its chance per round)
    "g_chips", "g_mult", "g_xmult", "g_dollars", "g_value", "decays", "resets", "perishes", "risk",
    # the chance of its effect (1 when certain), cards retriggered, copying a joker's ability
    "chance", "retrigger", "copies",
    # what it creates (how many), and whether that is a copy of something held or chosen
    "mk_tarot", "mk_planet", "mk_spectral", "mk_joker", "mk_card", "mk_consumable", "mk_tag", "copy",
    "hand_size", "hands", "discards", "joker_slots", "cons_slots", "shop_slots",
    # rule changes
    "rule_hand", "rule_economy", "discount", "rate", "levels_hand", "luck", "survival", "anti_boss", "ante_down",
    "alters_cards", "strips", "destroys_cards", "destroys_jokers", "loses_money", "sell_to_use", "edition",
    # when it acts
    "on_scored", "on_held", "on_hand", "on_discard", "on_round_end", "on_blind", "on_shop", "on_sell", "on_consumable", "on_boss", "no_boss", "passive",
    # what it names
    *(f"h_{h}" for h in HANDS), *(f"s_{s}" for s in SUITS), *(f"r_{r}" for r in RANKS), *(f"e_{e}" for e in ENHANCEMENTS),
    *(f"seal_{s}" for s in SEALS),
    # what it needs
    "few_cards", "four_cards", "one_card", "first_card", "rich", "poor", "discards_left", "no_discards", "discard_penalty",
    "first_hand", "first_discard", "last_hand", "round_only", "periodic", "big_deck", "thin_deck", "other_jokers", "empty_slots", "needs_room",
    "needs_many", "pos_right", "pos_leftmost", "mixed_suits", "same_hand", "varied_hand", "no_face", "needs_all", "variable",
    "skips", "pack_skips", "likes_planets", "likes_tarots", "likes_cards", "likes_destroyed", "likes_uncommon",
    # playing cards it takes at once (selected, discarded or changed), cards a pack shows
    "n_cards", "choices",
    # a joker's rarity, or that of the joker it creates
    *RARITIES,
    "is_joker", "is_planet", "is_tarot", "is_spectral", "is_voucher", "is_pack",
]  # fmt: skip
E = len(COLS)
SCALE = {
    "chips": 100, "mult": 10, "dollars": 5, "g_chips": 10, "g_mult": 2, "g_xmult": 0.25, "g_dollars": 2, "g_value": 2,
    "hand_size": 2, "hands": 2, "discards": 2, "rate": 4, "n_cards": 3, "choices": 5, "cap": 50,
}  # fmt: skip
FACES = (11, 12, 13)


def row(*flags: str, hand=(), suit=(), rank=(), enh=(), seal=(), **sizes: float) -> dict[str, float]:
    """One item's features: ``flags`` are columns set to 1, ``sizes`` columns with a printed size (``xmult`` is the
    multiplier itself), ``hand`` / ``suit`` / ``rank`` / ``enh`` / ``seal`` the hands, suits, ranks, enhancements and
    seals it names."""
    as_list = lambda v: [v] if isinstance(v, (str, int)) else list(v)  # noqa: E731
    r = dict.fromkeys(flags, 1.0)
    r.update({f"h_{h}": 1.0 for h in as_list(hand)})
    r.update({f"s_{s}": 1.0 for s in as_list(suit)})
    r.update({f"r_{k}": 1.0 for k in as_list(rank)})
    r.update({f"e_{e}": 1.0 for e in as_list(enh)})
    r.update({f"seal_{s}": 1.0 for s in as_list(seal)})
    for k, v in sizes.items():
        r[k] = (v - 1.0 if k == "xmult" else v) / SCALE.get(k, 1.0)
    r.setdefault("chance", 1.0)
    return r


JOKERS = {
    "j_joker": row("on_hand", mult=4),
    "j_greedy_joker": row("on_scored", suit="diamonds", mult=3),
    "j_lusty_joker": row("on_scored", suit="hearts", mult=3),
    "j_wrathful_joker": row("on_scored", suit="spades", mult=3),
    "j_gluttenous_joker": row("on_scored", suit="clubs", mult=3),
    "j_jolly": row("on_hand", hand="pair", mult=8),
    "j_zany": row("on_hand", hand="three", mult=12),
    "j_mad": row("on_hand", hand="two_pair", mult=10),
    "j_crazy": row("on_hand", hand="straight", mult=12),
    "j_droll": row("on_hand", hand="flush", mult=10),
    "j_sly": row("on_hand", hand="pair", chips=50),
    "j_wily": row("on_hand", hand="three", chips=100),
    "j_clever": row("on_hand", hand="two_pair", chips=80),
    "j_devious": row("on_hand", hand="straight", chips=100),
    "j_crafty": row("on_hand", hand="flush", chips=80),
    "j_half": row("on_hand", "few_cards", mult=20),
    "j_stencil": row("on_hand", "empty_slots", xmult=2),  # x1 per empty joker slot, itself included
    "j_four_fingers": row("rule_hand", "passive", hand=("flush", "straight")),
    "j_mime": row("on_held", "on_round_end", retrigger=1),
    "j_credit_card": row("rule_economy", "passive", "poor"),
    "j_ceremonial": row("on_blind", "on_hand", "destroys_jokers", "other_jokers", "pos_right", g_mult=6),  # twice the destroyed joker's sell value
    "j_banner": row("on_hand", "discards_left", chips=60),  # 30 per discard left
    "j_mystic_summit": row("on_hand", "no_discards", mult=15),
    "j_marble": row("on_blind", enh="stone", mk_card=1),
    "j_loyalty_card": row("on_hand", "periodic", xmult=4),
    "j_8_ball": row("on_scored", "needs_room", rank=8, mk_tarot=1, chance=0.25),
    "j_misprint": row("on_hand", mult=11.5),  # 0 to 23
    "j_dusk": row("on_scored", "last_hand", retrigger=1),
    "j_raised_fist": row("on_held", mult=10),  # twice the lowest held rank
    "j_chaos": row("rule_economy", "on_shop"),
    "j_fibonacci": row("on_scored", rank=(14, 2, 3, 5, 8), mult=8),
    "j_steel_joker": row("on_hand", enh="steel", xmult=2),  # x0.2 per Steel card in the deck
    "j_scary_face": row("on_scored", rank=FACES, chips=30),
    "j_abstract": row("on_hand", "other_jokers", mult=12),  # 3 per joker
    "j_delayed_grat": row("on_round_end", "discard_penalty", dollars=6),  # $2 per discard when none was used
    "j_hack": row("on_scored", rank=(2, 3, 4, 5), retrigger=1),
    "j_pareidolia": row("rule_hand", "passive", rank=FACES),
    "j_gros_michel": row("on_hand", "perishes", mult=15, risk=1 / 6),
    "j_even_steven": row("on_scored", rank=(2, 4, 6, 8, 10), mult=4),
    "j_odd_todd": row("on_scored", rank=(14, 3, 5, 7, 9), chips=31),
    "j_scholar": row("on_scored", rank=14, chips=20, mult=4),
    "j_business": row("on_scored", rank=FACES, dollars=2, chance=0.5),
    "j_supernova": row("on_hand", "same_hand", g_mult=1),
    "j_ride_the_bus": row("on_hand", "no_face", "resets", rank=FACES, g_mult=1),
    "j_space": row("on_hand", "levels_hand", chance=0.25),
    "j_egg": row("on_round_end", g_value=3),
    "j_burglar": row("on_blind", hands=3, discards=-4),  # all discards
    "j_blackboard": row("on_hand", "on_held", "needs_all", suit=("spades", "clubs"), xmult=3),
    "j_runner": row("on_hand", hand="straight", g_chips=15),
    "j_ice_cream": row("on_hand", "decays", "perishes", chips=100),
    "j_dna": row("on_hand", "first_hand", "one_card", "copy", mk_card=1),
    "j_splash": row("rule_hand", "passive"),
    "j_blue_joker": row("on_hand", "big_deck", chips=80),  # 2 per card left in the deck
    "j_sixth_sense": row("on_hand", "first_hand", "one_card", "destroys_cards", "needs_room", rank=6, mk_spectral=1),
    "j_constellation": row("on_consumable", "on_hand", "likes_planets", g_xmult=0.1),
    "j_hiker": row("on_scored", "alters_cards", g_chips=5),
    "j_faceless": row("on_discard", rank=FACES, dollars=5, n_cards=3),
    "j_green_joker": row("on_hand", "discard_penalty", g_mult=1),
    "j_superposition": row("on_hand", "needs_all", "needs_room", hand="straight", rank=14, mk_tarot=1),
    "j_todo_list": row("on_hand", "variable", dollars=4),
    "j_cavendish": row("on_hand", "perishes", xmult=3, risk=0.001),
    "j_card_sharp": row("on_hand", "same_hand", "round_only", xmult=3),
    "j_red_card": row("on_shop", "on_hand", "pack_skips", g_mult=3),
    "j_madness": row("on_blind", "on_hand", "no_boss", "destroys_jokers", g_xmult=0.5),
    "j_square": row("on_hand", "four_cards", g_chips=4),
    "j_seance": row("on_hand", "needs_room", hand="straight_flush", mk_spectral=1),
    "j_riff_raff": row("on_blind", "needs_room", "common", mk_joker=2),
    "j_vampire": row("on_scored", "on_hand", "strips", enh="any", g_xmult=0.1),
    "j_shortcut": row("rule_hand", "passive", hand="straight"),
    "j_hologram": row("on_hand", "likes_cards", g_xmult=0.25),
    "j_vagabond": row("on_hand", "poor", "needs_room", mk_tarot=1),
    "j_baron": row("on_held", rank=13, xmult=1.5),
    "j_cloud_9": row("on_round_end", rank=9, dollars=4),  # $1 per 9 in the deck
    "j_rocket": row("on_round_end", "on_boss", dollars=1, g_dollars=2),
    "j_obelisk": row("on_hand", "varied_hand", "resets", g_xmult=0.2),
    "j_midas_mask": row("on_scored", "alters_cards", rank=FACES, enh="gold"),
    "j_luchador": row("anti_boss", "sell_to_use", "on_boss"),
    "j_photograph": row("on_scored", "first_card", rank=FACES, xmult=2),
    "j_gift": row("on_round_end", "other_jokers", g_value=1),
    "j_turtle_bean": row("passive", "decays", "perishes", hand_size=5),
    "j_erosion": row("on_hand", "thin_deck", mult=20),  # 4 per card below the starting deck size
    "j_reserved_parking": row("on_held", rank=FACES, dollars=1, chance=0.5),
    "j_mail": row("on_discard", "variable", dollars=5),
    "j_to_the_moon": row("rule_economy", "on_round_end", "rich", dollars=5),  # $1 more interest per $5
    "j_hallucination": row("on_shop", "needs_room", mk_tarot=1, chance=0.5),
    "j_fortune_teller": row("on_consumable", "on_hand", "likes_tarots", g_mult=1),
    "j_juggler": row("passive", hand_size=1),
    "j_drunkard": row("passive", discards=1),
    "j_stone": row("on_hand", enh="stone", chips=125),  # 25 per Stone card in the deck
    "j_golden": row("on_round_end", dollars=4),
    "j_lucky_cat": row("on_scored", "on_hand", enh="lucky", g_xmult=0.25),
    "j_baseball": row("on_hand", "other_jokers", "likes_uncommon", xmult=1.5),
    "j_bull": row("on_hand", "rich", chips=50),  # 2 per dollar
    "j_diet_cola": row("sell_to_use", mk_tag=1),
    "j_trading": row("on_discard", "first_discard", "one_card", "destroys_cards", dollars=3),
    "j_flash": row("on_shop", "on_hand", g_mult=2),
    "j_popcorn": row("on_hand", "decays", "perishes", mult=20),
    "j_trousers": row("on_hand", hand="two_pair", g_mult=2),
    "j_ancient": row("on_scored", "variable", xmult=1.5),
    "j_ramen": row("on_hand", "decays", "perishes", "discard_penalty", xmult=2),
    "j_walkie_talkie": row("on_scored", rank=(10, 4), chips=10, mult=4),
    "j_selzer": row("on_scored", "decays", "perishes", retrigger=1),
    "j_castle": row("on_discard", "on_hand", "variable", g_chips=3),
    "j_smiley": row("on_scored", rank=FACES, mult=5),
    "j_campfire": row("on_sell", "on_hand", "on_boss", "resets", g_xmult=0.25),
    "j_ticket": row("on_scored", enh="gold", dollars=4),
    "j_mr_bones": row("survival", "perishes", "passive"),
    "j_acrobat": row("on_hand", "last_hand", xmult=3),
    "j_sock_and_buskin": row("on_scored", rank=FACES, retrigger=1),
    "j_swashbuckler": row("on_hand", "other_jokers", mult=10),  # the other jokers' sell values
    "j_troubadour": row("passive", hand_size=2, hands=-1),
    "j_certificate": row("on_blind", seal=SEALS, mk_card=1),
    "j_smeared": row("rule_hand", "passive", hand="flush", suit=SUITS),
    "j_throwback": row("on_hand", "skips", g_xmult=0.25),
    "j_hanging_chad": row("on_scored", "first_card", retrigger=2),
    "j_rough_gem": row("on_scored", suit="diamonds", dollars=1),
    "j_bloodstone": row("on_scored", suit="hearts", xmult=1.5, chance=0.5),
    "j_arrowhead": row("on_scored", suit="spades", chips=50),
    "j_onyx_agate": row("on_scored", suit="clubs", mult=7),
    "j_glass": row("on_hand", "likes_destroyed", enh="glass", g_xmult=0.75),
    "j_ring_master": row("rule_economy", "passive"),
    "j_flower_pot": row("on_hand", "needs_all", suit=SUITS, xmult=3),
    "j_blueprint": row("copies", "other_jokers", "pos_right", "passive"),
    "j_wee": row("on_scored", "on_hand", rank=2, g_chips=8),
    "j_merry_andy": row("passive", discards=3, hand_size=-1),
    "j_oops": row("luck", "passive"),
    "j_idol": row("on_scored", "variable", xmult=2),
    "j_seeing_double": row("on_hand", "mixed_suits", suit="clubs", xmult=2),
    "j_matador": row("on_hand", "on_boss", dollars=8),
    "j_hit_the_road": row("on_discard", "on_hand", "resets", "round_only", rank=11, g_xmult=0.5),
    "j_duo": row("on_hand", hand="pair", xmult=2),
    "j_trio": row("on_hand", hand="three", xmult=3),
    "j_family": row("on_hand", hand="four", xmult=4),
    "j_order": row("on_hand", hand="straight", xmult=3),
    "j_tribe": row("on_hand", hand="flush", xmult=2),
    "j_stuntman": row("on_hand", chips=250, hand_size=-2),
    "j_invisible": row("on_round_end", "sell_to_use", "other_jokers", "copy", mk_joker=1),
    "j_brainstorm": row("copies", "other_jokers", "pos_leftmost", "passive"),
    "j_satellite": row("on_round_end", "likes_planets", dollars=4),  # $1 per distinct planet used
    "j_shoot_the_moon": row("on_held", rank=12, mult=13),
    "j_drivers_license": row("on_hand", "needs_many", enh="any", xmult=3),  # 16 enhanced cards in the deck
    "j_cartomancer": row("on_blind", "needs_room", mk_tarot=1),
    "j_astronomer": row("rule_economy", "on_shop", "likes_planets"),
    "j_burnt": row("on_discard", "first_discard", "levels_hand"),
    "j_bootstraps": row("on_hand", "rich", mult=10),  # 2 per $5
    "j_caino": row("on_hand", "likes_destroyed", rank=FACES, g_xmult=1),
    "j_triboulet": row("on_scored", rank=(13, 12), xmult=2),
    "j_yorick": row("on_discard", "on_hand", "periodic", g_xmult=1),
    "j_chicot": row("anti_boss", "on_boss", "passive"),
    "j_perkeo": row("on_shop", "copy", "edition", mk_consumable=1),
}  # fmt: skip
# The hand a planet levels, and the chips and mult a level adds.
PLANETS = {
    "c_mercury": ("pair", 15, 1), "c_venus": ("three", 20, 2), "c_earth": ("full_house", 25, 2), "c_mars": ("four", 30, 3),
    "c_jupiter": ("flush", 15, 2), "c_saturn": ("straight", 30, 3), "c_uranus": ("two_pair", 20, 1),
    "c_neptune": ("straight_flush", 40, 4), "c_pluto": ("high_card", 10, 1), "c_planet_x": ("five", 35, 3),
    "c_ceres": ("flush_house", 40, 4), "c_eris": ("flush_five", 50, 3),
}  # fmt: skip
TAROTS = {
    "c_fool": row("copy", "needs_room", mk_consumable=1),  # the last tarot or planet used
    "c_magician": row("alters_cards", enh="lucky", n_cards=2),
    "c_high_priestess": row("needs_room", mk_planet=2),
    "c_empress": row("alters_cards", enh="mult", n_cards=2),
    "c_emperor": row("needs_room", mk_tarot=2),
    "c_heirophant": row("alters_cards", enh="bonus", n_cards=2),
    "c_lovers": row("alters_cards", enh="wild", n_cards=1),
    "c_chariot": row("alters_cards", enh="steel", n_cards=1),
    "c_justice": row("alters_cards", enh="glass", n_cards=1),
    "c_hermit": row("rich", dollars=20, cap=20),
    "c_wheel_of_fortune": row("other_jokers", "edition", chance=0.25),
    "c_strength": row("alters_cards", n_cards=2),
    "c_hanged_man": row("destroys_cards", "thin_deck", n_cards=2),
    "c_death": row("alters_cards", "copy", n_cards=2),
    "c_temperance": row("other_jokers", dollars=20, cap=50),
    "c_devil": row("alters_cards", enh="gold", n_cards=1),
    "c_tower": row("alters_cards", enh="stone", n_cards=1),
    "c_star": row("alters_cards", suit="diamonds", n_cards=3),
    "c_moon": row("alters_cards", suit="clubs", n_cards=3),
    "c_sun": row("alters_cards", suit="hearts", n_cards=3),
    "c_judgement": row("needs_room", mk_joker=1),
    "c_world": row("alters_cards", suit="spades", n_cards=3),
}  # fmt: skip
SPECTRALS = {
    "c_familiar": row("destroys_cards", rank=FACES, enh="any", mk_card=3, n_cards=1),
    "c_grim": row("destroys_cards", rank=14, enh="any", mk_card=2, n_cards=1),
    "c_incantation": row("destroys_cards", rank=range(2, 11), enh="any", mk_card=4, n_cards=1),
    "c_talisman": row("alters_cards", seal="gold", n_cards=1),
    "c_aura": row("alters_cards", "edition", n_cards=1),
    "c_wraith": row("needs_room", "loses_money", "rare", mk_joker=1),
    "c_sigil": row("alters_cards", suit=SUITS, n_cards=8),  # the whole hand becomes one suit
    "c_ouija": row("alters_cards", hand_size=-1, n_cards=8),  # the whole hand becomes one rank
    "c_ectoplasm": row("other_jokers", "edition", joker_slots=1, hand_size=-1),
    "c_immolate": row("destroys_cards", "thin_deck", dollars=20, n_cards=5),
    "c_ankh": row("other_jokers", "copy", "destroys_jokers", mk_joker=1),
    "c_deja_vu": row("alters_cards", seal="red", n_cards=1),
    "c_hex": row("other_jokers", "edition", "destroys_jokers"),
    "c_trance": row("alters_cards", seal="blue", n_cards=1),
    "c_medium": row("alters_cards", seal="purple", n_cards=1),
    "c_cryptid": row("copy", mk_card=2, n_cards=1),
    "c_soul": row("needs_room", "legendary", mk_joker=1),
    "c_black_hole": row("levels_hand", hand=HANDS),
}  # fmt: skip
VOUCHERS = {
    "v_overstock_norm": row("on_shop", shop_slots=1),
    "v_overstock_plus": row("on_shop", shop_slots=1),
    "v_clearance_sale": row("rule_economy", "on_shop", discount=0.25),
    "v_liquidation": row("rule_economy", "on_shop", discount=0.5),
    "v_hone": row("on_shop", "edition", rate=2),
    "v_glow_up": row("on_shop", "edition", rate=4),
    "v_reroll_surplus": row("rule_economy", "on_shop", dollars=2),  # per reroll
    "v_reroll_glut": row("rule_economy", "on_shop", dollars=2),
    "v_crystal_ball": row("passive", cons_slots=1),
    "v_omen_globe": row("on_shop", mk_spectral=1, chance=0.2),
    "v_telescope": row("on_shop", "likes_planets", "same_hand"),
    "v_observatory": row("on_hand", "likes_planets", xmult=1.5),  # per held planet of the played hand
    "v_grabber": row("passive", hands=1),
    "v_nacho_tong": row("passive", hands=1),
    "v_wasteful": row("passive", discards=1),
    "v_recyclomancy": row("passive", discards=1),
    "v_tarot_merchant": row("on_shop", "likes_tarots", rate=2.4),
    "v_tarot_tycoon": row("on_shop", "likes_tarots", rate=8),
    "v_planet_merchant": row("on_shop", "likes_planets", rate=2.4),
    "v_planet_tycoon": row("on_shop", "likes_planets", rate=8),
    "v_seed_money": row("rule_economy", "on_round_end", "rich", dollars=5),  # the interest cap, raised by $5 a round
    "v_money_tree": row("rule_economy", "on_round_end", "rich", dollars=10),
    "v_blank": row("passive"),
    "v_antimatter": row("passive", joker_slots=1),
    "v_magic_trick": row("on_shop", mk_card=1, rate=4),
    "v_illusion": row("on_shop", "edition", enh="any", mk_card=1, rate=4),
    "v_hieroglyph": row("ante_down", hands=-1),
    "v_petroglyph": row("ante_down", discards=-1),
    "v_directors_cut": row("anti_boss", "on_boss", "rule_economy", "periodic"),  # once per ante
    "v_retcon": row("anti_boss", "on_boss", "rule_economy"),
    "v_paint_brush": row("passive", hand_size=1),
    "v_palette": row("passive", hand_size=1),
}  # fmt: skip
# Per pack kind: what it holds, and (cards shown, cards taken) of the normal, jumbo and mega pack.
PACKS = {
    "arcana": ("mk_tarot", {"normal": (3, 1), "jumbo": (5, 1), "mega": (5, 2)}),
    "celestial": ("mk_planet", {"normal": (3, 1), "jumbo": (5, 1), "mega": (5, 2)}),
    "standard": ("mk_card", {"normal": (3, 1), "jumbo": (5, 1), "mega": (5, 2)}),
    "buffoon": ("mk_joker", {"normal": (2, 1), "jumbo": (4, 1), "mega": (4, 2)}),
    "spectral": ("mk_spectral", {"normal": (2, 1), "jumbo": (4, 1), "mega": (4, 2)}),
}


def pack(key: str) -> dict[str, float]:
    _, kind, size, _ = key.split("_")
    holds, sizes = PACKS[kind]
    shown, taken = sizes[size]
    return row("on_shop", choices=shown, **{holds: taken})


def table(on: bool) -> torch.Tensor:
    """The table a new network reads: ``matrix()``, or zeros of its shape."""
    return matrix().clone() if on else torch.zeros_like(matrix())


@functools.cache
def tryable() -> torch.Tensor:
    """[N_CENTERS] whether a center is a joker, tarot or planet."""
    t = torch.zeros(az.N_CENTERS, dtype=torch.bool)
    t[[i for k, i, _ in az.centers() if k in JOKERS or k in TAROTS or k in PLANETS]] = True
    return t


@functools.cache
def matrix() -> torch.Tensor:
    """[N_CENTERS, E] features by center id."""
    rows = {k: {**r, "is_joker": 1.0} for k, r in JOKERS.items()}
    rows |= {k: {**r, "is_tarot": 1.0} for k, r in TAROTS.items()}
    rows |= {k: {**row("levels_hand", hand=h, chips=c, mult=m), "is_planet": 1.0} for k, (h, c, m) in PLANETS.items()}
    rows |= {k: {**r, "is_spectral": 1.0} for k, r in SPECTRALS.items()}
    rows |= {k: {**r, "is_voucher": 1.0} for k, r in VOUCHERS.items()}
    centers = az.centers()
    rows |= {k: {**pack(k), "is_pack": 1.0} for k, _, _ in centers if k.startswith("p_")}
    assert rows.keys() == {k for k, _, _ in centers}, sorted(rows.keys() ^ {k for k, _, _ in centers})
    m = torch.zeros(az.N_CENTERS, E)
    for k, i, rarity in centers:
        for c, v in rows[k].items():
            m[i, COLS.index(c)] = v
        if rarity:
            m[i, COLS.index(RARITIES[rarity - 1])] = 1.0
    return m
