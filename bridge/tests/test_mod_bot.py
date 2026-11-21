"""The mod's bot runs (mod/src/bot.lua, core.lua, act.lua, endpoints.lua) on
the player's profile, in the game's own LuaJIT with the installed game's and
Steamodded's functions they work with: the run select's Play buttons, the
save thread (writing to an in-memory disk), Steamodded's per-stake usage
tables and deck wins. Around them stands a small model of the game (starting
a run, its save step). Skipped where the game, its Lovely dump or Steamodded
is not installed."""

from __future__ import annotations

import ctypes
import faulthandler
import os
import zipfile
from pathlib import Path

import pytest
from antezero_bridge.launch import GAME_DIR, MODS_DIR

ROOT = Path(__file__).resolve().parents[2]
LUA51 = Path(os.environ.get("BALATRO_LUA51", GAME_DIR / "lua51.dll"))
PATCHED = MODS_DIR / "lovely" / "dump"  # the game's files Lovely patched, as they run
SMODS_SRC = MODS_DIR / "smods" / "src"


def _game_file(rel: str) -> str:
    if (PATCHED / rel).exists():
        return (PATCHED / rel).read_text(encoding="utf-8")
    with zipfile.ZipFile(GAME_DIR / "Balatro.exe") as z:
        return z.read(rel).decode("utf-8")


def _definition(text: str, header: str) -> str:
    """The top-level definition starting at the line ``header`` begins."""
    lines = text.splitlines()
    i = next(i for i, line in enumerate(lines) if line.startswith(header))
    j = next(j for j in range(i + 1, len(lines)) if lines[j] == "end")
    return "\n".join(lines[i : j + 1])


def _sources() -> str:
    misc = _game_file("functions/misc_functions.lua")
    events = _game_file("functions/common_events.lua")
    buttons = _game_file("functions/button_callbacks.lua")
    game = _game_file("game.lua")
    utils = (SMODS_SRC / "utils.lua").read_text(encoding="utf-8")
    overrides = (SMODS_SRC / "overrides.lua").read_text(encoding="utf-8")
    select = (SMODS_SRC / "utils" / "run_select.lua").read_text(encoding="utf-8")
    return "\n".join(
        [
            _game_file("engine/string_packer.lua"),
            *(_definition(misc, h) for h in ("function EMPTY(", "function copy_table(", "function remove_save(")),
            _definition(events, "function unlock_notify("),
            *(_definition(buttons, h) for h in ("G.FUNCS.start_setup_run =", "G.FUNCS.buy_from_shop =")),
            *(_definition(game, h) for h in ("function Game:save_settings(", "function Game:save_progress(")),
            "save_thread = assert(loadstring(" + _long_string(_game_file("engine/save_manager.lua")) + "))",
            *(_definition(utils, h) for h in ("function SMODS.stake_from_index(", "function convert_usage_entry(", "function convert_save_data(")),
            _definition(overrides, "function set_deck_win("),
            *(
                _definition(select, h)
                for h in (
                    "G.FUNCS.run_select_start_run =",
                    "G.FUNCS.run_select_quick_start =",
                    "function SMODS.RunSelect.Functions.start_run(",
                    "local start_run = Game.start_run",  # Steamodded's Game:start_run
                )
            ),
        ]
    )


def _long_string(s: str) -> str:
    return "[==[" + s + "]==]"


WORLD = r"""
local function noop() end
clock, disk, shown = 0, {}, {}
local queue = {}
require = noop -- the save thread's
love = {
  timer = {getTime = function() return clock end},
  update = noop,
  filesystem = {
    getInfo = function(f) if disk[f] then return {} end end,
    read = function(f) return disk[f] end,
    write = function(f, s) disk[f] = s return true end,
    append = function(f, s) disk[f] = (disk[f] or '') .. s return true end,
    remove = function(f) disk[f] = nil return true end,
    createDirectory = noop,
  },
  data = {compress = function(_, _, s) return s end, decompress = function(_, _, s) return s end},
  system = {getOS = function() return 'Windows' end},
  thread = {getChannel = function() return {demand = function() return table.remove(queue, 1) end, push = noop} end},
}
Game = {}
function Game:delete_run() end
function Game:main_menu() end
-- The game's own start of a run, as far as the profile and saves go.
function Game:start_run(args)
  G.SAVED_GAME = nil
  local back = G.GAME.viewed_back or G.P_CENTERS.b_red
  for _, c in pairs(G.P_CENTERS) do if args.deck_choice and c.name == args.deck_choice.name then back = c end end
  G.GAME = {won = false, seeded = args.seed ~= nil, stake = 1, round = 0, round_resets = {ante = 1},
            pseudorandom = {seed = args.seed or 'RUN'}, selected_back = {effect = {center = back}}}
  G:save_settings()
  unlock_notify()
end
Card = {init = noop, apply_to_run = noop, open = noop, set_ability = noop, remove = noop}
Controller = {queue_L_cursor_press = noop, queue_R_cursor_press = noop, key_press = noop, button_press = noop}
JSON = {decode = function() error('no antezero.json') end}
sendInfoMessage, sendErrorMessage, get_current_pool = noop, noop, noop
set_discover_tallies, set_profile_progress, set_challenge_unlock, unlock_achievement = noop, noop, noop, noop
function create_unlock_overlay(key) shown[#shown+1] = key end
SMODS = {current_mod = {config = {count = false, speed = 1, search = false}}, Atlas = function() return {key = 'bot'} end,
         RunSelect = {Functions = {clean_up = noop}, Setup = {choices = {}}, Pages = {}}, RunSelectPage = {obj_buffer = {}},
         build_stake_chain = function() return {} end}
AZB = {VERSION = 'test', rpc = {BAD_REQUEST = 'BAD_REQUEST', NOT_ALLOWED = 'NOT_ALLOWED', INVALID_STATE = 'INVALID_STATE', INTERNAL_ERROR = 'INTERNAL_ERROR'}}
function AZB.rpc.register(r) AZB.rpc[r.name] = r.execute end
local white = {key = 'stake_white', order = 1, vanilla_index = 1}
local red = {key = 'b_red', name = 'Red Deck', order = 1, unlocked = true, discovered = true}
local blue = {key = 'b_blue', name = 'Blue Deck', order = 2, unlocked = true, discovered = true}
G = setmetatable({
  FUNCS = {start_run = function(e, args) G:delete_run() G:start_run(args) end},
  ARGS = {}, C = {}, UIT = {},
  SETTINGS = {profile = 1, tutorial_complete = true, GAMESPEED = 4, current_setup = 'New Run', paused = false},
  PROFILES = {{last_choices = {deck_choice = 'b_red', stake_choice = 'stake_white'}, MEMORY = {stake = 1},
               high_scores = {current_streak = {amt = 4}}, career_stats = {c_wins = 2},
               deck_usage = {b_blue = {count = 1, order = 2, wins = {[1] = 2}, losses = {}}}}},
  P_CENTERS = {b_red = red, b_blue = blue,
               j_joker = {key = 'j_joker', name = 'Joker', order = 1, unlocked = true, discovered = true},
               j_blueprint = {key = 'j_blueprint', name = 'Blueprint', order = 2, unlocked = false, discovered = false}},
  P_BLINDS = {}, P_TAGS = {}, P_SEALS = {}, P_STAKES = {stake_white = white},
  P_CENTER_POOLS = {Back = {red, blue}, Stake = {white}},
  SAVE_MANAGER = {channel = {push = function(_, msg) queue[#queue+1] = STR_UNPACK(STR_PACK(msg)) end}}, -- a copy, as LOVE sends it
  GAME = {won = false},
  STATES = {SHOP = 5, GAME_OVER = 4},
}, {__index = Game})
G.P_LOCKED = {G.P_CENTERS.j_blueprint}
G.FUNCS.exit_overlay_menu = function() G.OVERLAY_MENU = nil end
"""

MOD = r"""
convert_save_data(G.PROFILES[1], true) -- as Steamodded loads a profile
AZB.json = dofile(MOD .. 'src/json.lua')
for _, f in ipairs({'core', 'act', 'bot', 'endpoints'}) do dofile(MOD .. 'src/' .. f .. '.lua') end
AZB.state = function() return '{}' end
AZB.bot.make_hud = function() end
AZB.bot.enabled = true

-- The game's save step (in Game:update), then the save thread's writes.
function flush()
  local f = G.FILE_HANDLER
  if f and f.update_queued then
    if f.progress then G.SAVE_MANAGER.channel:push({type = 'save_progress', save_progress = G.ARGS.save_progress})
    elseif f.settings then
      G.SAVE_MANAGER.channel:push({type = 'save_settings', save_settings = G.ARGS.save_settings,
                                   profile_num = G.SETTINGS.profile, save_profile = G.PROFILES[G.SETTINGS.profile]})
    end
    if f.run then
      G.SAVE_MANAGER.channel:push({type = 'save_run', save_table = G.ARGS.save_run, profile_num = G.SETTINGS.profile})
      G.SAVED_GAME = nil
    end
    G.FILE_HANDLER = {}
  end
  G.SAVE_MANAGER.channel:push({type = 'kill'})
  save_thread()
end

-- misc_functions' save_run, as far as saving goes
function save_run()
  if G.F_NO_SAVING == true then return end
  G.ARGS.save_run = {GAME = {back = G.GAME.selected_back.effect.center.key}}
  G.FILE_HANDLER = G.FILE_HANDLER or {}
  G.FILE_HANDLER.run, G.FILE_HANDLER.update_queued = true, true
end

function play(deck) -- the run select's Play button with `deck` picked
  SMODS.RunSelect.Setup.choices = {deck_choice = deck, stake_choice = 'stake_white'}
  G.FUNCS.run_select_start_run()
end
function profile() return G.PROFILES[1] end
function on_disk(file) return disk[file] and STR_UNPACK(disk[file]) end
function deck_wins(entry) return convert_usage_entry(STR_UNPACK(STR_PACK(entry))).wins[1] end -- as saved and loaded
"""


class _Lua:
    def __init__(self) -> None:
        lib = ctypes.CDLL(str(LUA51))
        lib.luaL_newstate.restype = ctypes.c_void_p
        lib.luaL_openlibs.argtypes = [ctypes.c_void_p]
        lib.luaL_loadbuffer.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
        lib.lua_pcall.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.lua_tolstring.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.lua_tolstring.restype = ctypes.c_char_p
        lib.lua_settop.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.lua_close.argtypes = [ctypes.c_void_p]
        self.lib, self.state = lib, lib.luaL_newstate()
        lib.luaL_openlibs(self.state)

    def run(self, code: str, name: str = "test") -> str | None:
        lib, L = self.lib, self.state
        b = code.encode()
        # LuaJIT raises Lua errors as Windows exceptions, which faulthandler
        # reports as fatal even when a pcall catches them.
        reporting = faulthandler.is_enabled()
        faulthandler.disable()
        try:
            rc = lib.luaL_loadbuffer(L, b, len(b), name.encode()) or lib.lua_pcall(L, 0, 1, 0)
        finally:
            if reporting:
                faulthandler.enable()
        out = lib.lua_tolstring(L, -1, None)
        lib.lua_settop(L, 0)
        if rc:
            raise AssertionError((out or b"?").decode())
        return out.decode() if out is not None else None

    def close(self) -> None:
        self.lib.lua_close(self.state)


@pytest.fixture
def game():
    if not (LUA51.exists() and PATCHED.exists() and SMODS_SRC.exists() and (GAME_DIR / "Balatro.exe").exists()):
        pytest.skip("needs the game's lua51.dll, its Lovely dump and Steamodded")
    lua = _Lua()
    mod = (ROOT / "mod").as_posix() + "/"
    lua.run(WORLD, "world")
    lua.run(_sources(), "game")
    lua.run(f"MOD = {mod!r}\n" + MOD, "mod")
    yield lua
    lua.close()


def test_a_win_in_a_run_that_does_not_count_stays_out_of_the_profile(game):
    game.run("""
    play('b_blue')
    assert(AZB.bot.protected, 'a bot run that does not count')
    set_deck_win()
    G:main_menu()
    local e = profile().deck_usage.b_blue
    assert(e.wins_by_key.stake_white == 2, 'the per-stake view after the run: ' .. tostring(e.wins_by_key.stake_white))
    AZB.bot.enabled = false
    play('b_blue')
    set_deck_win() -- the player's own win
    flush()
    local saved = on_disk('1/profile.jkr').deck_usage.b_blue
    assert(deck_wins(saved) == 3, 'wins saved: ' .. deck_wins(saved))
    """)


def test_a_run_started_from_the_screen_of_one_that_does_not_count_keeps_its_choices(game):
    game.run("""
    play('b_red')
    assert(AZB.bot.protected)
    SMODS.current_mod.config.count = true -- the next bot run counts
    play('b_blue')
    assert(G.GAME.selected_back.effect.center.key == 'b_blue' and AZB.bot.active and not AZB.bot.protected)
    flush()
    assert(profile().last_choices.deck_choice == 'b_blue', 'in memory')
    assert(on_disk('1/profile.jkr').last_choices.deck_choice == 'b_blue', 'saved')
    """)
    # the player's own run, from the screen of a bot run that does not count
    game.run("""
    SMODS.current_mod.config.count = false
    play('b_blue')
    assert(AZB.bot.protected)
    AZB.bot.enabled = false
    play('b_red')
    assert(not AZB.bot.protected and not AZB.bot.active)
    flush()
    assert(profile().last_choices.deck_choice == 'b_red', 'in memory')
    assert(on_disk('1/profile.jkr').last_choices.deck_choice == 'b_red', 'saved')
    """)


def test_a_bot_run_that_does_not_count_leaves_the_choices_and_streak_as_they_were(game):
    game.run("""
    G.SAVED_GAME = {GAME = {won = false}} -- the player's run saved at the menu: Play abandons it
    play('b_blue')
    assert(AZB.bot.protected and profile().high_scores.current_streak.amt == 0)
    flush()
    local saved = on_disk('1/profile.jkr')
    assert(saved.high_scores.current_streak.amt == 4 and saved.last_choices.deck_choice == 'b_red', 'saved during the run')
    G:main_menu()
    assert(profile().high_scores.current_streak.amt == 4 and profile().last_choices.deck_choice == 'b_red')
    """)


def test_the_skipped_tutorial_is_never_saved(game):
    game.run("""
    SMODS.current_mod.config.count = true
    G.SETTINGS.tutorial_complete, G.SETTINGS.tutorial_progress = false, {section = 'small_blind', completed_parts = {}}
    play('b_red')
    assert(G.SETTINGS.tutorial_complete, 'skipped for the run')
    flush()
    local s = on_disk('settings.jkr')
    assert(s.tutorial_complete == false and s.tutorial_progress.section == 'small_blind', 'the start of the run saved the skip')
    G:save_progress()
    flush()
    assert(on_disk('settings.jkr').tutorial_complete == false, 'a progress save during the run saved the skip')
    G:main_menu()
    assert(G.SETTINGS.tutorial_complete == false and G.SETTINGS.tutorial_progress.section == 'small_blind')
    """)


def test_a_run_that_does_not_count_saves_only_the_players_own_state(game):
    game.run("""
    AZB.bot.enabled = false
    play('b_red')
    save_run()          -- the player's own run queues a checkpoint ...
    G.SETTINGS.GAMESPEED = 2
    G:save_settings()   -- ... and a settings change
    disk['1/unlock_notify.jkr'] = 'j_joker\\n' -- an unlock not shown yet
    AZB.bot.enabled = true
    play('b_blue')      -- and Play is pressed before the game sent them
    assert(AZB.bot.protected)
    flush()
    assert(on_disk('1/save.jkr').GAME.back == 'b_red', "the player's run checkpoint")
    assert(on_disk('settings.jkr').GAMESPEED == 2, "the player's settings")
    assert(disk['1/unlock_notify.jkr'] and #shown == 0, 'the pending unlock was shown and deleted')

    -- the run unlocks and discovers a card, wins, saves; the player changes a setting meanwhile
    local bp = G.P_CENTERS.j_blueprint
    bp.unlocked, bp.discovered = true, true
    profile().career_stats.c_wins = 3
    G:save_progress()
    G.SETTINGS.GAMESPEED = 3
    G:save_settings()
    save_run()
    remove_save()
    flush()
    assert(on_disk('settings.jkr').GAMESPEED == 3, "the player's setting is saved")
    local p = on_disk('1/profile.jkr')
    assert(p.career_stats.c_wins == 2, 'the run saved its stats')
    assert(not (on_disk('1/meta.jkr') or {unlocked = {}}).unlocked.j_blueprint, 'the run saved its unlock')
    assert(on_disk('1/save.jkr').GAME.back == 'b_red', 'the run was saved, or deleted the player\\'s')

    G:main_menu()
    assert(not bp.unlocked and profile().career_stats.c_wins == 2 and G.SETTINGS.GAMESPEED == 3)
    """)


def test_the_developer_tools_save_override_does_not_reach_a_bot_run(game):
    game.run("""
    AZB.config.save_guard = false -- AZB_ALLOW_SAVES=1
    disk['1/save.jkr'] = STR_PACK({GAME = {back = 'b_red'}})
    play('b_blue')
    save_run()
    remove_save()
    flush()
    assert(on_disk('1/save.jkr').GAME.back == 'b_red')
    """)


def test_every_play_button_starts_the_bot_but_holding_r_restarts_the_players_run(game):
    game.run("""
    G.FUNCS.run_select_quick_start()
    assert(AZB.bot.active, 'Quick Start')
    G:main_menu()
    G.FUNCS.start_setup_run({config = {}})
    assert(AZB.bot.active, "the vanilla screen's Play")
    G:main_menu()
    G.FUNCS.start_setup_run() -- holding R
    assert(not AZB.bot.active and not AZB.bot.protected, 'a restart')
    """)


def test_a_run_handed_back_says_it_does_not_count(game):
    game.run("""
    play('b_red')
    AZB.rpc.az_bot({run = AZB.bot.run.id, stop = 'stopped: test'}, function() end)
    AZB.bot.tick()
    assert(AZB.bot.line:find('does not count'), AZB.bot.line)
    """)


def test_the_bot_hears_of_a_menu_opened_since_its_last_action(game):
    game.run("""
    play('b_red')
    local reply
    local function poll() AZB.rpc.az_bot({run = AZB.bot.run.id}, function(r) reply = r end) return reply end
    assert(not poll().interrupted)
    G.OVERLAY_MENU = {}
    AZB.bot.tick()
    assert(poll().paused and reply.interrupted)
    G.OVERLAY_MENU = nil
    AZB.bot.tick()
    assert(not poll().paused and reply.interrupted, 'closed since')
    AZB.bot.acting({type = 'reroll'}, '')
    assert(not poll().interrupted, 'the next action')
    """)


def test_a_buy_the_game_refuses_is_refused(game):
    reply = game.run("""
    G.STATE, G.STATE_COMPLETE = G.STATES.SHOP, true
    G.shop, G.E_MANAGER, G.CONTROLLER, G.TIMERS = {}, {queues = {base = {}}}, {locks = {}}, {TOTAL = 0}
    G.shop_jokers = {cards = {{is = function() return true end}}}
    G.FUNCS.check_for_buy_space = function() return false end -- a full joker row
    AZB.quiet = AZB.config.quiet_frames
    local reply
    AZB.rpc.az_act({action = {type = 'buy', index = 1}}, function(r) reply = r end)
    for _ = 1, AZB.config.quiet_frames + 1 do AZB.tick() end
    return AZB.json.encode(reply)
    """)
    assert '"name":"BAD_REQUEST"' in reply and "no room" in reply
