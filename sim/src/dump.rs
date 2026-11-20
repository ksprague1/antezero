//! The game's state dump (`H.state` of the oracle harness and `az_state` of
//! the mod, see `sim/oracle/lua/harness.lua`) as a simulator [`Game`], and a
//! simulator game as such a dump.
//!
//! A dump is taken at a decision point, where the simulator's event queue is
//! empty and the per-action flags are clear, so a game is the dump's
//! `G.GAME`, card areas and RNG streams. Three things the simulator keeps are
//! not the game's and are derived: [`Run::blinds_cleared`],
//! [`Shop::voucher_seen`] and the free rearranges since the last other
//! action (none).
//!
//! [`Game`]: crate::Game
//! [`Run::blinds_cleared`]: crate::game::Run::blinds_cleared
//! [`Shop::voucher_seen`]: crate::game::Shop::voucher_seen

mod read;
mod write;

pub use read::DumpError;
pub(crate) use read::read;
pub(crate) use write::write;
