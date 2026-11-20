use std::env;
use std::ffi::CStr;
use std::path::{Path, PathBuf};

use crate::lua::State;

/// The harness' entry point: it evaluates one Lua expression (see `lua/harness.lua`).
const HARNESS_ENTRY: &CStr = c"oracle";

#[must_use]
pub fn default_dll() -> PathBuf {
	if let Some(p) = env::var_os("BALATRO_LUA51") {
		return PathBuf::from(p);
	}
	PathBuf::from(r"C:\Program Files (x86)\Steam\steamapps\common\Balatro\lua51.dll")
}

#[must_use]
pub fn default_src() -> PathBuf {
	if let Some(p) = env::var_os("BALATRO_SRC") {
		return PathBuf::from(p);
	}
	Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/balatro")
}

fn lua_dir() -> PathBuf {
	Path::new(env!("CARGO_MANIFEST_DIR")).join("lua")
}

/// Whether the bare `LuaJIT` state can be created on this machine.
#[must_use]
pub fn dll_available() -> bool {
	cfg!(windows) && default_dll().exists()
}

/// Whether the full game harness can run on this machine.
#[must_use]
pub fn available() -> bool {
	dll_available() && default_src().join("game.lua").exists()
}

/// `s` as a Lua 5.1 string literal.
#[must_use]
pub fn lua_str(s: &str) -> String {
	use std::fmt::Write;

	let mut out = String::with_capacity(s.len() + 2);
	out.push('"');
	for &b in s.as_bytes() {
		match b {
			b'"' => out.push_str("\\\""),
			b'\\' => out.push_str("\\\\"),
			0x20..=0x7e => out.push(char::from(b)),
			// Control and non-ASCII bytes as decimal escapes (byte exact).
			_ => write!(out, "\\{b:03}").expect("writing to a String cannot fail"),
		}
	}
	out.push('"');
	out
}

/// The path `p` as a Lua string literal, with forward slashes (which Lua's
/// `dofile` and `package.path` take on Windows too).
#[must_use]
pub fn lua_path(p: &Path) -> String {
	lua_str(&p.display().to_string().replace('\\', "/"))
}

/// A live `LuaJIT` state, optionally running the Balatro harness.
#[derive(Debug)]
pub struct Oracle {
	lua: State,
}

impl Oracle {
	/// Loads `lua51.dll` and creates a bare `LuaJIT` state (no harness).
	///
	/// # Errors
	///
	/// Returns why the DLL could not be loaded or the state not created.
	pub fn new_bare() -> Result<Self, String> {
		Ok(Oracle {
			lua: State::new(&default_dll())?,
		})
	}

	/// Loads the full headless game harness.
	///
	/// # Errors
	///
	/// Returns why the DLL or the harness could not be loaded.
	pub fn new() -> Result<Self, String> {
		let mut o = Self::new_bare()?;
		let boot = format!(
			"HARNESS_SRC = {}\nHARNESS_DIR = {}\npackage.path = HARNESS_DIR..'/?.lua;'..HARNESS_SRC..'/?.lua;'..package.path\nrequire('harness')",
			lua_path(&default_src()),
			lua_path(&lua_dir())
		);
		o.exec(&boot)?;
		Ok(o)
	}

	/// Runs a chunk of Lua code.
	///
	/// # Errors
	///
	/// Returns the Lua error message.
	pub fn exec(&mut self, code: &str) -> Result<(), String> {
		self.lua.exec(code)
	}

	/// Evaluates a Lua chunk that returns a string (or a number; `nil`
	/// gives an empty string).
	///
	/// # Errors
	///
	/// Returns the Lua error message, or what the chunk returned instead.
	pub fn eval(&mut self, code: &str) -> Result<String, String> {
		self.lua.eval(code)
	}

	/// Evaluates the harness expression `expr` (such as `H.act{...}`) and
	/// returns its string result (as [`eval`](Self::eval)), with a traceback
	/// on errors.
	///
	/// # Errors
	///
	/// Returns the Lua error message, or what the expression returned instead.
	pub fn command(&mut self, expr: &str) -> Result<String, String> {
		self.lua.call(HARNESS_ENTRY, expr)
	}
}
