//! Stands in for `lua.rs` off Windows, where the game's `lua51.dll` cannot
//! load: no Lua state can exist, so the oracle compiles but never runs.

use std::ffi::CStr;
use std::path::Path;

#[derive(Debug)]
pub enum State {}

impl State {
	pub fn new(_dll: &Path) -> Result<State, String> {
		Err("the oracle needs Windows (it runs the game's lua51.dll)".into())
	}

	pub fn exec(&mut self, _code: &str) -> Result<(), String> {
		match *self {}
	}

	pub fn eval(&mut self, _code: &str) -> Result<String, String> {
		match *self {}
	}

	pub fn call(&mut self, _name: &CStr, _arg: &str) -> Result<String, String> {
		match *self {}
	}
}
