//! The Lua 5.1 C API of the game's `lua51.dll`, loaded at runtime.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::os::windows::ffi::OsStrExt;
use std::path::{self, Path};
use std::{iter, mem, ptr, slice};

/// A `lua_State`, only ever behind a pointer.
#[repr(C)]
struct LuaState {
	_opaque: [u8; 0],
}

/// A loaded module (`HMODULE`), only ever behind a pointer.
#[repr(C)]
struct ModuleHandle {
	_opaque: [u8; 0],
}

const LUA_GLOBALSINDEX: c_int = -10002;
const LUA_TNIL: c_int = 0;
const LUA_TNUMBER: c_int = 3;
const LUA_TSTRING: c_int = 4;
/// Name of the chunks passed to `luaL_loadbuffer` (shown in errors).
const CHUNK: &CStr = c"=oracle";

#[link(name = "kernel32")]
unsafe extern "system" {
	fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut ModuleHandle;
	fn GetProcAddress(module: *mut ModuleHandle, name: *const c_char) -> *mut c_void;
	fn FreeLibrary(module: *mut ModuleHandle) -> c_int;
	fn GetLastError() -> u32;
}
const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x0000_0008;

#[derive(Debug)]
struct Api {
	new_state: unsafe extern "C" fn() -> *mut LuaState,
	open_libs: unsafe extern "C" fn(*mut LuaState),
	load_buffer: unsafe extern "C" fn(*mut LuaState, *const c_char, usize, *const c_char) -> c_int,
	pcall: unsafe extern "C" fn(*mut LuaState, c_int, c_int, c_int) -> c_int,
	to_lstring: unsafe extern "C" fn(*mut LuaState, c_int, *mut usize) -> *const c_char,
	lua_type: unsafe extern "C" fn(*mut LuaState, c_int) -> c_int,
	set_top: unsafe extern "C" fn(*mut LuaState, c_int),
	get_field: unsafe extern "C" fn(*mut LuaState, c_int, *const c_char),
	push_lstring: unsafe extern "C" fn(*mut LuaState, *const c_char, usize),
	close: unsafe extern "C" fn(*mut LuaState),
}

#[derive(Debug)]
struct Module(*mut ModuleHandle);

impl Drop for Module {
	fn drop(&mut self) {
		// SAFETY: `self.0` is the handle `LoadLibraryExW` returned, released only here.
		unsafe { FreeLibrary(self.0) };
	}
}

/// A Lua state of the game's `LuaJIT`. Fields drop in order: the state is
/// closed before the DLL is released.
#[derive(Debug)]
pub struct State {
	api: Api,
	l: *mut LuaState,
	_module: Module,
}

impl Drop for State {
	fn drop(&mut self) {
		// SAFETY: `l` is the live state `new` created and is not used again;
		// the DLL is still loaded (`_module` drops after this).
		unsafe { (self.api.close)(self.l) };
	}
}

impl State {
	pub fn new(dll: &Path) -> Result<State, String> {
		// `LOAD_WITH_ALTERED_SEARCH_PATH` is undefined for a relative path.
		let dll = path::absolute(dll).map_err(|e| format!("{}: {e}", dll.display()))?;
		let wide: Vec<u16> = dll.as_os_str().encode_wide().chain(iter::once(0)).collect();
		// SAFETY: `wide` is a NUL-terminated UTF-16 path that outlives the
		// call, and the reserved file handle is null as required.
		let handle = unsafe {
			LoadLibraryExW(
				wide.as_ptr(),
				ptr::null_mut(),
				LOAD_WITH_ALTERED_SEARCH_PATH,
			)
		};
		if handle.is_null() {
			// SAFETY: `GetLastError` has no preconditions.
			let code = unsafe { GetLastError() };
			return Err(format!("could not load {} (error {code})", dll.display()));
		}
		let module = Module(handle);
		macro_rules! sym {
			($name:literal : $ty:ty) => {{
				// SAFETY: `module.0` is a loaded module and the name is a NUL-terminated literal.
				let p = unsafe {
					GetProcAddress(module.0, concat!($name, "\0").as_ptr().cast::<c_char>())
				};
				if p.is_null() {
					return Err(format!("missing symbol {}", $name));
				}
				// SAFETY: the symbol is the Lua 5.1 C API function of this signature.
				unsafe { mem::transmute::<*mut c_void, $ty>(p) }
			}};
		}
		let api = Api {
			new_state: sym!("luaL_newstate": unsafe extern "C" fn() -> *mut LuaState),
			open_libs: sym!("luaL_openlibs": unsafe extern "C" fn(*mut LuaState)),
			load_buffer: sym!("luaL_loadbuffer": unsafe extern "C" fn(*mut LuaState, *const c_char, usize, *const c_char) -> c_int),
			pcall: sym!("lua_pcall": unsafe extern "C" fn(*mut LuaState, c_int, c_int, c_int) -> c_int),
			to_lstring: sym!("lua_tolstring": unsafe extern "C" fn(*mut LuaState, c_int, *mut usize) -> *const c_char),
			lua_type: sym!("lua_type": unsafe extern "C" fn(*mut LuaState, c_int) -> c_int),
			set_top: sym!("lua_settop": unsafe extern "C" fn(*mut LuaState, c_int)),
			get_field: sym!("lua_getfield": unsafe extern "C" fn(*mut LuaState, c_int, *const c_char)),
			push_lstring: sym!("lua_pushlstring": unsafe extern "C" fn(*mut LuaState, *const c_char, usize)),
			close: sym!("lua_close": unsafe extern "C" fn(*mut LuaState)),
		};
		// SAFETY: `luaL_newstate` has no preconditions.
		let l = unsafe { (api.new_state)() };
		if l.is_null() {
			return Err("luaL_newstate failed".into());
		}
		// SAFETY: `l` is a new, non-null state.
		unsafe { (api.open_libs)(l) };
		Ok(State {
			api,
			l,
			_module: module,
		})
	}

	pub fn exec(&mut self, code: &str) -> Result<(), String> {
		// SAFETY: `self.l` is a live state, borrowed exclusively; Lua copies
		// the chunk, whose pointer is valid for the length passed, and a
		// failed load or call leaves its message on the stack for `pop_error`.
		unsafe {
			if (self.api.load_buffer)(
				self.l,
				code.as_ptr().cast::<c_char>(),
				code.len(),
				CHUNK.as_ptr(),
			) != 0
			{
				return Err(self.pop_error());
			}
			if (self.api.pcall)(self.l, 0, 0, 0) != 0 {
				return Err(self.pop_error());
			}
		}
		Ok(())
	}

	pub fn eval(&mut self, code: &str) -> Result<String, String> {
		// SAFETY: as in `exec`; the successful call leaves its one result on
		// the stack for `pop_result`.
		unsafe {
			if (self.api.load_buffer)(
				self.l,
				code.as_ptr().cast::<c_char>(),
				code.len(),
				CHUNK.as_ptr(),
			) != 0
			{
				return Err(self.pop_error());
			}
			if (self.api.pcall)(self.l, 0, 1, 0) != 0 {
				return Err(self.pop_error());
			}
			self.pop_result()
		}
	}

	/// Calls the global function `name(arg)` and returns its string result
	/// (as [`eval`](Self::eval)).
	pub fn call(&mut self, name: &CStr, arg: &str) -> Result<String, String> {
		// SAFETY: as in `exec`; `name` is NUL-terminated and Lua copies the
		// argument string. The call leaves its one result (or its error
		// message) on the stack for `pop_result` (`pop_error`).
		unsafe {
			(self.api.get_field)(self.l, LUA_GLOBALSINDEX, name.as_ptr());
			(self.api.push_lstring)(self.l, arg.as_ptr().cast::<c_char>(), arg.len());
			if (self.api.pcall)(self.l, 1, 1, 0) != 0 {
				return Err(self.pop_error());
			}
			self.pop_result()
		}
	}

	/// Pops the single result of a call: a string, a number (as text) or
	/// `nil` (an empty string).
	///
	/// # Safety
	///
	/// The result must be on top of the stack.
	unsafe fn pop_result(&mut self) -> Result<String, String> {
		// SAFETY: the caller guarantees a value at index -1 of the live state.
		let t = unsafe { (self.api.lua_type)(self.l, -1) };
		let r = match t {
			LUA_TNIL => Ok(String::new()),
			LUA_TNUMBER | LUA_TSTRING => {
				// SAFETY: the value at index -1 is still there.
				let bytes = unsafe { self.top_bytes() }.expect("Lua converts strings and numbers");
				String::from_utf8(bytes).map_err(|e| format!("the result is not UTF-8: {e}"))
			}
			_ => Err(format!("expected a string result, got Lua type {t}")),
		};
		// SAFETY: clearing the stack of the live state is always valid.
		unsafe { (self.api.set_top)(self.l, 0) };
		r
	}

	/// The bytes of the string or number on top of the stack, or `None` if
	/// it is another type.
	///
	/// # Safety
	///
	/// The stack must not be empty.
	unsafe fn top_bytes(&mut self) -> Option<Vec<u8>> {
		let mut len = 0usize;
		// SAFETY: the caller guarantees a value at index -1; `len` is a valid out pointer.
		let p = unsafe { (self.api.to_lstring)(self.l, -1, &raw mut len) };
		if p.is_null() {
			return None;
		}
		// SAFETY: Lua returned `len` bytes that stay valid while the value is
		// on the stack; they are copied before anything pops it.
		Some(unsafe { slice::from_raw_parts(p.cast::<u8>(), len) }.to_vec())
	}

	/// Pops the error of a failed load or call: its message, or the Lua type
	/// of an error value that is not a string.
	///
	/// # Safety
	///
	/// The error must be on top of the stack.
	unsafe fn pop_error(&mut self) -> String {
		// SAFETY: the caller guarantees a value at index -1 of the live state.
		let message = match unsafe { self.top_bytes() } {
			Some(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
			// SAFETY: as above.
			None => format!("error value of Lua type {}", unsafe {
				(self.api.lua_type)(self.l, -1)
			}),
		};
		// SAFETY: clearing the stack of the live state is always valid.
		unsafe { (self.api.set_top)(self.l, 0) };
		message
	}
}
