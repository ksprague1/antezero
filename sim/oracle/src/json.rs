//! A small strict JSON parser for the harness' state dumps (numbers the
//! harness cannot write as JSON are strings: `"inf"`, `"-inf"`, `"nan"`).

use std::collections::BTreeMap;

/// A parsed JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
	Null,
	Bool(bool),
	Num(f64),
	Str(String),
	Arr(Vec<Json>),
	Obj(BTreeMap<String, Json>),
}

impl Json {
	/// Member `k` of an object (`Null` if absent or not an object).
	#[must_use]
	pub fn get(&self, k: &str) -> &Json {
		static NULL: Json = Json::Null;
		match self {
			Json::Obj(m) => m.get(k).unwrap_or(&NULL),
			_ => &NULL,
		}
	}

	/// The number, also one the harness wrote as `"inf"`, `"-inf"` or
	/// `"nan"`.
	#[must_use]
	pub fn as_f64(&self) -> Option<f64> {
		match self {
			Json::Num(n) => Some(*n),
			Json::Str(s) if s == "inf" => Some(f64::INFINITY),
			Json::Str(s) if s == "-inf" => Some(f64::NEG_INFINITY),
			Json::Str(s) if s == "nan" => Some(f64::NAN),
			_ => None,
		}
	}

	#[must_use]
	pub fn as_str(&self) -> Option<&str> {
		if let Json::Str(s) = self {
			Some(s)
		} else {
			None
		}
	}

	/// A flag the harness dumps: `null` (Lua's `nil`) is false.
	///
	/// # Panics
	///
	/// If the value is neither a boolean nor `null`.
	#[must_use]
	pub fn as_flag(&self) -> bool {
		match self {
			Json::Bool(b) => *b,
			Json::Null => false,
			other => panic!("expected a flag (a boolean or null), got {other:?}"),
		}
	}

	/// A list the harness dumps: `null` (Lua's `nil`) is empty.
	///
	/// # Panics
	///
	/// If the value is neither an array nor `null`.
	#[must_use]
	pub fn as_list(&self) -> &[Json] {
		match self {
			Json::Arr(a) => a,
			Json::Null => &[],
			other => panic!("expected an array or null, got {other:?}"),
		}
	}

	#[must_use]
	pub fn as_object(&self) -> Option<&BTreeMap<String, Json>> {
		if let Json::Obj(m) = self {
			Some(m)
		} else {
			None
		}
	}
}

/// Parses one JSON document (nothing but whitespace may follow it).
///
/// # Errors
///
/// Returns what is malformed and at which byte.
pub fn parse(s: &str) -> Result<Json, String> {
	let mut p = Parser { src: s, i: 0 };
	let v = p.value()?;
	p.ws();
	if p.i != s.len() {
		return Err(p.err("trailing characters"));
	}
	Ok(v)
}

struct Parser<'src> {
	src: &'src str,
	i: usize,
}

impl Parser<'_> {
	fn err(&self, what: &str) -> String {
		format!("{what} at byte {}", self.i)
	}

	fn bytes(&self) -> &[u8] {
		self.src.as_bytes()
	}

	fn ws(&mut self) {
		while self.i < self.src.len()
			&& matches!(self.bytes()[self.i], b' ' | b'\t' | b'\n' | b'\r')
		{
			self.i += 1;
		}
	}

	fn peek(&self) -> Result<u8, String> {
		self.bytes()
			.get(self.i)
			.copied()
			.ok_or_else(|| self.err("unexpected end"))
	}

	fn expect(&mut self, c: u8) -> Result<(), String> {
		if self.peek()? != c {
			return Err(self.err(&format!("expected {:?}", char::from(c))));
		}
		self.i += 1;
		Ok(())
	}

	fn literal(&mut self, word: &str, v: Json) -> Result<Json, String> {
		if !self.src[self.i..].starts_with(word) {
			return Err(self.err("bad literal"));
		}
		self.i += word.len();
		Ok(v)
	}

	/// Skips the ASCII digits at the cursor; returns how many there were.
	fn digits(&mut self) -> usize {
		let start = self.i;
		while self.bytes().get(self.i).is_some_and(u8::is_ascii_digit) {
			self.i += 1;
		}
		self.i - start
	}

	/// A number: `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`.
	fn number(&mut self) -> Result<Json, String> {
		let start = self.i;
		if self.peek()? == b'-' {
			self.i += 1;
		}
		match self.bytes().get(self.i) {
			Some(b'0') => self.i += 1,
			Some(b'1'..=b'9') => {
				self.digits();
			}
			_ => return Err(self.err("bad number")),
		}
		if self.bytes().get(self.i) == Some(&b'.') {
			self.i += 1;
			if self.digits() == 0 {
				return Err(self.err("bad number"));
			}
		}
		if matches!(self.bytes().get(self.i), Some(b'e' | b'E')) {
			self.i += 1;
			if matches!(self.bytes().get(self.i), Some(b'+' | b'-')) {
				self.i += 1;
			}
			if self.digits() == 0 {
				return Err(self.err("bad number"));
			}
		}
		let text = &self.src[start..self.i];
		text.parse::<f64>()
			.map(Json::Num)
			.map_err(|e| format!("bad number {text:?} at byte {start}: {e}"))
	}

	fn value(&mut self) -> Result<Json, String> {
		self.ws();
		match self.peek()? {
			b'n' => self.literal("null", Json::Null),
			b't' => self.literal("true", Json::Bool(true)),
			b'f' => self.literal("false", Json::Bool(false)),
			b'"' => Ok(Json::Str(self.string()?)),
			b'[' => {
				self.i += 1;
				let mut v = Vec::new();
				self.ws();
				if self.peek()? == b']' {
					self.i += 1;
					return Ok(Json::Arr(v));
				}
				loop {
					v.push(self.value()?);
					self.ws();
					match self.peek()? {
						b',' => self.i += 1,
						b']' => {
							self.i += 1;
							return Ok(Json::Arr(v));
						}
						_ => return Err(self.err("expected ',' or ']'")),
					}
				}
			}
			b'{' => {
				self.i += 1;
				let mut m = BTreeMap::new();
				self.ws();
				if self.peek()? == b'}' {
					self.i += 1;
					return Ok(Json::Obj(m));
				}
				loop {
					self.ws();
					let k = self.string()?;
					self.ws();
					self.expect(b':')?;
					let v = self.value()?;
					m.insert(k, v);
					self.ws();
					match self.peek()? {
						b',' => self.i += 1,
						b'}' => {
							self.i += 1;
							return Ok(Json::Obj(m));
						}
						_ => return Err(self.err("expected ',' or '}'")),
					}
				}
			}
			b'-' | b'0'..=b'9' => self.number(),
			_ => Err(self.err("unexpected character")),
		}
	}

	/// The four hex digits of a `\u` escape.
	fn hex4(&mut self) -> Result<u32, String> {
		let h = self
			.bytes()
			.get(self.i..self.i + 4)
			.ok_or_else(|| self.err("short \\u escape"))?;
		let v = h
			.iter()
			.try_fold(0, |v, &b| Some(v * 16 + char::from(b).to_digit(16)?))
			.ok_or_else(|| self.err("bad \\u escape"))?;
		self.i += 4;
		Ok(v)
	}

	fn string(&mut self) -> Result<String, String> {
		self.expect(b'"')?;
		let mut out = String::new();
		// Unescaped text is copied in runs; a run ends at an ASCII byte (`"`
		// or `\`), which is a char boundary.
		let mut run = self.i;
		loop {
			let c = self.peek().map_err(|_| self.err("unterminated string"))?;
			match c {
				b'"' => {
					out.push_str(&self.src[run..self.i]);
					self.i += 1;
					return Ok(out);
				}
				b'\\' => {
					out.push_str(&self.src[run..self.i]);
					self.i += 1;
					let e = self.peek()?;
					self.i += 1;
					let ch = match e {
						b'"' => '"',
						b'\\' => '\\',
						b'/' => '/',
						b'b' => '\u{8}',
						b'f' => '\u{c}',
						b'n' => '\n',
						b'r' => '\r',
						b't' => '\t',
						b'u' => {
							let hi = self.hex4()?;
							let code = if (0xd800..0xdc00).contains(&hi) {
								if !self.src[self.i..].starts_with("\\u") {
									return Err(self.err("unpaired surrogate"));
								}
								self.i += 2;
								let lo = self.hex4()?;
								if !(0xdc00..0xe000).contains(&lo) {
									return Err(self.err("unpaired surrogate"));
								}
								0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00)
							} else {
								hi
							};
							char::from_u32(code).ok_or_else(|| self.err("bad \\u escape"))?
						}
						_ => return Err(self.err("bad escape")),
					};
					out.push(ch);
					run = self.i;
				}
				0..0x20 => return Err(self.err("control character in string")),
				_ => self.i += 1,
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn parses_harness_output() {
		let j = parse(r#" {"a":[1,-2.5e3,"inf",0,-0.5,1E+2],"b":{"c":true,"d":null},"e":"x\"\\\n\u00e9\u0001\ud83d\ude00","f":[]} "#).unwrap();
		assert_eq!(j.get("a").as_list().len(), 6);
		assert_eq!(j.get("a").as_list()[1].as_f64(), Some(-2500.0));
		assert_eq!(j.get("a").as_list()[2].as_f64(), Some(f64::INFINITY));
		assert_eq!(j.get("a").as_list()[4].as_f64(), Some(-0.5));
		assert_eq!(j.get("a").as_list()[5].as_f64(), Some(100.0));
		assert!(j.get("b").get("c").as_flag());
		assert_eq!(j.get("b").get("d"), &Json::Null);
		assert_eq!(j.get("e").as_str(), Some("x\"\\\né\u{1}😀"));
		assert_eq!(parse("\"\\t\\r\\/é\"").unwrap(), Json::Str("\t\r/é".into()));
		assert_eq!(parse("{}").unwrap(), Json::Obj(BTreeMap::new()));
	}

	#[test]
	fn rejects_malformed_input() {
		for bad in [
			"",
			"n",
			"nul",
			"true garbage",
			"[1 2]",
			"[",
			"[1,",
			"{\"a\" 1}",
			"{\"a\":1,}",
			"[1,]",
			"\"abc",
			"\"\\x\"",
			"\"\\u12\"",
			"\"\\u+041\"",
			"\"\\ud800\"",
			"\"a\nb\"",
			"01",
			"01x",
			"1.",
			"-.5",
			"1.e5",
			"1e",
			"+1",
			"{1:2}",
			"-",
		] {
			assert!(parse(bad).is_err(), "{bad:?} parsed");
		}
	}
}
