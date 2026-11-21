//! Conversions between the simulator's number types.
//!
//! The game keeps every quantity in a Lua number (an `f64`), Rust indexes with
//! `usize`, and the observation is made of `f32` and `i32`. Each function here
//! is the one place where its `as` conversion is justified. The conversions
//! that must be exact check their range first, in every build.

/// A count or index as a Lua number.
///
/// # Panics
///
/// Panics if `n` is 2^53 or more, which a Lua number does not hold exactly.
#[inline]
#[must_use]
#[expect(
	clippy::cast_precision_loss,
	reason = "`n` was just checked to be below 2^53, so the result is exact"
)]
pub const fn float(n: usize) -> f64 {
	assert!(
		n < 1 << 53,
		"a count this large is not exact as a Lua number"
	);
	n as f64
}

/// The integer part of a Lua number as a count. Like `as`, it saturates:
/// negative numbers and NaN give 0.
#[inline]
#[must_use]
#[expect(
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "truncating and saturating is the point"
)]
pub const fn trunc(x: f64) -> usize {
	x as usize
}

/// A Lua number that holds a whole number, as an `i64`; `None` for any
/// other number (a fraction, an infinity, NaN, or beyond 2^63).
#[must_use]
#[expect(
	clippy::cast_possible_truncation,
	reason = "`x` was just checked to be a whole number within i64's range"
)]
pub fn whole(x: f64) -> Option<i64> {
	const LIMIT: f64 = 9_223_372_036_854_775_808.0; // 2^63
	if x.fract() == 0.0 && (-LIMIT..LIMIT).contains(&x) {
		Some(x as i64)
	} else {
		None
	}
}

/// A Lua number as an observation feature, rounded to the nearest `f32`.
#[inline]
#[must_use]
#[expect(
	clippy::cast_possible_truncation,
	reason = "observation features are f32; rounding is intended"
)]
pub const fn feat(x: f64) -> f32 {
	x as f32
}

/// A small index or table length as a categorical id.
///
/// # Panics
///
/// Panics if `n` does not fit an `i32`.
#[inline]
#[must_use]
#[expect(
	clippy::cast_possible_truncation,
	clippy::cast_possible_wrap,
	reason = "`n` was just checked to fit an i32"
)]
pub const fn id(n: usize) -> i32 {
	assert!(
		n <= i32::MAX as usize,
		"an index this large does not fit an id"
	);
	n as i32
}

/// An index or count as the `u32` of card ids and counters.
///
/// # Panics
///
/// Panics if `n` does not fit a `u32`.
#[inline]
#[must_use]
#[expect(
	clippy::cast_possible_truncation,
	reason = "`n` was just checked to fit a u32"
)]
pub const fn index32(n: usize) -> u32 {
	assert!(
		n <= u32::MAX as usize,
		"an index this large does not fit a u32"
	);
	n as u32
}

/// A `u32` counter as an index.
///
/// # Panics
///
/// Never on the 32- and 64-bit targets the simulator builds for.
#[inline]
#[must_use]
pub fn index(n: u32) -> usize {
	usize::try_from(n).expect("a u32 fits a usize on every target the simulator builds for")
}

/// An index or count as a `u64` (a seed or hash input).
///
/// # Panics
///
/// Never on the 32- and 64-bit targets the simulator builds for.
#[inline]
#[must_use]
pub fn index64(n: usize) -> u64 {
	u64::try_from(n).expect("a usize fits a u64 on every target the simulator builds for")
}

/// A slot index or card count as the `u8` that actions and ops carry.
///
/// # Panics
///
/// Panics if `n` does not fit a `u8`.
#[inline]
#[must_use]
#[expect(
	clippy::cast_possible_truncation,
	reason = "`n` was just checked to fit a u8"
)]
pub const fn byte(n: usize) -> u8 {
	assert!(
		n <= u8::MAX as usize,
		"an index this large does not fit a byte"
	);
	n as u8
}

/// `hash` reduced to an index below `len`.
///
/// # Panics
///
/// Panics if `len` is 0.
#[inline]
#[must_use]
#[expect(
	clippy::cast_possible_truncation,
	reason = "the remainder is below `len`, a usize"
)]
pub const fn pick(hash: u64, len: usize) -> usize {
	(hash % len as u64) as usize
}
