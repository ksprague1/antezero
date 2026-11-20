/// Where a token's subject is; the discriminant is the token's zone
/// feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Zone {
	Pad,
	Hand,
	Joker,
	Consumable,
	ShopCard,
	ShopVoucher,
	ShopBooster,
	Pack,
	Tag,
	Blind,
	HandType,
	Deck,
	Discard,
}

impl Zone {
	pub const COUNT: i32 = Zone::Discard.id() + 1;

	/// The zone feature.
	#[must_use]
	pub const fn id(self) -> i32 {
		self as i32
	}

	/// A pile's tokens show every card in full, even a face-down one (only
	/// the deck's tokens also show it face up).
	#[must_use]
	pub const fn is_pile(self) -> bool {
		matches!(self, Zone::Deck | Zone::Discard)
	}
}

/// A zone whose cards get a build preview.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemZone {
	Joker,
	Consumable,
	ShopCard,
	ShopVoucher,
	Pack,
}

impl ItemZone {
	pub const ALL: [ItemZone; 5] = [
		ItemZone::Joker,
		ItemZone::Consumable,
		ItemZone::ShopCard,
		ItemZone::ShopVoucher,
		ItemZone::Pack,
	];
}

impl From<ItemZone> for Zone {
	fn from(zone: ItemZone) -> Zone {
		match zone {
			ItemZone::Joker => Zone::Joker,
			ItemZone::Consumable => Zone::Consumable,
			ItemZone::ShopCard => Zone::ShopCard,
			ItemZone::ShopVoucher => Zone::ShopVoucher,
			ItemZone::Pack => Zone::Pack,
		}
	}
}
