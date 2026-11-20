//! Shop and `G.FUNCS.use_card` (`button_callbacks.lua`, `UI_definitions.lua`
//! `create_card_for_shop`, card.lua `redeem`).

use crate::calculate_joker::LifecycleCtx;
use crate::card::{AnyCard, Card, CardLike, Kind};
use crate::card_lifecycle::Pricing;
use crate::card_mask::CardMask;
use crate::data::{BlindKind, Booster, Edition, Enhancement, Tarot, Voucher};
use crate::event::Event;
use crate::event_queue::{Blocks, EditionTiming, EventQueue, Trigger};
use crate::game::{Area, Areas, Blind, CardSet, Flow, Game, State};
use crate::pools::{NewCard, PlayingSet};
use crate::rng::{Key, trace};
use crate::round::Round;
use crate::run::History;
use crate::{
	areas, blind, calculate_joker, card_lifecycle, consumables, flow, num, packs, pools, tags,
};

/// The boosters a shop offers.
pub const BOOSTER_SLOTS: usize = 2;

/// One of the shop's booster slots this round (`G.GAME.current_round.used_packs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackSlot {
	Unrolled,
	Offered(Booster),
	Opened,
}

/// The shop: whether it is open, what it offers this round, and the rules
/// that price and stock it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shop {
	/// `G.shop` exists.
	pub open: bool,
	/// `G.GAME.shop_free`.
	pub free: bool,
	/// `G.GAME.shop_d6ed`.
	pub d6ed: bool,
	pub reroll_cost: f64,
	pub reroll_cost_increase: f64,
	pub free_rerolls: f64,
	/// The reroll cost a round starts from (`G.GAME.round_resets.reroll_cost`).
	pub base_reroll_cost: f64,
	pub temp_reroll_cost: Option<f64>,
	/// `G.GAME.current_round.used_packs`.
	pub booster_slots: [PackSlot; BOOSTER_SLOTS],
	/// The ante's voucher (`G.GAME.current_round.voucher`).
	pub voucher: Option<Voucher>,
	/// The ante's voucher has been shown in a shop (observation only).
	pub voucher_seen: bool,
	/// `G.GAME.shop.joker_max`.
	pub joker_max: f64,
	pub discount_percent: f64,
	pub edition_rate: f64,
	pub joker_rate: f64,
	pub tarot_rate: f64,
	pub planet_rate: f64,
	pub spectral_rate: f64,
	pub playing_card_rate: f64,
	/// `G.GAME.first_shop_buffoon`.
	pub first_buffoon: bool,
}

impl Default for Shop {
	fn default() -> Shop {
		Shop {
			open: false,
			free: false,
			d6ed: false,
			reroll_cost: 5.0,
			reroll_cost_increase: 0.0,
			free_rerolls: 0.0,
			base_reroll_cost: 1.0,
			temp_reroll_cost: None,
			booster_slots: [PackSlot::Unrolled; BOOSTER_SLOTS],
			voucher: None,
			voucher_seen: false,
			joker_max: 2.0,
			discount_percent: 0.0,
			edition_rate: 1.0,
			joker_rate: 20.0,
			tarot_rate: 4.0,
			planet_rate: 4.0,
			spectral_rate: 0.0,
			playing_card_rate: 0.0,
			first_buffoon: false,
		}
	}
}

pub(crate) fn locate_shop_card(areas: &Areas, id: u32) -> Option<(Area, usize)> {
	[Area::ShopJokers, Area::ShopBoosters, Area::ShopVouchers]
		.into_iter()
		.find_map(|a| Some((a, areas.position(a, id)?)))
}

fn shop_card_mut(areas: &mut Areas, id: u32) -> Option<&mut dyn CardLike> {
	let (a, i) = locate_shop_card(areas, id)?;
	Some(areas.card_at_mut(a, i))
}

pub(crate) fn update_shop(
	flow: &mut Flow, areas: &mut Areas, shop: &mut Shop, events: &mut EventQueue,
) {
	if !flow.state_complete {
		let shop_exists = shop.open;
		if !shop_exists {
			shop.open = true;
			areas.shop_jokers.clear();
			areas.shop_vouchers.clear();
			areas.shop_boosters.clear();
			// G.UIDEF.shop()'s shop-sign event
			events.push(Event::Wait);
		}
		events.push(Event::Shop(ShopEvent::ShopOuter { shop_exists }));
		flow.state_complete = true;
	}
}

fn shop_populate(g: &mut Game, shop_exists: bool) {
	tags::apply_shop_start(&mut g.shop, &mut g.tags, &mut g.events);
	if !shop_exists {
		fill_shop_jokers(g);
		if let Some(v) = g.shop.voucher {
			g.shop.voucher_seen = true;
			let c = make_shop_card(g, Kind::Voucher(v));
			// A 'shop'-type area: insertion order is kept (no sort).
			g.areas.shop_vouchers.push(c);
		}
		for i in 0..BOOSTER_SLOTS {
			let b = match g.shop.booster_slots[i] {
				PackSlot::Unrolled => {
					let b = pools::get_pack(&g.run, &mut g.shop, &mut g.rng);
					g.shop.booster_slots[i] = PackSlot::Offered(b);
					b
				}
				PackSlot::Offered(b) => b,
				PackSlot::Opened => continue,
			};
			let mut c = make_shop_card(g, Kind::Booster(b));
			c.booster_slot = Some(num::byte(i));
			g.areas.shop_boosters.push(c);
		}
		tags::apply_voucher_add(&mut g.tags, &mut g.events);
		tags::apply_shop_final_pass(&mut g.shop, &mut g.tags, &mut g.events);
	}
	trace!(g.rng, "shop_populate");
	// save_run
	g.events.push(Event::Wait);
}

/// A new card of `kind` for a shop area, materializing.
pub(crate) fn make_shop_card(g: &mut Game, kind: Kind) -> Card {
	let c = card_lifecycle::make_card(g, kind);
	g.events.materialize(1.0);
	c
}

/// Fills the joker row up to `shop_joker_max` (the shop's
/// `for i = 1, shop_joker_max - #G.shop_jokers.cards` loops).
fn fill_shop_jokers(g: &mut Game) {
	let n = num::trunc(g.shop.joker_max).saturating_sub(g.areas.shop_jokers.len());
	for _ in 0..n {
		let c = create_card_for_shop(g);
		g.areas.shop_jokers.push(c);
	}
}

/// What a card of the shop's row is rolled as.
#[derive(Clone, Copy, Debug)]
enum ShopRoll {
	Card(CardSet),
	Playing(PlayingSet),
}

/// `create_card_for_shop(G.shop_jokers)`.
pub(crate) fn create_card_for_shop(g: &mut Game) -> AnyCard {
	if let Some(mut c) = tags::create_store_joker(g) {
		tags::modify_store_joker(&mut g.tags, &mut g.events, &mut c);
		return c.into();
	}
	let total = g.shop.joker_rate
		+ g.shop.tarot_rate
		+ g.shop.planet_rate
		+ g.shop.playing_card_rate
		+ g.shop.spectral_rate;
	let key = g.run.key_ante("cdt");
	let polled = g.rng.random(&key) * total;
	let has_illusion = g.history.used_vouchers.contains(Voucher::Illusion);
	let illusion_key = Key::from("illusion");
	let illusion = has_illusion && g.rng.random(&illusion_key) > 0.6;
	let opts = [
		(ShopRoll::Card(CardSet::Joker), g.shop.joker_rate),
		(ShopRoll::Card(CardSet::Tarot), g.shop.tarot_rate),
		(ShopRoll::Card(CardSet::Planet), g.shop.planet_rate),
		(
			ShopRoll::Playing(if illusion {
				PlayingSet::Enhanced
			} else {
				PlayingSet::Base
			}),
			g.shop.playing_card_rate,
		),
		(ShopRoll::Card(CardSet::Spectral), g.shop.spectral_rate),
	];
	let mut check = 0.0;
	for (roll, val) in opts {
		if polled > check && polled <= check + val {
			let mut c: AnyCard = match roll {
				ShopRoll::Card(set) => {
					pools::create_card(g, NewCard::new(set, Area::ShopJokers, "sho")).into()
				}
				ShopRoll::Playing(set) => pools::create_playing_card(
					&mut g.areas,
					&g.run,
					g.shop.discount_percent,
					&mut g.rng,
					set,
					"sho",
				)
				.into(),
			};
			let id = c.core().id;
			g.events
				.push(Event::Shop(ShopEvent::StoreModify { card: id }));
			if matches!(roll, ShopRoll::Playing(_))
				&& has_illusion
				&& g.rng.random(&illusion_key) > 0.8
			{
				let p = g.rng.random(&illusion_key);
				let e = if p > 1.0 - 0.15 {
					Edition::Polychrome
				} else if p > 0.5 {
					Edition::Holo
				} else {
					Edition::Foil
				};
				card_lifecycle::set_edition(
					&mut c,
					Some(e),
					None,
					Pricing::new(g.shop.discount_percent, &g.areas.jokers),
					&mut g.areas.limits,
				);
				g.events.edition_fx(Some(e), EditionTiming::Delayed);
			}
			return c;
		}
		check += val;
	}
	unreachable!("create_card_for_shop: no card type selected (polled {polled})")
}

/// `change_shop_size(mod)`.
pub(crate) fn change_shop_size(g: &mut Game, m: f64) {
	g.shop.joker_max += m;
	if g.shop.open {
		if m < 0.0 {
			while num::float(g.areas.shop_jokers.len()) > g.shop.joker_max {
				let Some(c) = g.areas.shop_jokers.pop() else {
					break;
				};
				card_lifecycle::unmark_used(&g.areas, &mut g.history, &c);
			}
		}
		if m > 0.0 {
			fill_shop_jokers(g);
		}
	}
}

/// `calculate_reroll_cost(true)`: the price of a reroll as things stand
/// (a paid reroll raises `reroll_cost_increase` first).
pub(crate) fn calculate_reroll_cost(shop: &mut Shop) {
	if shop.free_rerolls < 0.0 {
		shop.free_rerolls = 0.0;
	}
	if shop.free_rerolls > 0.0 {
		shop.reroll_cost = 0.0;
		return;
	}
	shop.reroll_cost =
		shop.temp_reroll_cost.unwrap_or(shop.base_reroll_cost) + shop.reroll_cost_increase;
}

pub(crate) fn do_reroll(reroll_cost: f64, events: &mut EventQueue) {
	if reroll_cost > 0.0 {
		events.push(Event::EaseDollars(-reroll_cost));
	}
	events.push(Event::Shop(ShopEvent::RerollShopRun));
	events.push_after(0.3, Event::Shop(ShopEvent::RerollOuter));
	// save_run
	events.push(Event::Wait);
}

/// `G.FUNCS.check_for_buy_space(card)`.
#[must_use]
pub(crate) fn has_buy_space(areas: &Areas, c: &AnyCard) -> bool {
	let AnyCard::Card(c) = c else {
		return true;
	};
	let neg = if c.core.edition == Some(Edition::Negative) {
		1.0
	} else {
		0.0
	};
	match c.kind {
		Kind::Voucher(_) => true,
		Kind::Joker(_) => num::float(areas.jokers.len()) < areas.limits.jokers.limit + neg,
		Kind::Tarot(_) | Kind::Planet(_) | Kind::Spectral(_) => {
			num::float(areas.consumables.len()) < areas.limits.consumables.limit + neg
		}
		Kind::Booster(_) => false,
	}
}

/// Buys the shop card at `idx`; with `use_on`, a consumable that is used
/// at once on those hand cards.
pub(crate) fn do_buy(
	shop_jokers: &[AnyCard], events: &mut EventQueue, idx: usize, use_on: Option<CardMask>,
) {
	let id = shop_jokers[idx].core().id;
	events.push_after(0.1, Event::Shop(ShopEvent::BuyRun { card: id, use_on }));
}

fn buy_run(g: &mut Game, id: u32, use_on: Option<CardMask>) {
	let i = areas::position_of(&g.areas.shop_jokers, id)
		.expect("nothing takes a shop card between the buy and its event");
	let mut c = g.areas.shop_jokers.remove(i);
	assert!(
		use_on.is_none() || c.is_consumable(),
		"only a consumable is bought and used at once (`check_buy_and_use`), not {c:?}"
	);
	card_lifecycle::add_to_deck(g, &mut c);
	let cost = c.core().cost;
	let to_use = match c {
		AnyCard::Playing(mut c) => {
			c.core.face_down = true;
			g.areas.deck.insert(0, c);
			calculate_joker::calc_jokers(g, LifecycleCtx::PlayingCardAdded { n: 1 });
			None
		}
		AnyCard::Card(c) => {
			if let Some(targets) = use_on {
				Some((c, targets))
			} else {
				if c.kind.is_consumable() {
					g.areas.consumables.push(c);
				} else {
					g.areas.jokers.push(c);
				}
				// calculate_joker({buying_card = true, card = g}) event
				g.events.push(Event::Wait);
				None
			}
		}
	};
	if cost != 0.0 {
		g.events.push(Event::EaseDollars(-cost));
	}
	if let Some((c, targets)) = to_use {
		use_card(g, c.into(), Some(Area::ShopJokers), targets);
	}
}

pub(crate) fn do_sell(g: &mut Game, area: Area, idx: usize) {
	let id = g.areas.card_at(area, idx).core().id;
	if area == Area::Jokers {
		calculate_joker::calc_joker(g, idx, LifecycleCtx::SellingSelf);
	}
	// The coin sound's 'after' event, then `delay(0.2)`.
	g.events.delay(0.2);
	g.events.delay(0.2);
	g.events
		.push(Event::Shop(ShopEvent::SellRun { card: id, area }));
	for j in 0..g.areas.jokers.len() {
		if g.areas.jokers[j].core.id != id {
			calculate_joker::calc_joker(g, j, LifecycleCtx::SellingCard);
		}
	}
}

pub(crate) fn do_end_shop(g: &mut Game) {
	calculate_joker::calc_jokers(g, LifecycleCtx::EndingShop);
	g.events.push(Event::Wait);
	g.events
		.push_after(0.5, Event::Shop(ShopEvent::ToggleShopRun));
}

/// `G.FUNCS.use_card` for a card already removed from `area`, with the
/// hand cards `highlighted`.
pub(crate) fn use_card(g: &mut Game, card: AnyCard, area: Option<Area>, highlighted: CardMask) {
	let prev = g.flow.state;
	if !g.flow.state.is_pack() {
		g.flow.state = State::PlayTarot;
	}
	let id = card.core().id;
	let mut dissolve = true;
	let mut delay_factor = 1.0;
	match card {
		AnyCard::Playing(mut c) => {
			c.core.face_down = true;
			calculate_joker::calc_jokers(g, LifecycleCtx::PlayingCardAdded { n: 1 });
			card_lifecycle::add_to_deck(g, &mut c);
			g.areas.deck.insert(0, c);
			dissolve = false;
			delay_factor = 0.2;
		}
		AnyCard::Card(card) => match card.kind {
			k @ (Kind::Tarot(_) | Kind::Planet(_) | Kind::Spectral(_)) => {
				let hanged_glass = if k == Kind::Tarot(Tarot::HangedMan) {
					highlighted
						.indices()
						.filter(|&i| g.areas.hand.cards[i].enhancement == Enhancement::Glass)
						.count()
				} else {
					0
				};
				if !matches!(
					prev,
					State::TarotPack | State::PlanetPack | State::SpectralPack
				) {
					// draw_card(G.hand, G.play, ..., card)
					g.events.wait(Trigger::Before(0.1));
				}
				g.events.delay(0.2);
				consumables::use_consumable(g, &card, highlighted);
				g.areas.consumed.push(card);
				calculate_joker::calc_jokers(g, LifecycleCtx::UsingConsumeable {
					kind: k,
					hanged_glass,
				});
			}
			Kind::Joker(_) => {
				let mut c = card;
				card_lifecycle::add_to_deck(g, &mut c);
				g.areas.jokers.push(c);
				dissolve = false;
				delay_factor = 0.2;
			}
			Kind::Booster(b) => {
				g.events.delay(0.1);
				if let Some(slot) = card.booster_slot {
					g.shop.booster_slots[usize::from(slot)] = PackSlot::Opened;
				}
				g.events.wait(Trigger::Before(0.1));
				packs::open_booster(
					&mut g.flow,
					&mut g.pack,
					&mut g.events,
					b,
					card.core.id,
					card.core.cost,
					prev,
				);
				g.areas.consumed.push(card);
				return;
			}
			Kind::Voucher(v) => {
				g.events.delay(0.1);
				g.events.wait(Trigger::Before(0.1));
				let cost = card.core.cost;
				g.areas.consumed.push(card);
				redeem(
					&mut g.history,
					&mut g.round,
					&mut g.shop,
					&mut g.events,
					v,
					cost,
				);
			}
		},
	}
	g.events.push_after(
		0.2,
		Event::Shop(ShopEvent::UseFinishOuter {
			prev,
			area,
			card: id,
			dissolve,
			delay_factor,
		}),
	);
}

/// `Card:redeem()` for the voucher `v`, bought at `cost`.
fn redeem(
	history: &mut History, round: &mut Round, shop: &mut Shop, events: &mut EventQueue, v: Voucher,
	cost: f64,
) {
	history.used_vouchers.insert(v);
	events.delay(0.4);
	events.push(Event::EaseDollars(-cost));
	shop.voucher = None;
	apply_voucher(round, events, v, VoucherSource::Redeemed);
	events.delay(0.6);
}

/// What gives the run a voucher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VoucherSource {
	Redeemed,
	/// A deck's starting voucher (`from_back`).
	Deck,
}

/// `Card:apply_to_run(center)`.
pub(crate) fn apply_voucher(
	round: &mut Round, events: &mut EventQueue, v: Voucher, source: VoucherSource,
) {
	use Voucher as V;
	let extra = v.def().extra;
	match v {
		V::OverstockNorm | V::OverstockPlus => events.push(Event::Shop(ShopEvent::ShopSize(1.0))),
		V::TarotMerchant | V::TarotTycoon => events.push(Event::Shop(ShopEvent::SetRate {
			which: Rate::Tarot,
			v: 4.0 * extra,
		})),
		V::PlanetMerchant | V::PlanetTycoon => events.push(Event::Shop(ShopEvent::SetRate {
			which: Rate::Planet,
			v: 4.0 * extra,
		})),
		V::Hone | V::GlowUp => events.push(Event::Shop(ShopEvent::SetRate {
			which: Rate::Edition,
			v: extra,
		})),
		V::MagicTrick | V::Illusion => events.push(Event::Shop(ShopEvent::SetRate {
			which: Rate::PlayingCard,
			v: extra,
		})),
		V::CrystalBall => events.push(Event::Shop(ShopEvent::ConsumableSlot)),
		V::ClearanceSale | V::Liquidation => events.push(Event::Shop(ShopEvent::Discount(extra))),
		V::RerollSurplus | V::RerollGlut => {
			if source == VoucherSource::Redeemed {
				events.push(Event::Shop(ShopEvent::RerollDiscount(extra)));
			}
		}
		V::SeedMoney | V::MoneyTree => events.push(Event::Shop(ShopEvent::InterestCap(extra))),
		V::Grabber | V::NachoTong => {
			round.base_hands += extra;
			events.push(Event::EaseHands(extra));
		}
		V::PaintBrush | V::Palette => flow::change_hand_size(events, 1.0),
		V::Wasteful | V::Recyclomancy => {
			round.base_discards += extra;
			events.push(Event::EaseDiscard(extra));
		}
		V::Antimatter => events.push(Event::Shop(ShopEvent::JokerSlot)),
		V::Hieroglyph => {
			events.push(Event::EaseAnte(-antes_back(extra)));
			round.base_hands -= extra;
			events.push(Event::EaseHands(-extra));
		}
		V::Petroglyph => {
			events.push(Event::EaseAnte(-antes_back(extra)));
			round.base_discards -= extra;
			events.push(Event::EaseDiscard(-extra));
		}
		// The game reads these from the redeemed vouchers where they apply.
		V::OmenGlobe | V::Telescope | V::Observatory | V::Blank | V::DirectorsCut | V::Retcon => {}
	}
}

/// The antes Hieroglyph and Petroglyph take back (their `extra`).
fn antes_back(extra: f64) -> i32 {
	i32::try_from(num::trunc(extra)).expect("a voucher lowers the ante by one")
}

/// The events of the shop, of vouchers taking effect and of `use_card`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ShopEvent {
	ShopOuter {
		shop_exists: bool,
	},
	ShopPopulate {
		shop_exists: bool,
	},
	StoreModify {
		card: u32,
	},
	RerollShopRun,
	/// `G.FUNCS.reroll_shop`'s 'after' event queueing the joker pass.
	RerollOuter,
	RerollShopJokers,
	BuyRun {
		card: u32,
		/// The hand cards to use it on at once, for a consumable.
		use_on: Option<CardMask>,
	},
	SellRun {
		card: u32,
		area: Area,
	},
	VerdantDisable,
	ToggleShopRun,
	UseFinishOuter {
		prev: State,
		area: Option<Area>,
		card: u32,
		dissolve: bool,
		delay_factor: f64,
	},
	UseFinishInner {
		prev: State,
		area: Option<Area>,
		delay_factor: f64,
	},
	ShopSize(f64),
	SetRate {
		which: Rate,
		v: f64,
	},
	/// `G.GAME.interest_cap`, which Seed Money and Money Tree raise.
	InterestCap(f64),
	ConsumableSlot,
	Discount(f64),
	RerollDiscount(f64),
	JokerSlot,
}

/// A `G.GAME` rate a voucher sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rate {
	Tarot,
	Planet,
	Edition,
	PlayingCard,
}

pub(crate) fn run_event(g: &mut Game, e: ShopEvent) {
	use ShopEvent as E;
	match e {
		E::ShopOuter { shop_exists } => {
			g.events.push_event(
				Event::Shop(E::ShopPopulate { shop_exists }),
				Trigger::After(0.2),
				Blocks::Barrier,
			);
		}
		E::ShopPopulate { shop_exists } => shop_populate(g, shop_exists),
		E::StoreModify { card } => {
			let c = shop_card_mut(&mut g.areas, card).expect(
				"the queue runs out before any action, so the new shop card is still there",
			);
			if let Some(c) = c.as_card_mut() {
				tags::modify_store_joker(&mut g.tags, &mut g.events, c);
			}
		}
		E::RerollShopRun => {
			let free = g.shop.free_rerolls > 0.0;
			g.shop.free_rerolls = (g.shop.free_rerolls - 1.0).max(0.0);
			if !free {
				g.shop.reroll_cost_increase += 1.0;
			}
			calculate_reroll_cost(&mut g.shop);
			while let Some(c) = g.areas.shop_jokers.pop() {
				card_lifecycle::unmark_used(&g.areas, &mut g.history, &c);
			}
			fill_shop_jokers(g);
		}
		E::RerollOuter => g.events.push(Event::Shop(E::RerollShopJokers)),
		E::RerollShopJokers => {
			calculate_joker::calc_jokers(g, LifecycleCtx::RerollShop);
		}
		E::BuyRun { card, use_on } => buy_run(g, card, use_on),
		E::SellRun { card, area } => sell_run(&g.areas, &g.blind, &mut g.events, card, area),
		E::VerdantDisable => blind::disable_blind(g),
		E::ToggleShopRun => close_shop(&mut g.flow, &mut g.areas, &mut g.history, &mut g.shop),
		E::UseFinishOuter {
			prev,
			area,
			card,
			dissolve,
			delay_factor,
		} => {
			if dissolve {
				g.events.start_dissolve(card);
			}
			g.events.push_after(
				0.1,
				Event::Shop(E::UseFinishInner {
					prev,
					area,
					delay_factor,
				}),
			);
		}
		E::UseFinishInner {
			prev,
			area,
			delay_factor,
		} => {
			g.flow.state = prev;
			// Using a held consumable inside a pack does not count as a pick.
			if prev.is_pack() && g.pack.open.is_some() && area != Some(Area::Consumables) {
				if let Some(choices) = g.pack.choices.filter(|&n| n > 1) {
					g.pack.choices = Some(choices - 1);
				} else {
					packs::end_consumeable(&mut g.events, delay_factor);
				}
			}
		}
		E::ShopSize(m) => change_shop_size(g, m),
		E::SetRate { which, v } => match which {
			Rate::Tarot => g.shop.tarot_rate = v,
			Rate::Planet => g.shop.planet_rate = v,
			Rate::Edition => g.shop.edition_rate = v,
			Rate::PlayingCard => g.shop.playing_card_rate = v,
		},
		E::InterestCap(v) => g.run.interest_cap = v,
		E::ConsumableSlot => g.areas.limits.consumables.limit += 1.0,
		E::Discount(d) => {
			g.shop.discount_percent = d;
			card_lifecycle::set_cost_all(&mut g.areas, g.shop.discount_percent);
		}
		E::RerollDiscount(d) => {
			g.shop.base_reroll_cost -= d;
			g.shop.reroll_cost = (g.shop.reroll_cost - d).max(0.0);
		}
		E::JokerSlot => g.areas.limits.jokers.limit += 1.0,
	}
}

/// The shop closing: its cards leave the game, on to blind select.
fn close_shop(flow: &mut Flow, areas: &mut Areas, history: &mut History, shop: &mut Shop) {
	shop.open = false;
	for c in &areas.shop_jokers {
		card_lifecycle::unmark_used(areas, history, c);
	}
	for c in areas.shop_vouchers.iter().chain(&areas.shop_boosters) {
		card_lifecycle::unmark_used(areas, history, c);
	}
	areas.shop_jokers.clear();
	areas.shop_vouchers.clear();
	areas.shop_boosters.clear();
	flow.state_complete = false;
	flow.state = State::BlindSelect;
}

/// The sale of the card `card` of `area`: its money, and Verdant Leaf
/// disabled by a sold joker.
fn sell_run(areas: &Areas, blind: &Blind, events: &mut EventQueue, card: u32, area: Area) {
	let i = areas
		.position(area, card)
		.expect("nothing removes a card between its sale and the sale event");
	let c = areas.card_at(area, i);
	events.push(Event::EaseDollars(c.core().sell_cost));
	events.start_dissolve(card);
	events.delay(0.3);
	let is_joker = c
		.as_card()
		.is_some_and(|c| matches!(c.kind, Kind::Joker(_)));
	if is_joker && blind.kind == Some(BlindKind::FinalLeaf) {
		events.push(Event::Shop(ShopEvent::VerdantDisable));
	}
}
