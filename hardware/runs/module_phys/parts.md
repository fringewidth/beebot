# Parts catalogue summary (2026-11-15)

Caveat: web fetching was weak this session (Pololu category page, Dynalloy page 404, LCSC page without price). Only CH32V003F4U6 specs (QFN-20 3x3, 16 KB, LCSC C5299908) were read on a vendor page. Everything else is from memory or search snippets and marked `verified: false` in parts.json. AliExpress specs and prices are typical ranges; order a few variants and measure.

## Micro gearmotors
1. 6 mm planetary (0615), approx 6x20 mm, 1.5 g, 3 V, output torque approx 5-50 mN m depending on ratio, approx 3 USD. Catch: spec sheets on listings are inconsistent.
1. 4 mm planetary (0408), approx 4x18 mm, 0.8 g, approx 3.5 USD. Catch: very low torque, fragile gearbox.
1. 8 mm planetary, approx 8x24 mm, approx 3.5 USD. More torque, long.
1. N20 (10x12x26 mm, 9 g, 30-200 mN m at 1:100-1:300, approx 2.2 USD). Robust fallback, stocked at Robu.in; too large for sub-20 mm cubes.
1. Micro worm gearmotor: self-locking, offset shaft. 1.7-2 g servos: only 180 degree travel, 8x12x22 mm.

## Vibration and coreless motors
1. Coin vibration 8-10 mm x 2.7-3.4, 0.7-0.9 g, approx 0.4 USD, 60-80 mA. Cheapest locomotion actuator (stick-slip). Catch: uncontrolled direction.
1. 4 mm cylindrical vibration and 0408 / 0615 coreless: raw motors at 30k rpm, 400-600 mA stall; need custom reduction.

## SMA wire
1. Flexinol 0.025 mm (approx 45 g force, 45 mA), 0.05 mm (approx 150 g, 200 mA), 0.1 mm (approx 600 g, 1 A). Catch: cooling time, current spikes, tiny strokes (4 percent), shipping to India uncertain. Nitinol from AliExpress is cheaper but inconsistent.

## Magnets
1. N52 discs 3x1 to 6x2 mm and cubes 2-3 mm: approx 0.04-0.10 USD each (approx 4-10 USD per 100 on AliExpress), pull approx 0.4-3 N.
1. Diametrically magnetised 4x2 / 6x3 mm: approx 0.4-0.6 USD each. Catch: confirm magnetisation axis; brittle.

## Energy storage
1. LiPo 301010 (10 mAh), 401012 (15 mAh), 401020 (20-30 mAh), 501225 (40-60 mAh): 2-2.5 USD, 1C-2C so 10-100 mA only. Catch: cannot run motors.
1. High-rate (5C-10C) drone micro cells, 50-100 mAh, approx 6x14x22 mm: only class that sustains 250-500 mA.
1. LIR1220/1254/2032: 8-45 mAh, peak under 80 mA. Silver oxide: primary, 1-3 mA, useless for motors.
1. Supercaps: CPH3225A 11 mF 3.2x2.5x0.9 mm (MCU bursts only); 0.1 F coin high ESR; 1 F can too big.

## MCU and radio
1. CH32V003F4U6, QFN-20 3x3 mm, approx 0.1-0.2 USD (price unverified), LCSC C5299908. Best smallest; no radio.
1. CH32V003J4M6 SOP-8, STM32C011 TSSOP-20, ATtiny402 SOIC-8.
1. ESP32-C3FH4 QFN 5x5, nRF52 modules approx 13x18 mm. Catch: radio peaks approx 350 mA on ESP32; weak cell cannot supply it. IR LED + phototransistor 0603 for neighbour comms instead.

## Drivers, charging, sensors
1. DRV8837 (WSON-8 2x2 mm, approx 0.6-0.8 USD): smallest H-bridge. DRV8210 similar, unverified. Discrete SOT-23 FETs cost 0.03 USD for one-way drive and SMA.
1. TP4054 (SOT-23-5, 0.08 USD), MCP73831 (0.7 USD, set 15 mA for tiny cells).
1. Hall DRV5032, LIS2DW12 (2x2 mm LGA), LSM6DS3TR-C.

## Contacts
1. Mini pogo 1.5 mm pitch x 2 mm, approx 0.1 USD; SMD spring clips; magnetic pogo 6x3x2 mm approx 1.2 USD (bulky).

## Fabrication
1. JLCPCB rigid PCB 0.4-0.8 mm: approx 2 USD per 5 plus 15-25 USD shipping and Indian duty; PCBA setup approx 8 USD plus extended-part fees; flex 5 pcs minimum. All from memory.
1. JLC3DP resin (min wall approx 0.8 mm) or MJF nylon for a 15 mm shell: approx 1.5-2 USD each, get instant quote.
1. Robu.in / Robocraze: local N20, magnets, LiPo, servos, 3-7 day delivery.

## Bearings
1. MR52 / 682 (2x5x2.5), MR62 (2x6), steel pins 1-2 mm, silicone O-rings 5-8 mm: 0.03-0.25 USD each.

## Gaps
1. No verified India-shipping data for Flexinol (Dynalloy page failed), Seiko CPH3225A, TI parts at DigiKey.
1. No bare nRF52 WLCSP that hobby PCBA can place; only modules.

## Orbray (formerly Namiki) micro motors
Read on the Orbray series page this session; no price, mass, minimum order or lead time published, so all entries are inquiry only. The Sales Partners page lists China, Taiwan, Korea, US reps and Europe, not India; no Digi-Key or Mouser listing found. Gearheads are sold only with a motor.
1. BMN04-0829P337: 4 mm x 18.4 mm brushless, 337:1, 98 rpm, 5.44 mN m stall, 3 V. P79: 420 rpm, 1.49 mN m. P18: 1790 rpm, 0.41 mN m. Needs a brushless driver.
1. CMS07-1707 geared (7 mm brushed, 24.9-31.6 mm long, 4.7/22/105/494:1): P4.7 gives 4040 rpm and approx 4 mN m.
1. BMN02-0593 (2.4 mm) stall 0.01 mN m; 1.5 mm geared and 0.6 mm brushless have no published specs; locking-system series not read.
1. Catch: likely tens to hundreds of USD each, low volume, import from Japan; not a fit for the 10 USD target.
