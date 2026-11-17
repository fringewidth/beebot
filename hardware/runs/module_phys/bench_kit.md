# Bench-test kit order list (2026-11-15)

Prepared for the 8-11 mm SMA-ratchet cube bench tests. Nothing was bought. `verified` = the number was read on the vendor page in this session (LCSC pages were read through their embedded price data; Dynalloy, JC Magnetics, MowerBot Pro, Images SI, DigiKey India, Hubtronics, KSP and JLCPCB pages were read by fetch). Everything else is a search snippet or an assumption and is labelled so. Exchange rate 88 INR per USD is an assumption.

## Order tonight

Lines marked A are assumptions (price not read). Totals are for the orderable lines only.

1. **LCSC** (MCU, dev boards and programmers, driver, FET, accelerometer, IR LED, photodiode): parts USD 39.45 verified, shipping to India not read, assume USD 25.
1. **Dynalloy quote email** for 25, 50, 76 um (and 38 um) Flexinol HT, plus **MowerBot Pro** 38 um 5 m USD 11 (verified). Wire parts about USD 150 (upper bound, A), shipping A 40.
1. **JC Magnetics** 1 x 2 mm and 2 x 1.5 mm N52 discs, about USD 30 (one line verified), shipping A 30.
1. **AliExpress** spheres 1.0, 1.5, 2.0 mm and 3 mm (not reachable from here, not verified): assume USD 60 for the four. Place this order first, it has the longest delivery.
1. **Mouser India** CeraCharge x10 (A, USD 60); do not wait for CPH3225A.
1. **Indian shops** (Zbotic LiPo, Engineer Store supercap, scale, microscope, tweezers): about INR 12,500 (A).
1. **Do not order tonight**: CPH3225A (DigiKey India, 0 stock, 18 weeks), the printer and resin (team likely has one; Saturn 4 Ultra is out of stock at KSP, INR 61,500).

Computed total over all order-tonight lines: parts USD 527.66, shipping allowance USD 95, customs allowance 30 percent: **USD 809.46, about INR 71,232**. Of this only the LCSC lines (about USD 40) and a handful of others are verified; the rest is an estimate with wide error (plus or minus 40 percent).

Longest delivery: AliExpress spheres, 15-30 days (A), and the Dynalloy quote path, 10-20 days after the quote; both land before 31 Dec. CPH3225A at DigiKey India (18 weeks) does not. JLCPCB flex PCB is not in tonight's list; a first order takes about 2-3 weeks door to door (A).

## Items by category

### SMA wire

| item | vendor | product / part | unit price | qty | line USD | India shipping | est days | stock | verified |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flexinol 90C 25 um (0.001 in), 5 m | [Dynalloy (quote)](https://dynalloy.com/flexinol-actuator-wire-price-guide/) | Flexinol 0.001 in, 90C (quote) | 14 USD | 5 | 70.0 | unknown (quote by email; DHL typical) | quote 1-3 d + 7-14 d | - | false |
| Flexinol 90C 38 um (0.0015 in) HT, 5 m | [MowerBot Pro](https://www.mowerbotpro.com/products/flexinol-wire-0-0015-ht/) | Flexinol Wire 0.0015 in HT, SKU UZWAX604644 | 11 USD | 1 | 11.0 | unknown (page silent; 3-day shipping claim) | 7-21 (A) | 55 in stock | true |
| Nitinol/Flexinol 90C 50 um (0.002 in), 5 m | [Dynalloy (quote)](https://dynalloy.com/flexinol-actuator-wire-price-guide/) | Flexinol 0.002 in, 90C (quote) | 10 USD | 5 | 50.0 | unknown (quote) | quote 1-3 d + 7-14 d | - | false |
| Flexinol 90C 76 um (0.003 in), 2 m | [Dynalloy (quote); Images SI as US retail](https://www.imagesco.com/nitinol/wire.html) | Flexinol 0.003 in HT | 8.2 USD | 2 | 16.4 | Images SI mentions international shipping | 10-20 (A) | - | true |

1. Flexinol 90C 25 um (0.001 in), 5 m: Price guide (read): 0.001 in is USD 14-7 per m up to 100 m; 14 used as worst case, so line is an upper bound. 90C availability of 0.001 in not stated on the page; ask in the quote. Backup: Kellogg's Research Labs (quote only, ultra-fine from 0.015 mm, kelloggsresearchlabs.com; page read, no prices).
1. Flexinol 90C 38 um (0.0015 in) HT, 5 m: Pack is 5 m; USD 11.00 (was 21.50). Shipping destination not stated; confirm at checkout. Also on Amazon US (5 ft only). Backup: Dynalloy quote.
1. Nitinol/Flexinol 90C 50 um (0.002 in), 5 m: Guide band for 0.001-0.003 in is USD 12-6 per m (table 1), 7-6 for 0.003-0.004 in (table 2); 10 is an assumption (A). Robotshop lists 0.002 in LT (70C) in 5 m packs (ca.robotshop.com/products/dynalloy-flexinol-002-lt-actuator-wire, price not read, 90C HT not found); LT is a poor substitute.
1. Flexinol 90C 76 um (0.003 in), 2 m: Images SI page read: USD 2.50 per ft for 0.003-0.010 in (about 8.2 per m), LT or HT, cut to length. Price verified; HT stock and India shipping cost not shown. Robotshop Deluxe Flexinol Kit includes 1 m of 0.003 in HT plus eyelets (search snippet only).

### SMA crimps

| item | vendor | product / part | unit price | qty | line USD | India shipping | est days | stock | verified |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Crimps for 25-76 um SMA wire (100) | [Dynalloy / Robotshop kit / local brass tube](https://www.robotshop.com/products/dynalloy-flexinol-deluxe-kit) | Dynalloy Flexinol Deluxe Kit (20 eyelets/splices + 1 m 0.003 in HT) | 30 USD | 1 | 30.0 | Robotshop ships worldwide (A); page returned 403 | 10-20 (A) | - | false |

1. Crimps for 25-76 um SMA wire (100): Price is an assumption. Images SI does not sell crimps stand-alone (precrimp service, USD 125 setup). No stand-alone 0.3-0.5 mm ID crimp found at a fetchable page. Practical fallback: brass or copper tube 0.5 mm OD / 0.3 mm ID from a local hobby shop or a jeweller (cut to 1.5 mm); the team has to cut these.

### Magnets

| item | vendor | product / part | unit price | qty | line USD | India shipping | est days | stock | verified |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N52 Ni sphere 1.0 mm (100) | AliExpress (not reachable from this session) | not found | 15 USD | 1 | 15.0 | yes (A) | 15-30 (A) | - | false |
| N52 Ni sphere 1.5 mm (100) | AliExpress (not reachable) | not found | 15 USD | 1 | 15.0 | yes (A) | 15-30 (A) | - | false |
| N52 Ni sphere 2.0 mm (100) | AliExpress (not reachable) | not found | 15 USD | 1 | 15.0 | yes (A) | 15-30 (A) | - | false |
| N52 Ni sphere 3 mm (50) | [The Magnet Baron (US)](https://themagnetbaron.com/products/50pcs-3mm-1-8-sphere-magnets-n52) | 50pcs 3mm (1/8 in) Sphere Magnets N52 | 15 USD | 1 | 15.0 | unknown (US seller; checkout) | 10-20 (A) | reported sold out (search snippet) | false |
| N52 Ni disc 1 x 2 mm (100) [substitute for 1 x 1] | [JC Magnetics](https://www.jc-magnetics.com/Magnet-N52-1mmx2mm-discs-cylinders) | 100pcs 1mm x 2mm N52 Discs Cylinder, SKU JCN52-DSC1-2 | 15 USD | 1 | 15.0 | not stated on page | 10-20 (A) | In Stock | true |
| N52 Ni disc 2 x 1.5 mm (100) [substitute for 2 x 1] | [JC Magnetics](https://www.jc-magnetics.com/Magnet-N52-2mmx1_5mm-discs) | Magnet N52 2mmx1.5mm discs | 14.5 USD | 1 | 14.5 | not stated | 10-20 (A) | - | false |

1. N52 Ni sphere 1.0 mm (100): No vendor page for a 1.0 mm N52 sphere was found; AliExpress refused connection from this environment. Price is an assumption. Fallback: 1 x 1 mm cylinders (JC Magnetics, below).
1. N52 Ni sphere 1.5 mm (100): As above.
1. N52 Ni sphere 2.0 mm (100): As above.
1. N52 Ni sphere 3 mm (50): Page returned 429 here; price assumed. Second source: supermagnete K-03-C (403 here). Indian seller for 3 mm spheres not found (Probots sells 3x3 mm discs, INR 24).
1. N52 Ni disc 1 x 2 mm (100) [substitute for 1 x 1]: USD 15 for 1 pack, 13 for 2+. Ni-Cu-Ni 15-25 um, axial. A 1 x 1 mm disc was not found at a fetchable page; JC Magnetics lists 1x1.5 and 2x1.5 mm in search results (not read).
1. N52 Ni disc 2 x 1.5 mm (100) [substitute for 2 x 1]: Price from search snippet, page not read.

### Storage

| item | vendor | product / part | unit price | qty | line USD | India shipping | est days | stock | verified |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TDK CeraCharge 1812 (BCT1812, B73180A0101M062), 100 uAh | [Mouser India](https://www.mouser.in/ProductDetail/TDK/B73180A0101M062) | B73180A0101M062 (A: part number from memory of the family; the page fetch timed out) | 6 USD | 10 | 60.0 | yes, Mouser India ships to India (A) | 5-10 (A) | - | false |
| Seiko CPH3225A, 11 mF | [DigiKey India](https://www.digikey.in/en/products/detail/seiko-instruments/CPH3225A/8692444) | CPH3225A (cut tape) | 171.008 INR | 10 | 19.43 | yes | not before 14 Dec; manufacturer lead 18 weeks | 0 in stock, 10,000 expected 2026-12-14 | true |
| LiPo, nearest to 301010 (301020 45 mAh) | [Zbotic](https://zbotic.in/product/kp-original-301020-45mah-3-7v-single-cell-rechargeable-lipo-battery/) | KP Original 301020 45mAh 3.7V LiPo | 101.48 INR | 5 | 5.77 | yes, free shipping (search snippet) | 3-7 (A) | - | false |
| Supercap 0.47 F 5.5 V (coin/SMD) x10 | [The Engineer Store / Makerbazar / Robu](https://theengineerstore.in/hi/products/0-47-f-5-5-v-super-capacitor) | 0.47 F, 5.5 V super capacitor | 89 INR | 10 | 10.11 | yes (Indian sellers) | 3-7 (A) | - | false |

1. TDK CeraCharge 1812 (BCT1812, B73180A0101M062), 100 uAh: Unverified: Mouser page timed out, DigiKey/LCSC listing not found. Price assumed. Check stock before paying; CeraCharge has been on long lead times.
1. Seiko CPH3225A, 11 mF: INR 249.19 x1, 171.01 x10, 140.10 x50. Out of stock: DO NOT order tonight; fails the New Year bar. Use a 0.1-0.5 F supercap instead and look for CPH3225A on LCSC or AliExpress (no listing found).
1. LiPo, nearest to 301010 (301020 45 mAh): Page returned 403; INR 101.48 incl GST per snippet. No 301010 / 401012 (10 mAh) listing found in India. Zbotic also lists 401020 160 mAh, 451313 40 mAh.
1. Supercap 0.47 F 5.5 V (coin/SMD) x10: INR 89 per search snippets; Robu's own 0.47 F page is out of stock (restock 25-30 days). Pages not read.

### MCU

| item | vendor | product / part | unit price | qty | line USD | India shipping | est days | stock | verified |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CH32V003F4U6 QFN-20 | [LCSC](https://www.lcsc.com/product-detail/C5299908.html) | WCH CH32V003F4U6, LCSC C5299908 | 0.3079 USD | 20 | 6.16 | yes, LCSC ships to India (A) | 5-10 (A) | 4990 | true |
| CH32V003 dev board x2 + WCH-LinkE (bundle, 2 sets) | [LCSC](https://www.lcsc.com/product-detail/C5236707.html) | CH32V003F4P6-EVT+WCH-LinkE, LCSC C5236707 | 7.2565 USD | 2 | 14.51 | yes (A) | 5-10 (A) | 2 (exactly enough, fragile) | true |

1. CH32V003F4U6 QFN-20: Price ladder: 5=0.3079, 50=0.2478, 150=0.2220. 20 pcs at the 5-tier.
1. CH32V003 dev board x2 + WCH-LinkE (bundle, 2 sets): Each set = EVT dev board + WCH-LinkE; 10 = 6.2357. Two sets give 2 dev boards and 2 programmers. Backups: Hubtronics WCH-LinkE-R0-1v3 INR 550 + 18% GST (page read, out of stock); Evelta CAPUF-LinkE INR 708 incl GST (search snippet).

### Electronics

| item | vendor | product / part | unit price | qty | line USD | India shipping | est days | stock | verified |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DRV8837DSGR (WSON-8 2x2) x20 | [LCSC](https://www.lcsc.com/product-detail/C39159.html) | TI DRV8837DSGR, LCSC C39159 | 0.2384 USD | 20 | 4.77 | yes (A) | 5-10 (A) | 115410 | true |
| DMG1012T-7 (SOT-523) x20 | [LCSC](https://www.lcsc.com/product-detail/C20512.html) | DIODES DMG1012T-7, LCSC C20512 | 0.0905 USD | 20 | 1.81 | yes (A) | 5-10 (A) | 36520 | true |
| LIS2DW12TR accelerometer x5 | [LCSC](https://www.lcsc.com/product-detail/C189624.html) | ST LIS2DW12TR, LCSC C189624 | 0.9119 USD | 5 | 4.56 | yes (A) | 5-10 (A) | 15045 | true |
| IR LED 0402-class (1005 metric) x20 | [LCSC](https://www.lcsc.com/product-detail/C5382142.html) | NATIONSTAR NCD1005I1, LCSC C5382142 | 0.2807 USD | 20 | 5.61 | yes (A) | 5-10 (A) | 2996 | true |
| Photodiode 940 nm, smallest in stock (2x3 mm) x20 | [LCSC](https://www.lcsc.com/product-detail/C6081357.html) | NEWOPTO XYC-PD3020F90-S3, LCSC C6081357 | 0.1014 USD | 20 | 2.03 | yes (A) | 5-10 (A) | 29430 | true |

1. DRV8837DSGR (WSON-8 2x2) x20: Ladder: 5=0.2384, 50=0.1821. Search results flag NRND (not read on TI).
1. DMG1012T-7 (SOT-523) x20: 20 V, 0.63 A. Cheaper clone C2830301 (Tech Public DMG1012T, 0.0295 at 20, stock 57200) also read. Note 0.63 A rating vs 0.25-0.5 A SMA pulse: margin is thin.
1. LIS2DW12TR accelerometer x5: Ladder: 1=0.9119, 10=0.7930.
1. IR LED 0402-class (1005 metric) x20: Ladder: 1=0.3568, 10=0.2807, 100=0.2073. Package and wavelength taken from the LCSC title/search, confirm on datasheet. Everlight IR16-213C (C2943465), Inolux IN-S42CTQHIR (C6879327) and Kingbright APHHS1005F3C (C5360595) are 0402 940 nm but show stock 0 on LCSC today. Meihua MHT130ZVRCT (C2962093) stock 1420, 0.5388 at 10.
1. Photodiode 940 nm, smallest in stock (2x3 mm) x20: No 0402 photodiode found at LCSC. Smallest in stock found is 2x3 mm: too big for the cube, fine for the bench. Alternatives read: Vishay VEMD2000X01 2.3 mm (C4353739, stock 308, 0.4980 at 10), Everlight PD70-01B (C72056, 4.4x3.9 mm). Alternative idea: use the IR LED itself reverse-biased as a detector.

### Tools

| item | vendor | product / part | unit price | qty | line USD | India shipping | est days | stock | verified |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Jeweller's scale 0.001 g, 200 g | [IndiaMART seller (Delhi)](https://m.indiamart.com/impcat/precision-digital-scale.html) | Precision Digital Scale 210 g x 0.001 g | 10500 INR | 1 | 119.32 | yes (Indian) | 3-7 (A) | - | false |
| Pocket scale 0.01 g (magnet pull tests, as a force gauge) | [Amazon.in / iScale India](https://iscaleindia.com/products/baijnath-premnath-kes-professional-pocket-scale-200g-x-0-01g-precision-ideal-for-jewelry-and-lab-use-golden-edition) | KES pocket scale 200 g x 0.01 g | 299 INR | 1 | 3.4 | yes | 2-5 (A) | - | false |
| Digital force gauge | not sourced | not found | 0 USD | 0 | 0.0 | - | - | - | false |
| USB digital microscope 1000x (budget) | [iBELL / amiciKart / Moglix](https://www.amicikart.com/products/amicicare-1000x-usb-digital-microscope-with-8-led-adjustable-brightness-and-0-3m-hd-camera-compatible-with-windows-and-android-wired) | amiciCare 1000X USB microscope | 1499 INR | 1 | 17.03 | yes | 3-7 (A) | - | false |
| Fine tweezers set (ESD, Vetus-style) | local / Robu | not researched | 500 INR | 1 | 5.68 | yes | - | - | false |
| SMD hot plate / reflow | Indian sellers | not researched | 2500 INR | 1 | 28.41 | yes | - | - | false |

1. Jeweller's scale 0.001 g, 200 g: Search snippet only. Worth buying locally instead; a 0.001 g scale below INR 10k is rare.
1. Pocket scale 0.01 g (magnet pull tests, as a force gauge): Price from tracker snippet (lowest INR 299, list 699). Pull test method: magnet on the scale pan, pull with a thread from above.
1. Digital force gauge: No gauge priced; the 0.01 g scale covers 0.1 mN steps. Dropped from the total.
1. USB digital microscope 1000x (budget): Snippet prices INR 1,488-2,155. 1000x claims are optical nonsense; expect about 20-50x usable. For wire threading a real stereo zoom microscope is better, not priced here.
1. Fine tweezers set (ESD, Vetus-style): Assumed INR 500.
1. SMD hot plate / reflow: Assumed INR 2,500; JLCPCB PCBA makes this optional.

### Printer

| item | vendor | product / part | unit price | qty | line USD | India shipping | est days | stock | verified |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MSLA printer: Elegoo Saturn 4 Ultra 16K | [KSP Electronics (out of stock); Electropi](https://kspelectronics.in/product/elegoo-saturn-4-ultra-16k-3d-printer/) | ELEGOO Saturn 4 Ultra 16K | 61500 INR | 1 | 698.86 | yes, pan-India | 3-7 (A) | out of stock at KSP; Ideal3D sold out; Electropi INR 57,810 + 18% GST (snippet) | true |
| Tough resin 1 kg (Siraya Fast ABS-like) | [Amazon.in (tracker)](https://price-history.in/product/siraya-tech-fast-abs-like-3d-xXtFnCyp) | Siraya Tech Fast ABS-Like Tough 1 kg | 7800 INR | 1 | 88.64 | yes | 3-7 (A) | - | false |

1. MSLA printer: Elegoo Saturn 4 Ultra 16K: 14 x 19 um XY on page (4.2-4.7 pix at 0.2 mm tooth pitch is 10-14 pixels, fine). Mars 5 Ultra, Anycubic M5/M7, Phrozen Sonic Mini 8K S not priced in India (no pages read). Team may already own a printer (concepts.md says 'the team's printer'): then drop this line.
1. Tough resin 1 kg (Siraya Fast ABS-like): Tracker snippets: INR 7,226-8,140. Siraya Blu not found in India (AUD 149 in Australia). Cheaper: SunLu ABS-like at Robu (price not seen), Elegoo ABS-like not found.

### PCB services

1. JLCPCB flex (verified, jlcpcb.com/capabilities/flex-pcb-capabilities): min trace/space 3.5/3.5 mil at 0.5 oz and 4/4 mil at 1 oz; min hole 0.1 mm; no minimum board size listed (panelise under 20 x 20 mm); 1, 2 or 4 layers; 2-layer 0.11 mm on 25 um polyimide; no rigid-flex; SMT assembly with stiffeners. 0201 assembly: not confirmed (assembly capability page returned 404; JLC is believed to place 0201, A).
1. India shipping (secondary source, Zbotic comparison, not verified): DHL about INR 700 and 4-6 days; economy about INR 250 and 10-15 days.
1. Customs (secondary): about 10% BCD plus 18% IGST on PCBs; the claim that parcels under INR 5,000 pass duty free is unverified. Magnets (HS 8505) and ICs (HS 8542) likely similar; the courier collects duty before delivery.
1. PCBWay: not checked. Alternative for small flex runs.

### Didel gears

1. Didel (didel.com): home page has no shop; Gears.pdf/Microgears.pdf (search results) list module 0.2 POM pinions (9T, 12T), spur gears and worms mod 0.2 and 0.3, with a 2008 price list (CHF 2.60-2.90). Stock and India shipping unknown; email info@didel.com. Unverified.

## Gaps

1. N52 spheres 1.0 / 1.5 / 2.0 mm: no vendor page found at all. AliExpress refused connection. Fallback is 1 x 1 mm cylinders.
1. 0402 photodiode: none at LCSC; 2 x 3 mm part chosen for the bench.
1. LiPo 301010 / 401012 (10 mAh): not stocked in India; 301020 45 mAh is the nearest.
1. SMA crimps: no stand-alone vendor page; use brass tube from a hobby shop.
1. Robu, Robocraze, Evelta, Supermagnete, Robotshop and Mouser India pages mostly returned 403, 404, 429 or timeouts here.
