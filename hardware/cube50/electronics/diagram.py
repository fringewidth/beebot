"""Draw the Cube50 wiring diagram (blocks, power path, signals) to electronics/wiring.png.
Run from the repo root: .venv/bin/python hardware/cube50/electronics/diagram.py
The netlist in netlist.csv is the authority; this picture is its summary.
Changed 2026-11-17 11:24 (+0530): DRV8833 #3 and the two hinge deploy (sled) motors with their four
end switches, driven from the MCP23017 (now 16 of 16 pins).
Changed 2026-11-17 (60 mm redesign): disc motors are Pololu #3079 with #4761 quadrature encoders (PCNT inputs on
GPIO4, 5, 33, 26), the disc AS5600s are gone, two disc-lock SMA wires (GPIO45, 46) make 8 SMA channels, the disc
sense resistors are 0.215 R (0.93 A chop), the hinge AS5600s read the axle end 1:1."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "wiring.png"
W, H = 1980, 1150


def font(size):
    for f in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


F, FS, FT = font(19), font(15), font(26)
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
POWER, SIGNAL, SMA = (200, 40, 40), (40, 80, 180), (150, 90, 20)


def block(x, y, w, h, title, lines=(), fill=(244, 244, 240)):
    d.rounded_rectangle((x, y, x + w, y + h), 10, fill=fill, outline=(60, 60, 60), width=2)
    d.text((x + 10, y + 8), title, fill=(0, 0, 0), font=F)
    for k, t in enumerate(lines):
        d.text((x + 10, y + 36 + 20 * k), t, fill=(50, 50, 50), font=FS)
    return (x, y, w, h)


def wire(points, colour, label=None, at=0, width=3):
    d.line(points, fill=colour, width=width, joint="curve")
    if label:
        x, y = points[at]
        d.text((x + 6, y - 20), label, fill=colour, font=FS)


d.text((20, 15), "Cube50 wiring: main board (core), two foot sides (top shown, bottom the same), power path in red, signals in blue, SMA in brown", fill=(0, 0, 0), font=FT)

# power path
block(20, 80, 250, 110, "Cell 1S LiPo 302535", ["about 220 mAh, high-rate 2.5 A", "10k NTC taped on"])
block(330, 80, 270, 110, "Protection", ["DW01A + FS8205A", "cut-off 2.4 V / 4.3 V, 3 A"])
block(660, 60, 270, 130, "USB-C (-x cover, y 17 z 11.5)", ["5 V in -> TP4056 charger 195 mA", "D+/D- -> ESP32-S3 native USB", "CC: 2 x 5.1k to GND"])
block(330, 230, 270, 110, "3.3 V LDO AP2112K", ["VBAT -> 3V3, 600 mA", "ESP32, IMU, MCP, AS5600, encoders"])
block(660, 230, 270, 110, "6 V boost TPS61088", ["VBAT -> VM 6.0 V, 2 A out", "EN = DRV_NSLEEP (GPIO21)"])
wire([(270, 175), (330, 175)], POWER)
wire([(600, 135), (640, 135), (640, 285), (600, 285)], POWER, "VBAT", at=1)
wire([(640, 285), (660, 285)], POWER)
wire([(795, 190), (795, 230)], POWER)
wire([(660, 170), (620, 170), (620, 205), (465, 205), (465, 190)], POWER, "VBUS 5 V (charge)", at=3)

# MCU
block(40, 400, 420, 340, "ESP32-S3-MINI-1-N8", [
    "GPIO1 VBAT/2      GPIO2,3 hinge AS5600 OUT x2",
    "GPIO4,5,33,26 disc quadrature A/B x2 (PCNT)",
    "GPIO6-9 DRV8833 xISEN (via 1k/1uF)",
    "GPIO10 cell NTC",
    "GPIO11-18 motor IN1/IN2 x4 (MCPWM)",
    "GPIO21 DRV nSLEEP x3 + boost EN",
    "GPIO47 DRV nFAULT (wired-OR)",
    "GPIO38 SDA, GPIO39 SCL (400 kHz)",
    "GPIO40,41,42,35 ring SMA (LEDC)",
    "GPIO36,37 stow-bolt SMA, GPIO45,46 disc-lock SMA",
    "GPIO48 IMU INT1, GPIO34 LED",
    "GPIO19/20 USB D-/D+",
    "radio: ESP-NOW ch 1 to the gateway",
])
wire([(465, 340), (465, 380), (250, 380), (250, 400)], POWER, "3V3", at=1)

block(560, 400, 260, 110, "LSM6DS3TR-C (0x6A)", ["6-axis IMU, 416 Hz", "on the main board"])
block(560, 530, 260, 130, "MCP23017 (0x20), 16 of 16", ["in x12: ring end x4, seated x2,", "bolt x2, sled out/in x4", "out x4: deploy IN (GPA7, GPB5-7)"])
wire([(460, 450), (560, 450)], SIGNAL, "I2C")
wire([(510, 450), (510, 585), (560, 585)], SIGNAL)

block(900, 400, 300, 150, "DRV8833 #1 (hinges)", ["VM = 6 V, AISEN 0.33R, BISEN 0.33R", "A: top hinge N20 1000:1", "B: bottom hinge N20 1000:1", "chop limit 0.6 A"])
block(900, 570, 300, 150, "DRV8833 #2 (discs)", ["VM = 6 V, xISEN 0.215R (2 x 0.43R)", "A: top disc Pololu 297.92:1 (+14.29:1)", "B: bottom disc, same", "chop limit 0.93 A, firmware 0.94 A cap"])
wire([(930, 340), (930, 400)], POWER, "VM 6 V")
wire([(1215, 340), (1215, 645), (1200, 645)], POWER)
wire([(930, 340), (1215, 340)], POWER)
wire([(460, 520), (880, 520), (880, 475), (900, 475)], SIGNAL, "motor IN x8, ISEN x4", at=0)
wire([(880, 520), (880, 645), (900, 645)], SIGNAL)

block(1290, 400, 240, 110, "Hinge motors x2", ["in the core, on the hinge train", "AS5600 on the axle end, 1:1"])
block(1290, 570, 240, 110, "Disc motors x2", ["Pololu #3079 + #4761 encoder", "51,087 counts per disc turn", "6-pin SH: motor+A/B+3V3/GND"])

# hinge deploy (sled) drive
block(1650, 400, 310, 150, "DRV8833 #3 (deploy)", ["VM = 6 V, xISEN 0.43R (not read)", "A: top sled motor, B: bottom", "on/off from the MCP23017", "chop limit 0.47 A (motor stall)"])
block(1650, 570, 310, 150, "Deploy motors x2", ["N20 1:150, M3 lead screw", "brass nut in each hinge sled", "7.9 mm stroke, self-locking", "switches: sled out, sled in"])
wire([(1215, 340), (1805, 340), (1805, 400)], POWER)
wire([(1805, 550), (1805, 570)], POWER)
wire([(740, 660), (740, 776), (1630, 776), (1630, 475), (1650, 475)], SIGNAL)
wire([(1630, 690), (1650, 690)], SIGNAL)
d.text((1236, 712), "MCP: deploy IN x4 out,", fill=SIGNAL, font=FS)
d.text((1236, 730), "sled switches x4 in", fill=SIGNAL, font=FS)
wire([(1200, 455), (1290, 455)], POWER)
wire([(1200, 625), (1290, 625)], POWER)
wire([(1530, 455), (1600, 455), (1600, 760), (300, 760), (300, 730)], SIGNAL, "AS5600 OUT x2 (analog) + 3V3/GND", at=2)
wire([(1530, 625), (1600, 625)], SIGNAL)
d.text((1296, 690), "encoder A/B -> GPIO4,5,33,26", fill=SIGNAL, font=FS)

# SMA
block(40, 820, 420, 150, "SMA drivers on the main board", ["8 x AO3400A low side, gate 100R + 100k", "high side: VBAT direct (cell protected)", "0.55 W per wire, phase-staggered, max 3 on", "47 uF on VBAT at the bank (guess)"])
wire([(250, 730), (250, 820)], SMA, "LEDC PWM x8", at=0)
block(560, 820, 300, 150, "Stow bolts x2", ["SMA 0.15 mm x 50 mm each", "spring returns the bolt", "switch: bolt up"])
wire([(460, 895), (560, 895)], SMA)
block(960, 790, 380, 200, "Foot (top shown)", ["FFC 6-way across the hinge, looped round the knuckle:", "VBAT, EXT, RET, SW_EXT, SW_RET, LOCK",
                                                  "foot board -> 5 sprung contacts", "-> 5 ring tracks under the disc plate",
                                                  "disc: 2 SMA (extend / retract), bistable ring,", "2 end switches to VBAT (47k/100k divider in body)"])
wire([(460, 940), (960, 940)], SMA, "EXT_LOW, RET_LOW, LOCK_LOW per foot (FFC)", at=0)
block(560, 990, 300, 100, "Disc locks x2 (in the feet)", ["SMA 0.15 mm x 52 mm each, 2.9 ohm", "spring pin engaged by default", "no switch: stop when disc moved > 5 deg"])
wire([(460, 955), (500, 955), (500, 1040), (560, 1040)], SMA)
wire([(1340, 890), (1600, 890), (1600, 600), (820, 600)], SIGNAL, "SW_EXT, SW_RET -> MCP23017", at=1)

block(1420, 80, 360, 230, "Gateway (host side)", ["ESP32-S3 dev board on the host's USB", "same firmware, -DGATEWAY=1", "serial: TX <cube> <type> <hex>", "ESP-NOW broadcast, channel 1", "telemetry 20 Hz per cube back", "", "camera tracking runs on the host"])
wire([(1420, 200), (1250, 200), (1250, 380), (460, 380), (460, 400)], SIGNAL, "2.4 GHz ESP-NOW", at=1, width=2)

d.text((20, H - 40), "Not shown: decoupling, pull-ups (I2C 4.7k, nFAULT 10k), VBAT divider 100k/100k, NTC divider 10k/10k, ESD on USB. Spec gaps: no power or data through faces, no charging through faces (USB-C only). Spare GPIO: 0, 43, 44.", fill=(60, 60, 60), font=FS)
img.save(OUT)
print("wrote", OUT)
