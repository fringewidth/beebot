# Cube50 radio protocol (version 3)

**Changed 2026-11-17 (60 mm redesign), version 2 to 3:** the discs turn through the quarter-turn lock
sequence. DISC now snaps its target to a quarter turn and answers twice (started, then the result with a
DiscResult code in `detail`); TELEMETRY appends `lock_result[2]` and `disc_counts[2]` and its faults byte gains
bit 7 (a disc lock did not hold); HELLO appends the protocol version as a ninth byte; a cube ignores every
frame whose `version` is not 3, except PING (it still answers HELLO, so a host can read the cube's version).
Disc speeds are lower: free speed at the disc is 0.80 rad/s, and a quarter turn takes about 4 s with release,
seat and verify. No message was removed or renumbered, and no existing field moved: a version 2 host reading
a version 3 telemetry frame sees the same fields at the same offsets up to `sled_fault`, then extra bytes.

**Changed 2026-11-17 11:24 (+0530), version 1 to 2:** hinge deploy drive. New DEPLOY message, the PIVOT
sequence deploys and retracts the hinge sled, TELEMETRY's switch word is 16 bits and carries the sled
states. A version 1 host misreads version 2 telemetry: check `version` in the header.

Transport: ESP-NOW, channel 1, one frame per packet (at most 250 bytes), no encryption. The host
talks through a gateway (an ESP32-S3 on USB, `pio run -e gateway`). Structs are packed and
little-endian; the authority is `include/protocol.h`.

## Frame

| field | type | meaning |
| --- | --- | --- |
| magic | u8 | 0xC5 |
| version | u8 | 3 |
| type | u8 | message type below |
| cube | u8 | host to cube: destination id (0xFF = every cube); cube to host: source id |
| seq | u16 | host sequence number; a cube echoes it in ACK |
| start_us | u32 | 0 = act on receipt; otherwise start when the synced clock's low 32 bits reach this (spec 7.3: timed starts across a robot) |

The cube id is the low byte of the cube's Wi-Fi MAC (printed in HELLO). Check a new batch for clashes.

## Host to cube

| type | name | payload | effect |
| --- | --- | --- | --- |
| 0x01 | PING | none | cube answers HELLO (firmware version, MAC) |
| 0x03 | STOP | none | every axis holds where it is, every SMA wire off, any pivot sequence abandoned |
| 0x04 | SYNC | u64 host_us | cube stores the offset to the host clock |
| 0x10 | HINGE | u8 foot, i16 target (0.01 deg, 0 = stowed, + = open), u16 rate (0.01 deg/s), u16 limit (mA) | one hinge move (bench or expert use; the pivot macro is safer); refused unless that hinge's sled is out (a move that stays at 0 is allowed) |
| 0x11 | DISC | u8 foot, i32 target (0.01 deg), u16 rate, u16 limit (mA), u8 relative | turns that foot's disc to the nearest quarter turn (multiple of 9000) to the target (plus the present angle if `relative`), through the lock sequence below; refused unless that foot is stowed, **both** sleds are in, no other disc is turning and the lock wire has cooled. **Two ACKs:** OK when it starts, then ARRIVED, RETURNED (stalled, drove back to the start quarter) or FAULT when it ends, with a DiscResult in `detail`. A target on the disc's present quarter answers both at once, without moving. Rate above about 4600 (0.01 deg/s, 46 deg/s, free speed) is pointless; the cube takes `limit` up to 940 mA |
| 0x12 | BOND | u8 foot, u8 extend (1 bond, 0 release) | heats one disc SMA until its end switch closes (at most 2 s) |
| 0x13 | LATCH | u8 foot, u8 open | pulls a stow bolt for 1.2 s |
| 0x14 | PIVOT | u8 foot, i16 angle (0.01 deg: 9000 or 18000), u8 release_after | the whole pivot below |
| 0x15 | DEPLOY | u8 foot, u8 out (1 out, 0 in) | drives that hinge's sled to its OUT or IN switch (one stroke, about 6 s, timeout 9 s); out is refused unless the foot is seated, its bolt is in and no disc is turning; in is refused unless the foot is seated, its bolt is in and its hinge is at 0. OK means started: watch `sled[]` in TELEMETRY |

The disc lock sequence (`src/disc.cpp`): heat the lock SMA wire and start turning, with the current limited to
600 mA until the disc has moved 5 degrees; stop heating (the pin rides on the ring); turn to the target quarter, where the spring snaps the pin in; wait
0.4 s; push 0.3 N m worth of current (250 mA) in the direction of the turn for 0.4 s: the encoder must move less than 1 degree, then the
angle is re-indexed to the exact quarter. Failure codes (DiscResult): 1 OK, 2 the disc did not move 5 degrees within 2 s of heating
(FAULT, nothing moved), 3 stalled and drove back (RETURNED), 4 the pin did not hold the push (FAULT, faults bit 7), 5 ended more than 6 degrees from
a quarter (FAULT), 6 motor fault or abandoned by STOP (FAULT).

PIVOT runs on the cube as a state machine (`src/main.cpp`):

1. Check: the foot is seated, its bolt is in and both sleds are in (else REFUSED); turn the disc to
   the nearest quarter turn through the lock sequence above (discs turn only while both sleds are in).
   A disc that fails (codes 2 to 6) ends the pivot with FAULT, or RETURNED for a stall.
1. Drive the foot's sled out until its OUT switch closes. Timeout (`DEPLOY_TIMEOUT_MS`), a start switch
   that does not open (`DEPLOY_LEAVE_MS`) or both switches closed: the sled is driven back in and the
   pivot answers FAULT.
1. Extend the disc's pins into the support's slots, pull the stow bolt, hinge to the angle with the
   current limit (the body turns about the foot's edge; the foot stays), wait 150 ms.
1. If release_after: retract the pins, swing the foot home to 0 and check that it is seated and the
   bolt is up (500 ms grace; if not: FAULT, the sled stays out).
1. Drive the sled in until its IN switch closes (failure: FAULT).
1. Report: ARRIVED, RETURNED or FAULT, with the hinge angle in `detail`.

A hinge stall drives back to the start, retracts the pins, brings the sled in if the foot came home
seated and bolted, and answers RETURNED (a harmless failure). Without release_after the pivot ends with
the foot open, bonded and its sled out: the host brings it home with HINGE 0, then DEPLOY in.

## Cube to host

| type | name | payload |
| --- | --- | --- |
| 0x02 | HELLO | u8 major (1), u8 minor (1), u8[6] MAC, u8 protocol version (3) |
| 0x20 | ACK | u16 seq, u8 type, u8 result (0 OK started, 1 ARRIVED, 2 RETURNED, 3 REFUSED, 4 FAULT), i16 detail |
| 0x21 | TELEMETRY | 20 Hz: u32 t_ms, u16 vbat_mV, i8 cell_C, u8 state, i16 hinge[2] (0.01 deg), i32 disc[2] (0.01 deg, unwrapped, from the motor encoder, re-indexed at each lock seating), i16 current_mA[4], i16 roll, pitch, yaw (0.01 deg), i16 acc_mg[3], u16 switches (MCP23017 bits 0-15 = GPA0-7, GPB0-7: 0 ext T, 1 ret T, 2 ext B, 3 ret B, 4 seated T, 5 seated B, 6 bolt T, 7 deploy T IN1 (output), 8 bolt B, 9 sled T out, 10 sled T in, 11 sled B out, 12 sled B in, 13 deploy T IN2, 14 deploy B IN1, 15 deploy B IN2 (outputs)), u8 faults (bit 0 driver, 1 encoder or axis fault, 2 SMA timeout, 3 battery low, 4 hot, 5 no command for 1 s, 6 sled fault, 7 a disc lock did not hold), u8 sled[2] (top, bottom: 0 unknown, 1 in, 2 out, 3 between, 4 going out, 5 going in, 6 fault), u8 sled_fault[2] (0 none, 1 timeout, 2 stuck at the start switch, 3 both switches closed), **u8 lock_result[2]** (top, bottom: DiscResult of the last disc turn, 0 none, 1 OK, 2 no release, 3 stalled, 4 lock slip, 5 missed quarter, 6 fault), **i32 disc_counts[2]** (raw 4x quadrature counts since boot, 51,087 per disc turn) |

A PIVOT gets two ACKs: OK when it starts, then ARRIVED, RETURNED, REFUSED or FAULT when it ends. Each
sled stroke adds about 6 s (task spec; the cube times out at 9 s), and the disc turn to the quarter adds up to
4 s, so a released pivot takes about 16 s more than the hinge move itself.

## Gateway serial lines (921600 baud)

Host to gateway: `TX <cube hex> <type hex> <payload hex>`. Example, a quarter pivot on the top
foot of cube 0x3A, releasing afterwards: `TX 3A 14 00282301` (foot 00, angle 9000 = 0x2328 sent
little-endian as 28 23, release 01). Top sled out on the same cube: `TX 3A 15 0001`. The gateway
answers `OK <seq>` and prints every frame it hears as `RX <cube hex> <type hex> <seq> <payload hex>`.
