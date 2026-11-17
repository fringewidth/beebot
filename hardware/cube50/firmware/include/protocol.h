// Cube50 radio messages (ESP-NOW, 2.4 GHz). Spec and examples in firmware/PROTOCOL.md.
// Little-endian, packed. Every frame starts with Header. One frame per ESP-NOW packet (<= 250 B).
// Version 2 (2026-11-17 11:24 +0530): hinge deploy drive: MSG_DEPLOY, 16-bit switch word, sled states.
// Version 3 (2026-11-17, 60 mm redesign): discs turn only to quarter turns through the lock sequence (MSG_DISC
// answers twice, like MSG_PIVOT); telemetry adds lock_result[2] and disc_counts[2], faults bit 7; HELLO appends
// the protocol version; a cube ignores every message but PING when the header's version is not its own.
#pragma once
#include <stdint.h>

constexpr uint8_t PROTO_MAGIC = 0xC5;
constexpr uint8_t PROTO_VERSION = 3;
constexpr uint8_t CUBE_BROADCAST = 0xFF;

enum MsgType : uint8_t {
  MSG_PING = 0x01,        // host -> cube(s): reply with HELLO
  MSG_HELLO = 0x02,       // cube -> host: id, firmware version, MAC
  MSG_STOP = 0x03,        // host -> cube(s): stop all drives, hold positions, no SMA
  MSG_SYNC = 0x04,        // host -> all: host clock in microseconds (for timed starts)
  MSG_HINGE = 0x10,       // host -> cube: move a foot hinge
  MSG_DISC = 0x11,        // host -> cube: turn a disc to the nearest quarter turn to the target (foot stowed, both sleds in); answers twice
  MSG_BOND = 0x12,        // host -> cube: extend or retract a disc's pins
  MSG_LATCH = 0x13,       // host -> cube: open (pull) or release a stow bolt
  MSG_PIVOT = 0x14,       // host -> cube: whole pivot sequence on one foot (see PROTOCOL.md)
  MSG_DEPLOY = 0x15,      // host -> cube: drive a hinge sled out or in (foot seated and bolted)
  MSG_ACK = 0x20,         // cube -> host: result of a command
  MSG_TELEMETRY = 0x21,   // cube -> host: state at 20 Hz
  MSG_LOG = 0x22,         // cube -> host: text line (bench mode)
};

enum Foot : uint8_t { FOOT_TOP = 0, FOOT_BOTTOM = 1 };

enum Result : uint8_t {
  RES_OK = 0,           // accepted (for moves: started)
  RES_ARRIVED = 1,      // move finished within ARRIVE_DEG
  RES_RETURNED = 2,     // stalled, drove back to the start: harmless failure
  RES_REFUSED = 3,      // precondition failed (foot not stowed for a disc move, a deploy interlock, battery low, busy)
  RES_FAULT = 4,        // driver fault, encoder out of range, SMA timeout, sled timeout or switch fault
};

#pragma pack(push, 1)
struct Header {
  uint8_t magic;     // PROTO_MAGIC
  uint8_t version;   // PROTO_VERSION
  uint8_t type;      // MsgType
  uint8_t cube;      // destination (host -> cube) or source (cube -> host); 0xFF = all
  uint16_t seq;      // host sequence number, echoed in ACK
  uint32_t start_us; // 0 = now; else start when the synced clock reaches this (spec 7.3)
};

struct HingeCmd { uint8_t foot; int16_t target_cdeg; uint16_t rate_cdps; uint16_t limit_mA; };
struct DiscCmd { uint8_t foot; int32_t target_cdeg; uint16_t rate_cdps; uint16_t limit_mA; uint8_t relative; };  // target is snapped to a multiple of 9000
struct BondCmd { uint8_t foot; uint8_t extend; };   // extend 1 = bond, 0 = release
struct LatchCmd { uint8_t foot; uint8_t open; };    // open 1 = pull the bolt (unlatch)
struct PivotCmd { uint8_t foot; int16_t angle_cdeg; uint8_t release_after; };  // 9000 or 18000
struct DeployCmd { uint8_t foot; uint8_t out; };   // out 1 = sled out (axle to the edge line), 0 = in
struct SyncCmd { uint64_t host_us; };

struct Ack { uint16_t seq; uint8_t type; uint8_t result; int16_t detail; };

struct Telemetry {
  uint32_t t_ms;
  uint16_t vbat_mV;
  int8_t cell_C;
  uint8_t state;             // State enum in main.cpp
  int16_t hinge_cdeg[2];     // foot angles, 0 = stowed
  int32_t disc_cdeg[2];      // unwrapped disc angles, from the motor encoder, re-indexed at each lock seating
  int16_t current_mA[4];     // hinge T, hinge B, disc T, disc B
  int16_t roll_cdeg, pitch_cdeg, yaw_cdeg;
  int16_t acc_mg[3];
  uint16_t switches;         // MCP23017 GPA0..7 then GPB0..7, bit per McpPin (pins.h); output bits read back the deploy drive's inputs
  uint8_t faults;            // bit 0 DRV nFAULT, 1 encoder, 2 SMA timeout, 3 battery low, 4 hot, 5 link lost, 6 sled fault, 7 disc lock did not hold
  uint8_t sled[2];           // SledState (deploy.h) of the top and bottom hinge sled
  uint8_t sled_fault[2];     // SledFault (deploy.h)
  uint8_t lock_result[2];    // DiscResult (disc.h) of the last disc turn on the top and bottom foot
  int32_t disc_counts[2];    // raw quadrature counts since boot (4x), for the encoder count check
};
#pragma pack(pop)
