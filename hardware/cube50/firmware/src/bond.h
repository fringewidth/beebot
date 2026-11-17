// Disc bonds, stow bolts and disc locks: eight SMA wires on low-side MOSFETs (LEDC PWM), state switches read
// through an MCP23017. This class owns the MCP23017's setup and reads both ports (16-bit switch word,
// bit = McpPin in pins.h); the hinge deploy drive (deploy.h) writes its outputs (changed 2026-11-17 11:24 +0530). The disc ring is bistable (over-centre spring): a wire is heated only
// until its end switch closes, then switched off; nothing is powered while a bond holds. A disc lock wire
// (changed 2026-11-17 for the 60 mm design) has no switch: DiscTurn heats it until the disc has moved
// LOCK_RELEASE_DEG and then calls lockStop. The eight LEDC channels are phase-staggered and at most
// SMA_MAX_ACTIVE wires heat at once, so the cell sees at most about two overlapping pulses.
#pragma once
#include <Arduino.h>

enum SmaChannel : uint8_t { SMA_T_EXT = 0, SMA_T_RET, SMA_B_EXT, SMA_B_RET, SMA_BOLT_T, SMA_BOLT_B, SMA_LOCK_T, SMA_LOCK_B, SMA_CHANNELS };
enum BondState : uint8_t { BOND_UNKNOWN = 0, BOND_RETRACTED, BOND_EXTENDED, BOND_BETWEEN };
enum OpState : uint8_t { OP_IDLE = 0, OP_HEATING, OP_DONE, OP_TIMEOUT };

class Bonds {
 public:
  void begin();
  void update(float vbat);              // call at 200 Hz
  bool extend(uint8_t foot);            // bond: pins out and turned under the partner's lips
  bool retract(uint8_t foot);           // release
  bool pullBolt(uint8_t foot, uint32_t hold_ms);  // unlatch a stowed foot for hold_ms, then let the spring return
  bool lockRelease(uint8_t foot);       // heat the disc lock wire until lockStop() (or LOCK_RELEASE_TIMEOUT_MS: OP_TIMEOUT)
  void lockStop(uint8_t foot);          // lock wire off now; the spring seats the pin once the wire has cooled
  bool lockHeating(uint8_t foot) const { return op_[foot ? SMA_LOCK_B : SMA_LOCK_T] == OP_HEATING; }
  BondState state(uint8_t foot) const;
  bool stowed(uint8_t foot) const;      // foot seated on the core
  bool boltUp(uint8_t foot) const;      // bolt in the foot's hole
  OpState op(SmaChannel c) const { return op_[c]; }
  uint16_t switches() const { return sw_; }  // both ports; output bits read back their latch
  void off();                           // every wire off now

 private:
  bool heat(SmaChannel c, float ohm, uint32_t max_ms, bool timed);  // false if SMA_MAX_ACTIVE wires already heat
  void smaWrite(SmaChannel c, float duty);
  bool readSwitches();
  uint16_t sw_ = 0;
  OpState op_[SMA_CHANNELS] = {};
  uint32_t t_on_[SMA_CHANNELS] = {}, t_off_[SMA_CHANNELS] = {}, max_ms_[SMA_CHANNELS] = {};
  float ohm_[SMA_CHANNELS] = {};
  bool timed_[SMA_CHANNELS] = {};      // ending at max_ms is the normal end (bolts); otherwise it is a timeout
  uint8_t slot_[SMA_CHANNELS] = {};    // phase slot while heating, 0..SMA_MAX_ACTIVE-1
};

extern Bonds bonds;
