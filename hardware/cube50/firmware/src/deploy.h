// Hinge deploy drive (added 2026-11-17 11:24 +0530). Each foot hinge rides on a sled that slides
// 7.9 mm between "in" (every hinge part inside the cube) and "out" (the axle on the edge line, for a
// pivot). One N20 1:150 gearmotor with an M3 lead screw per sled, on/off through DRV8833 #3 from four
// MCP23017 outputs; two end switches per sled on the MCP23017. The screw self-locks, so the motor
// is off at both ends. All MCP23017 traffic stays in the control task: out() and in() only post a
// request, update() acts on it.
//
// Interlocks (DESIGN.md, "Pivot: the retractable hinge"):
//   out: the foot is seated, its bolt is in, and neither disc is turning;
//   in:  the foot is seated, its bolt is in and its hinge is at 0;
//   Drives::move refuses a disc move unless both sleds are in, and a hinge move unless its sled is out.
#pragma once
#include <Arduino.h>

enum SledState : uint8_t {
  SLED_UNKNOWN = 0,  // before the first debounced read
  SLED_IN = 1,       // IN switch closed
  SLED_OUT = 2,      // OUT switch closed
  SLED_BETWEEN = 3,  // neither closed, not driving (pushed by hand, or stopped)
  SLED_GOING_OUT = 4,
  SLED_GOING_IN = 5,
  SLED_FAULT = 6,    // see SledFault; cleared by the next accepted out() or in()
};

enum SledFault : uint8_t {
  SF_NONE = 0,
  SF_TIMEOUT = 1,        // the end switch did not close within DEPLOY_TIMEOUT_MS
  SF_STUCK = 2,          // the start switch did not open within DEPLOY_LEAVE_MS (stalled, or motor reversed)
  SF_BOTH_SWITCHES = 3,  // both switches closed: wiring or switch fault
};

class Sleds {
 public:
  void begin();                     // both motors off (call after Bonds::begin has set up the MCP23017)
  void update(uint16_t sw);         // 200 Hz from the control task, with the MCP23017 switch word
  bool out(uint8_t foot);           // start driving the sled out; false if an interlock or a running stroke refuses
  bool in(uint8_t foot);            // start driving the sled in
  void stop();                      // both motors off now (they hold: the screw self-locks)
  SledState state(uint8_t foot) const { return st_[foot]; }
  SledFault fault(uint8_t foot) const { return fault_[foot]; }
  bool isIn(uint8_t foot) const { return st_[foot] == SLED_IN; }
  bool isOut(uint8_t foot) const { return st_[foot] == SLED_OUT; }
  bool allIn() const { return isIn(0) && isIn(1); }
  bool moving(uint8_t foot) const { return st_[foot] == SLED_GOING_OUT || st_[foot] == SLED_GOING_IN; }
  bool switchOut(uint8_t foot) const { return out_sw_[foot]; }   // debounced
  bool switchIn(uint8_t foot) const { return in_sw_[foot]; }
  uint32_t strokeMs(uint8_t foot) const { return stroke_ms_[foot]; }  // last stroke, start to end switch

 private:
  enum Req : uint8_t { REQ_NONE = 0, REQ_OUT, REQ_IN, REQ_STOP };
  void drive(uint8_t foot, int dir);  // +1 out, -1 in, 0 coast, 2 brake
  void writeOutputs();
  volatile Req req_[2] = {REQ_NONE, REQ_NONE};
  SledState st_[2] = {SLED_UNKNOWN, SLED_UNKNOWN};
  SledFault fault_[2] = {SF_NONE, SF_NONE};
  bool out_sw_[2] = {false, false}, in_sw_[2] = {false, false};
  bool raw_out_[2] = {false, false}, raw_in_[2] = {false, false};
  uint32_t raw_ms_[2] = {0, 0};      // when the raw switch pair last changed
  uint32_t t0_[2] = {0, 0}, brake_until_[2] = {0, 0}, stroke_ms_[2] = {0, 0};
  int dir_[2] = {0, 0};
  uint16_t olat_ = 0, olat_sent_ = 0xFFFF;
};

extern Sleds sleds;
