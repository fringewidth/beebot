// Disc turn sequence with the quarter-turn lock (60 mm design). Each foot's disc is held at quarter turns by
// a spring-loaded pin that is engaged by default (stowed and open); a 0.15 mm SMA wire releases it. The
// disc angle comes from the motor's quadrature encoder (Drives, counts), and every seated lock re-indexes
// it: the pin fixes the disc at an exact multiple of 90 degrees.
//
// Sequence of start(foot, target):
//   1. RELEASE: heat the lock wire and start turning (current limited to DISC_LOCK_RELEASE_LIMIT_A so a stuck
//      pin is not rammed). Once the disc has moved LOCK_RELEASE_DEG the pin rides on the ring: stop heating.
//      If it has not moved by LOCK_RELEASE_TIMEOUT_MS: heating off, hold, DR_NO_RELEASE.
//   2. TURN: full current limit, to the target quarter (the spring snaps the pin in on arrival).
//   3. SEAT: wait LOCK_SEAT_MS at the target.
//   4. VERIFY: push LOCK_TEST_TORQUE_NM (as current) in the direction of the turn for LOCK_TEST_MS; the encoder
//      must move less than LOCK_TEST_MAX_DEG, else DR_LOCK_SLIP. Then re-index the angle to the quarter.
// A stall drives back to the start quarter (harmless, DR_STALLED). One disc turns at a time (the cell
// cannot feed two disc motors at their cap).
#pragma once
#include <Arduino.h>

enum DiscResult : uint8_t {
  DR_NONE = 0,        // no turn yet
  DR_OK = 1,          // at the target quarter, lock verified (or already there)
  DR_NO_RELEASE = 2,  // the disc did not move LOCK_RELEASE_DEG while the lock wire heated
  DR_STALLED = 3,     // stalled, drove back to the start quarter
  DR_LOCK_SLIP = 4,   // the pin did not hold the verify push
  DR_MISSED = 5,      // ended further than LOCK_CAPTURE_DEG from the quarter (lost counts, or the pin skipped a hole)
  DR_FAULT = 6,       // motor fault, refused mid-way, or abandoned by stop()
};

class DiscTurn {
 public:
  bool start(uint8_t foot, float target_deg, float rate_dps, float limit_A);  // target snaps to the nearest quarter; false if refused
  void update();                           // 200 Hz from the control task
  void stop();                             // abandon any turn: lock wires off, discs hold
  bool busy() const { return step_[0] != DS_IDLE || step_[1] != DS_IDLE; }
  bool busy(uint8_t foot) const { return step_[foot] != DS_IDLE; }
  DiscResult result(uint8_t foot) const { return result_[foot]; }
  bool lockFault(uint8_t foot) const { return result_[foot] == DR_LOCK_SLIP; }
  float targetDeg(uint8_t foot) const { return target_[foot]; }

 private:
  enum Step : uint8_t { DS_IDLE = 0, DS_RELEASE, DS_TURN, DS_SEAT, DS_VERIFY };
  void finish(uint8_t foot, DiscResult r);
  Step step_[2] = {DS_IDLE, DS_IDLE};
  DiscResult result_[2] = {DR_NONE, DR_NONE};
  DiscResult pending_[2] = {DR_OK, DR_OK};  // result once SEAT and VERIFY are through (DR_STALLED after a stall)
  float target_[2] = {0, 0}, start_[2] = {0, 0}, limit_[2] = {0, 0}, dir_[2] = {1, 1};
  uint32_t t_[2] = {0, 0};
};

extern DiscTurn discTurn;
