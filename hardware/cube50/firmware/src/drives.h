// Four DC gearmotors on two DRV8833 bridges, each with a current sense. Hinge axes read an absolute
// AS5600 (analog out) on the axle end, 1:1 with the foot. Disc axes count the motor's quadrature encoder
// (Pololu #4761) with a PCNT unit: the angle is relative, and the quarter-turn lock is the index
// (reindex() sets the angle exactly once the pin has seated; see disc.h for the turn sequence).
// Position control along a minimum-jerk reference at 1 kHz, with a current limit and the hinge cube's
// stall rule: if the axis has not arrived STALL_S after its reference ended, it drives back to where
// it started (a harmless failure).
// A disc must be moved with DiscTurn (disc.h), which releases the lock first; a raw move() of a disc
// drives against the lock pin.
#pragma once
#include <Arduino.h>
#include "config.h"

enum Axis : uint8_t { HINGE_T = 0, HINGE_B = 1, DISC_T = 2, DISC_B = 3, AXES = 4 };

enum MoveState : uint8_t { IDLE = 0, MOVING = 1, ARRIVED = 2, RETURNING = 3, RETURNED = 4, FAULTED = 5 };

struct AxisCal {
  float zero_deg = 0.0f;   // hinge only: AS5600 reading at the axis zero (stowed). The disc zero is a count offset in RAM, set at boot and by reindex()
  float sign = 1.0f;       // +1 or -1 so that positive = foot opening / disc counter-clockwise seen from outside
};

class Drives {
 public:
  void begin();
  void calibrateZero(Axis a);           // present angle becomes zero (hinge: stored in NVS; disc: the lock index 0, RAM only)
  void flipSign(Axis a);                // reverse the axis direction (NVS, hinge and disc)
  void loadCal();
  void update();                        // call at 1 kHz from the control task
  bool move(Axis a, float target_deg, float rate_dps, float limit_A);  // false if refused (fault or a deploy interlock, deploy.h)
  void hold(Axis a);                    // hold here
  void setLimit(Axis a, float limit_A); // change the current limit of the running move
  void reindex(Axis a, float angle_deg);  // disc: the lock has seated, so the angle is exactly this (drops counting drift)
  // Constant-current push for lock and torque tests: open loop at the current limit for ms, direction
  // dir (+1 raises the angle). Afterwards the axis holds where it started. A disc push needs both sleds in.
  bool push(Axis a, float dir, float limit_A, uint32_t ms);
  bool pushing(Axis a) const { return push_ms_[a] != 0; }
  float pushMovedDeg(Axis a) const { return push_moved_[a]; }  // largest excursion from the start, degrees (positive)
  int32_t counts(Axis a) const { return counts_[a]; }          // disc: raw quadrature counts since boot
  void stopAll();                       // hold everything where it is
  void coast(Axis a);                   // zero output (bench only)
  void enable(bool on);
  float angle(Axis a) const { return angle_[a]; }
  float current(Axis a) const { return current_[a]; }
  MoveState state(Axis a) const { return st_[a]; }
  bool fault() const;
  float peakCurrent(Axis a) const { return peak_[a]; }
  void clearPeak(Axis a) { peak_[a] = 0; }

 private:
  void writeDuty(Axis a, float duty);
  float readEncoderDeg(Axis a);         // hinge AS5600 only
  int16_t readPcnt(Axis a);             // disc: the PCNT unit's counter
  void updateAngle(Axis a);
  float readCurrentA(Axis a);
  AxisCal cal_[AXES];
  float angle_[AXES] = {0}, raw_prev_[AXES] = {0}, turns_[AXES] = {0};
  int32_t counts_[AXES] = {0}, zero_counts_[AXES] = {0};
  int16_t pcnt_prev_[AXES] = {0};
  uint32_t push_ms_[AXES] = {0}, push_t0_[AXES] = {0};
  float push_dir_[AXES] = {0}, push_limit_[AXES] = {0}, push_duty_[AXES] = {0}, push_start_[AXES] = {0}, push_moved_[AXES] = {0};
  float current_[AXES] = {0}, peak_[AXES] = {0};
  float q0_[AXES] = {0}, q1_[AXES] = {0}, t0_[AXES] = {0}, span_[AXES] = {1};
  float integ_[AXES] = {0}, prev_err_[AXES] = {0}, limit_[AXES] = {0};
  float start_deg_[AXES] = {0};
  MoveState st_[AXES] = {IDLE, IDLE, IDLE, IDLE};
};

extern Drives drives;
float minJerk(float u);  // 10u^3 - 15u^4 + 6u^5 on [0, 1]
