// Bench-test console on USB serial (921600 baud). One command per line; results are JSON lines so
// they can be logged and tallied on the host (ASSEMBLY.md, "Bench tests").
//
// Changed 2026-11-17 11:24 (+0530): hinge deploy drive (bench deploy, sled); hinge tests deploy the sled first.
// Changed 2026-11-17 (60 mm redesign): disc moves run the lock sequence (disc.h); new "bench lock" (lock hold test)
// and "bench enc" (quadrature count check); "bench disc" is a constant-current push, run with the lock pin removed.
//
//   help
//   stop                                   hold every axis, all SMA off, both sleds off
//   cal <ht|hb|dt|db> zero                 this angle is zero (hinge: foot seated, stored; disc: lock index 0, not stored)
//   cal <ht|hb|dt|db> flip                 reverse the axis direction (if a move runs away)
//   move <ht|hb|dt|db> <deg> [dps] [A]     one move, prints the result ("refused" if a deploy interlock blocks it);
//                                          a disc move is the lock sequence to the nearest quarter turn, result is the DiscResult
//   sled <0|1> <out|in>                    one sled stroke, prints the result
//   bench hinge <0|1>                      unloaded sweep 0 -> 180 -> 0, current against angle every 5 deg
//   bench pivot <0|1> <deg> <N>            foot bonded to a fixed tile: body out to <deg> and back, N times
//   bench bond <0|1> <N>                   extend/retract N times against a test face: times and switch states
//   bench disc <0|1>                       torque ramp: disc pushes a lever on a scale at 0.2..0.94 A (3 s each; lock pin removed)
//   bench lock <0|1> [release]             lock hold test: push 0.3 N m worth of current both ways, the encoder must not
//                                          move; with "release" also heat the wire and show the disc then moves > 5 degrees
//   bench enc <0|1>                        encoder count check: 1 s at rest, a +90 degree turn, back; counts against 12,772
//   bench latch <0|1>                      pull the stow bolt, report the bolt and seat switches
//   bench deploy <0|1> <N>                 foot seated and bolted: sled out and in N times; stroke times, switch states, faults
//   bench imu                              roll, pitch, yaw at 50 Hz for 10 s
#ifndef GATEWAY
#include <Arduino.h>

#include "bond.h"
#include "config.h"
#include "deploy.h"
#include "disc.h"
#include "drives.h"
#include "imu.h"

namespace {
String line;

int axisOf(const String& s) {
  if (s == "ht") return HINGE_T;
  if (s == "hb") return HINGE_B;
  if (s == "dt") return DISC_T;
  if (s == "db") return DISC_B;
  return -1;
}

const char* stateName(MoveState s) {
  static const char* n[] = {"idle", "moving", "arrived", "returning", "returned", "faulted"};
  return n[s];
}

// Wait for a move to end; returns the final state. Prints nothing.
MoveState waitMove(Axis a, uint32_t timeout_ms = 15000) {
  uint32_t t = millis();
  while (millis() - t < timeout_ms) {
    MoveState s = drives.state(a);
    if (s == ARRIVED || s == RETURNED || s == FAULTED) return s;
    delay(5);
  }
  return drives.state(a);
}

const char* discResultName(DiscResult r) {
  static const char* n[] = {"none", "ok", "no_release", "stalled", "lock_slip", "missed_quarter", "fault"};
  return n[r];
}

// Wait for a disc turn (all four lock steps) to end; returns its result.
DiscResult waitDisc(uint8_t foot, uint32_t timeout_ms = 20000) {
  uint32_t t = millis();
  delay(20);  // the control task picks the turn up within 5 ms
  while (discTurn.busy(foot) && millis() - t < timeout_ms) delay(5);
  return discTurn.result(foot);
}

const char* sledName(SledState s) {
  static const char* n[] = {"unknown", "in", "out", "between", "going_out", "going_in", "fault"};
  return n[s];
}

const char* sledFaultName(SledFault f) {
  static const char* n[] = {"none", "timeout", "stuck", "both_switches"};
  return n[f];
}

// One sled stroke; returns true if it reached the wanted end. Waits past the firmware's own timeout.
bool sledStroke(uint8_t foot, bool out) {
  if (!(out ? sleds.out(foot) : sleds.in(foot))) return false;
  delay(20);  // the control task picks the request up within 5 ms
  uint32_t t = millis();
  while (sleds.moving(foot) && millis() - t < DEPLOY_TIMEOUT_MS + 1000) delay(5);
  return out ? sleds.isOut(foot) : sleds.isIn(foot);
}

void sledReport(uint8_t foot, const char* what, bool ok) {
  Serial.printf("{\"sled\":%u,\"%s\":%d,\"state\":\"%s\",\"fault\":\"%s\",\"ms\":%lu,\"sw_out\":%d,\"sw_in\":%d}\n", foot, what, ok,
                sledName(sleds.state(foot)), sledFaultName(sleds.fault(foot)), (unsigned long)sleds.strokeMs(foot), sleds.switchOut(foot),
                sleds.switchIn(foot));
}

bool waitBond(uint8_t foot, BondState want, uint32_t timeout_ms = SMA_TIMEOUT_MS + 500) {
  uint32_t t = millis();
  while (millis() - t < timeout_ms) {
    if (bonds.state(foot) == want) return true;
    delay(10);
  }
  return false;
}

void hingeSweep(uint8_t foot) {
  Axis h = foot ? HINGE_B : HINGE_T;
  if (!sleds.isOut(foot) && !sledStroke(foot, true)) {
    Serial.println("{\"bench\":\"hinge\",\"error\":\"sled did not deploy (foot seated and bolted?)\"}");
    return;
  }
  bonds.pullBolt(foot, 1200);
  delay(400);
  for (float target : {180.0f, 0.0f}) {
    drives.move(h, target, HINGE_RATE_DPS, HINGE_CURRENT_LIMIT);
    float next = drives.angle(h);
    while (drives.state(h) == MOVING) {
      if (fabsf(drives.angle(h) - next) >= 5.0f) {
        next = drives.angle(h);
        Serial.printf("{\"bench\":\"hinge\",\"foot\":%u,\"deg\":%.1f,\"A\":%.3f}\n", foot, next, drives.current(h));
      }
      delay(2);
    }
    Serial.printf("{\"bench\":\"hinge\",\"foot\":%u,\"target\":%.0f,\"result\":\"%s\"}\n", foot, target, stateName(drives.state(h)));
  }
  delay(300);
  sledReport(foot, "in", sledStroke(foot, false));
}

// The pivot test of ASSEMBLY.md: the foot is bonded to a fixed tile (on the floor, or on the side of
// a cube clamped to the bench), the body turns out and comes back. Counts reached and returned;
// the operator watches for harm (anything knocked, dropped or cracked) and types it in the log.
void pivotTrials(uint8_t foot, float deg, int n) {
  Axis h = foot ? HINGE_B : HINGE_T;
  int reached = 0, returned = 0, faulted = 0;
  if (!sleds.isOut(foot) && !sledStroke(foot, true)) {
    Serial.println("{\"bench\":\"pivot\",\"error\":\"sled did not deploy (foot seated and bolted?)\"}");
    return;
  }
  if (!(bonds.extend(foot) && waitBond(foot, BOND_EXTENDED))) {
    Serial.println("{\"bench\":\"pivot\",\"error\":\"bond did not extend\"}");
    return;
  }
  for (int k = 0; k < n; ++k) {
    bonds.pullBolt(foot, 1200);
    delay(400);
    drives.clearPeak(h);
    uint32_t t = millis();
    drives.move(h, deg, HINGE_RATE_DPS, HINGE_CURRENT_LIMIT);
    MoveState out = waitMove(h);
    float peak_out = drives.peakCurrent(h);
    float tilt = sqrtf(imu.roll() * imu.roll() + imu.pitch() * imu.pitch());
    delay(300);
    drives.move(h, 0.0f, HINGE_RATE_DPS, HINGE_CURRENT_LIMIT);
    MoveState back = waitMove(h);
    reached += out == ARRIVED;
    returned += out == RETURNED;
    faulted += out == FAULTED || back == FAULTED;
    Serial.printf("{\"bench\":\"pivot\",\"foot\":%u,\"trial\":%d,\"deg\":%.0f,\"out\":\"%s\",\"back\":\"%s\",\"peak_A\":%.3f,\"tilt_deg_at_goal\":%.1f,\"s\":%.2f,\"seated\":%d,\"bolt\":%d}\n",
                  foot, k, deg, stateName(out), stateName(back), peak_out, tilt, (millis() - t) / 1000.0f, bonds.stowed(foot), bonds.boltUp(foot));
    if (faulted) break;
    delay(500);
  }
  bonds.retract(foot);
  waitBond(foot, BOND_RETRACTED);
  sledReport(foot, "in", sledStroke(foot, false));
  Serial.printf("{\"bench\":\"pivot\",\"summary\":true,\"n\":%d,\"reached\":%d,\"returned\":%d,\"faulted\":%d}\n", n, reached, returned, faulted);
}

void bondCycles(uint8_t foot, int n) {
  int ok_ext = 0, ok_ret = 0;
  for (int k = 0; k < n; ++k) {
    uint32_t t0 = millis();
    while (!bonds.extend(foot)) delay(50);  // waits out the cool-down
    bool e = waitBond(foot, BOND_EXTENDED);
    uint32_t t1 = millis();
    while (!bonds.retract(foot)) delay(50);
    bool r = waitBond(foot, BOND_RETRACTED);
    ok_ext += e;
    ok_ret += r;
    Serial.printf("{\"bench\":\"bond\",\"foot\":%u,\"cycle\":%d,\"extended\":%d,\"extend_s\":%.2f,\"retracted\":%d,\"cycle_s\":%.2f}\n", foot, k, e, (t1 - t0) / 1000.0f, r,
                  (millis() - t0) / 1000.0f);
  }
  Serial.printf("{\"bench\":\"bond\",\"summary\":true,\"n\":%d,\"extended\":%d,\"retracted\":%d}\n", n, ok_ext, ok_ret);
}

// Disc against a lever arm resting on a kitchen scale: torque = scale grams x 9.81e-3 x lever m. Each
// step pushes at a constant current for 3 s; the operator reads the scale and types it next to the line.
// Run with the lock pin removed (it would take the torque) and both sleds in.
void discRamp(uint8_t foot) {
  Axis d = foot ? DISC_B : DISC_T;
  if (!sleds.allIn()) {
    Serial.println("{\"bench\":\"disc\",\"error\":\"both sleds must be in\"}");
    return;
  }
  for (float amps : {0.2f, 0.4f, 0.6f, 0.8f, DISC_CURRENT_LIMIT}) {
    if (!drives.push(d, 1.0f, amps, 3000)) {
      Serial.println("{\"bench\":\"disc\",\"error\":\"push refused\"}");
      return;
    }
    delay(3100);
    Serial.printf("{\"bench\":\"disc\",\"foot\":%u,\"limit_A\":%.2f,\"A\":%.3f,\"deg\":%.2f}\n", foot, amps, drives.current(d), drives.pushMovedDeg(d));
    delay(1200);  // the disc goes back to where it started
  }
}

// Lock hold test: with the pin engaged, 0.3 N m worth of current either way must not turn the disc
// (LOCK_TEST_MAX_DEG, set from the bench: the encoder is on the motor side, so backlash shows here too).
// With "release" the wire is heated while pushing: the disc must then turn LOCK_RELEASE_DEG, which proves the
// test can see motion. That leaves the disc off its quarter: run `move dt 0` (or db) afterwards.
void lockTest(uint8_t foot, bool release) {
  Axis d = foot ? DISC_B : DISC_T;
  if (!sleds.allIn()) {
    Serial.println("{\"bench\":\"lock\",\"error\":\"both sleds must be in\"}");
    return;
  }
  float held[2] = {0, 0};
  for (int k = 0; k < 2; ++k) {
    drives.push(d, k == 0 ? 1.0f : -1.0f, LOCK_TEST_CURRENT_A, LOCK_TEST_MS);
    delay(LOCK_TEST_MS + 100);
    held[k] = drives.pushMovedDeg(d);
    Serial.printf("{\"bench\":\"lock\",\"foot\":%u,\"push\":\"%s\",\"A\":%.3f,\"moved_deg\":%.3f,\"limit_deg\":%.2f,\"holds\":%d}\n", foot,
                  k == 0 ? "plus" : "minus", LOCK_TEST_CURRENT_A, held[k], LOCK_TEST_MAX_DEG, held[k] < LOCK_TEST_MAX_DEG);
    delay(200);
  }
  if (release) {
    uint32_t t = millis();
    float start = drives.angle(d);
    if (!bonds.lockRelease(foot)) {
      Serial.println("{\"bench\":\"lock\",\"error\":\"lock wire still cooling\"}");
      return;
    }
    drives.push(d, 1.0f, LOCK_TEST_CURRENT_A * 2.0f, LOCK_RELEASE_TIMEOUT_MS);
    while (fabsf(drives.angle(d) - start) < LOCK_RELEASE_DEG && millis() - t < LOCK_RELEASE_TIMEOUT_MS) delay(2);
    bonds.lockStop(foot);
    Serial.printf("{\"bench\":\"lock\",\"foot\":%u,\"released\":%d,\"ms\":%lu,\"moved_deg\":%.2f}\n", foot, fabsf(drives.angle(d) - start) >= LOCK_RELEASE_DEG,
                  (unsigned long)(millis() - t), fabsf(drives.angle(d) - start));
    drives.hold(d);
  }
}

// Encoder count check: the counts must not drift at rest, and a +90 degree turn must give
// DISC_COUNTS_PER_REV / 4 = 12,772 counts (error well under 1 %); the way back must return to the start count
// to within backlash. Raw counts are printed, so lock re-indexing cannot hide a count error.
void encoderCheck(uint8_t foot) {
  Axis d = foot ? DISC_B : DISC_T;
  int32_t c0 = drives.counts(d);
  delay(1000);
  int32_t c1 = drives.counts(d);
  Serial.printf("{\"bench\":\"enc\",\"foot\":%u,\"rest_counts\":%ld}\n", foot, (long)(c1 - c0));
  float a0 = drives.angle(d);
  if (!discTurn.start(foot, a0 + DISC_QUARTER_DEG, DISC_RATE_DPS, DISC_CURRENT_LIMIT)) {
    Serial.println("{\"bench\":\"enc\",\"error\":\"turn refused\"}");
    return;
  }
  DiscResult r = waitDisc(foot);
  int32_t c2 = drives.counts(d);
  float expect = DISC_COUNTS_PER_REV / 4.0f;
  Serial.printf("{\"bench\":\"enc\",\"foot\":%u,\"turn\":\"%s\",\"counts\":%ld,\"expected\":%.0f,\"error_pct\":%.3f}\n", foot, discResultName(r), (long)(c2 - c1), expect,
                100.0f * (fabsf((float)(c2 - c1)) - expect) / expect);
  delay(500);
  discTurn.start(foot, a0, DISC_RATE_DPS, DISC_CURRENT_LIMIT);
  r = waitDisc(foot);
  int32_t c3 = drives.counts(d);
  Serial.printf("{\"bench\":\"enc\",\"foot\":%u,\"back\":\"%s\",\"net_counts\":%ld}\n", foot, discResultName(r), (long)(c3 - c1));
}

void latchTest(uint8_t foot) {
  Serial.printf("{\"bench\":\"latch\",\"foot\":%u,\"before_bolt\":%d,\"seated\":%d}\n", foot, bonds.boltUp(foot), bonds.stowed(foot));
  bonds.pullBolt(foot, 2000);
  uint32_t t = millis();
  while (bonds.boltUp(foot) && millis() - t < 2000) delay(5);
  Serial.printf("{\"bench\":\"latch\",\"foot\":%u,\"pulled_ms\":%lu,\"bolt\":%d}\n", foot, millis() - t, bonds.boltUp(foot));
  delay(SMA_COOL_MS);
  Serial.printf("{\"bench\":\"latch\",\"foot\":%u,\"after_cool_bolt\":%d}\n", foot, bonds.boltUp(foot));
}

// Bench test 3b: the foot seated and bolted, the sled out and back in n times. Stall current is not
// visible to the firmware (the deploy bridge's sense resistors are not read): measure it with a meter
// in series with one motor lead.
void deployCycles(uint8_t foot, int n) {
  int ok_out = 0, ok_in = 0;
  uint32_t max_out = 0, max_in = 0;
  if (!(bonds.stowed(foot) && bonds.boltUp(foot) && sleds.isIn(foot))) {
    Serial.println("{\"bench\":\"deploy\",\"error\":\"start with the foot seated, the bolt in and the sled in\"}");
    return;
  }
  for (int k = 0; k < n; ++k) {
    bool o = sledStroke(foot, true);
    uint32_t ms_out = sleds.strokeMs(foot);
    bool sw_ok_out = sleds.switchOut(foot) && !sleds.switchIn(foot);
    const char* f_out = sledFaultName(sleds.fault(foot));
    delay(300);
    bool i = o && sledStroke(foot, false);
    uint32_t ms_in = sleds.strokeMs(foot);
    bool sw_ok_in = sleds.switchIn(foot) && !sleds.switchOut(foot);
    ok_out += o;
    ok_in += i;
    if (o) max_out = max(max_out, ms_out);
    if (i) max_in = max(max_in, ms_in);
    Serial.printf("{\"bench\":\"deploy\",\"foot\":%u,\"cycle\":%d,\"out\":%d,\"out_ms\":%lu,\"out_switches_ok\":%d,\"out_fault\":\"%s\",\"in\":%d,\"in_ms\":%lu,\"in_switches_ok\":%d,\"in_fault\":\"%s\"}\n",
                  foot, k, o, (unsigned long)ms_out, sw_ok_out, f_out, i, (unsigned long)ms_in, sw_ok_in, sledFaultName(sleds.fault(foot)));
    if (!(o && i)) break;  // a fault ends the run: look before cycling a jammed screw
    delay(300);
  }
  Serial.printf("{\"bench\":\"deploy\",\"summary\":true,\"n\":%d,\"out_ok\":%d,\"in_ok\":%d,\"max_out_ms\":%lu,\"max_in_ms\":%lu,\"timeout_ms\":%lu}\n", n, ok_out, ok_in,
                (unsigned long)max_out, (unsigned long)max_in, (unsigned long)DEPLOY_TIMEOUT_MS);
}

void imuStream() {
  for (int k = 0; k < 500; ++k) {
    Serial.printf("{\"bench\":\"imu\",\"roll\":%.2f,\"pitch\":%.2f,\"yaw\":%.2f}\n", imu.roll(), imu.pitch(), imu.yaw());
    delay(20);
  }
}

void run(const String& cmd) {
  char w[6][16] = {{0}};
  int n = sscanf(cmd.c_str(), "%15s %15s %15s %15s %15s %15s", w[0], w[1], w[2], w[3], w[4], w[5]);
  String c0 = w[0], c1 = w[1];
  if (c0 == "help") {
    Serial.println("stop | cal <ht|hb|dt|db> zero|flip | move <axis> <deg> [dps] [A] | sled <0|1> out|in | bench hinge|pivot|bond|disc|lock|enc|latch|deploy|imu ...");
  } else if (c0 == "stop") {
    discTurn.stop();
    drives.stopAll();
    bonds.off();
    sleds.stop();
  } else if (c0 == "cal" && n >= 3 && axisOf(c1) >= 0) {
    if (String(w[2]) == "flip") drives.flipSign((Axis)axisOf(c1));
    else drives.calibrateZero((Axis)axisOf(c1));
    Serial.printf("{\"cal\":\"%s\",\"%s\":true}\n", w[1], w[2]);
  } else if (c0 == "move" && n >= 3 && axisOf(c1) >= 0) {
    Axis a = (Axis)axisOf(c1);
    bool hinge = a == HINGE_T || a == HINGE_B;
    float dps = n >= 4 ? atof(w[3]) : (hinge ? HINGE_RATE_DPS : DISC_RATE_DPS);
    float amps = n >= 5 ? atof(w[4]) : (hinge ? HINGE_CURRENT_LIMIT : DISC_CURRENT_LIMIT);
    if (!hinge) {
      uint8_t foot = a == DISC_B;
      if (!discTurn.start(foot, atof(w[2]), dps, amps)) {
        Serial.printf("{\"move\":\"%s\",\"result\":\"refused\"}\n", w[1]);  // both sleds in, no other disc turning, lock wire cooled
        return;
      }
      DiscResult r = waitDisc(foot);
      Serial.printf("{\"move\":\"%s\",\"result\":\"%s\",\"deg\":%.2f,\"counts\":%ld}\n", w[1], discResultName(r), drives.angle(a), (long)drives.counts(a));
      return;
    }
    if (!drives.move(a, atof(w[2]), dps, amps)) {
      Serial.printf("{\"move\":\"%s\",\"result\":\"refused\"}\n", w[1]);  // a hinge needs its sled out
      return;
    }
    MoveState s = waitMove(a);
    Serial.printf("{\"move\":\"%s\",\"result\":\"%s\",\"deg\":%.2f}\n", w[1], stateName(s), drives.angle(a));
  } else if (c0 == "sled" && n >= 3) {
    uint8_t foot = (uint8_t)atoi(w[1]);
    bool out = String(w[2]) == "out";
    sledReport(foot, out ? "out" : "in", foot <= 1 && sledStroke(foot, out));
  } else if (c0 == "bench" && n >= 2) {
    uint8_t foot = n >= 3 ? (uint8_t)atoi(w[2]) : 0;
    if (c1 == "hinge") hingeSweep(foot);
    else if (c1 == "pivot" && n >= 5) pivotTrials(foot, atof(w[3]), atoi(w[4]));
    else if (c1 == "bond" && n >= 4) bondCycles(foot, atoi(w[3]));
    else if (c1 == "disc") discRamp(foot);
    else if (c1 == "lock") lockTest(foot, n >= 4 && String(w[3]) == "release");
    else if (c1 == "enc") encoderCheck(foot);
    else if (c1 == "latch") latchTest(foot);
    else if (c1 == "deploy" && n >= 4) deployCycles(foot, atoi(w[3]));
    else if (c1 == "imu") imuStream();
    else Serial.println("{\"error\":\"bad bench command\"}");
  } else {
    Serial.println("{\"error\":\"unknown command, try help\"}");
  }
}
}  // namespace

void benchSetup() { line.reserve(64); }

void benchPoll() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      run(line);
      line = "";
    } else if (line.length() < 63) {
      line += c;
    }
  }
}
#endif
