#include "disc.h"

#include "bond.h"
#include "config.h"
#include "deploy.h"
#include "drives.h"

DiscTurn discTurn;

namespace {
Axis discOf(uint8_t foot) { return foot ? DISC_B : DISC_T; }
}  // namespace

bool DiscTurn::start(uint8_t foot, float target_deg, float rate_dps, float limit_A) {
  if (foot > 1 || busy() || !sleds.allIn()) return false;
  Axis d = discOf(foot);
  float target = DISC_QUARTER_DEG * roundf(target_deg / DISC_QUARTER_DEG);
  target_[foot] = target;
  start_[foot] = drives.angle(d);
  limit_[foot] = limit_A;
  dir_[foot] = target >= start_[foot] ? 1.0f : -1.0f;
  if (fabsf(target - start_[foot]) < LOCK_CAPTURE_DEG) {  // already on that quarter: the pin holds it, nothing to release
    result_[foot] = DR_OK;
    return true;
  }
  if (!bonds.lockRelease(foot)) return false;
  if (!drives.move(d, target, rate_dps, min(limit_A, DISC_LOCK_RELEASE_LIMIT_A))) {
    bonds.lockStop(foot);
    return false;
  }
  pending_[foot] = DR_OK;
  result_[foot] = DR_NONE;
  t_[foot] = millis();
  step_[foot] = DS_RELEASE;
  return true;
}

void DiscTurn::finish(uint8_t foot, DiscResult r) {
  bonds.lockStop(foot);
  result_[foot] = r;
  step_[foot] = DS_IDLE;
}

void DiscTurn::stop() {
  for (uint8_t f = 0; f < 2; ++f) {
    if (step_[f] != DS_IDLE) finish(f, DR_FAULT);
  }
}

void DiscTurn::update() {
  uint32_t now = millis();
  for (uint8_t f = 0; f < 2; ++f) {
    if (step_[f] == DS_IDLE) continue;
    Axis d = discOf(f);
    MoveState ms = drives.state(d);
    switch (step_[f]) {
      case DS_RELEASE:
        if (fabsf(drives.angle(d) - start_[f]) > LOCK_RELEASE_DEG) {
          bonds.lockStop(f);                 // the pin rides on the ring now
          drives.setLimit(d, limit_[f]);
          step_[f] = DS_TURN;
        } else if (ms == FAULTED || ms == RETURNING || ms == RETURNED || now - t_[f] > LOCK_RELEASE_TIMEOUT_MS || bonds.op(f ? SMA_LOCK_B : SMA_LOCK_T) == OP_TIMEOUT) {
          drives.hold(d);
          finish(f, ms == FAULTED ? DR_FAULT : DR_NO_RELEASE);
        }
        break;
      case DS_TURN:
        if (ms == ARRIVED) {
          t_[f] = now;
          step_[f] = DS_SEAT;
        } else if (ms == RETURNED) {
          pending_[f] = DR_STALLED;
          target_[f] = start_[f];            // back on the start quarter
          t_[f] = now;
          step_[f] = DS_SEAT;
        } else if (ms == FAULTED) {
          finish(f, DR_FAULT);
        }
        break;
      case DS_SEAT:
        if (now - t_[f] < LOCK_SEAT_MS) break;
        if (fabsf(drives.angle(d) - target_[f]) > LOCK_CAPTURE_DEG) {
          finish(f, DR_MISSED);
        } else if (LOCK_VERIFY_AFTER_TURN) {
          // push the way the turn went: that takes up the gear backlash first, so a seated pin shows no motion
          float dir = pending_[f] == DR_STALLED ? -dir_[f] : dir_[f];
          if (drives.push(d, dir, LOCK_TEST_CURRENT_A, LOCK_TEST_MS)) step_[f] = DS_VERIFY;
          else finish(f, DR_FAULT);
        } else {
          drives.reindex(d, target_[f]);
          finish(f, pending_[f]);
        }
        break;
      case DS_VERIFY:
        if (drives.pushing(d)) break;
        if (drives.pushMovedDeg(d) < LOCK_TEST_MAX_DEG) {
          drives.reindex(d, target_[f]);
          finish(f, pending_[f]);
        } else {
          finish(f, DR_LOCK_SLIP);
        }
        break;
      default:
        break;
    }
  }
}
