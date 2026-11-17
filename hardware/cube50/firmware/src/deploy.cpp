#include "deploy.h"

#include <Wire.h>

#include "bond.h"
#include "config.h"
#include "disc.h"
#include "drives.h"
#include "pins.h"

Sleds sleds;

namespace {
constexpr uint8_t MCP_ADDR = 0x20;
constexpr uint8_t MCP_OLATA = 0x14;  // OLATB follows (IOCON.BANK = 0, sequential addressing)
const McpPin IN1[2] = {MCP_DEP_T_IN1, MCP_DEP_B_IN1};
const McpPin IN2[2] = {MCP_DEP_T_IN2, MCP_DEP_B_IN2};
const McpPin SW_OUT[2] = {MCP_SLED_OUT_T, MCP_SLED_OUT_B};
const McpPin SW_IN[2] = {MCP_SLED_IN_T, MCP_SLED_IN_B};

Axis hingeAxis(uint8_t foot) { return foot ? HINGE_B : HINGE_T; }
bool discTurning() {
  if (discTurn.busy()) return true;
  for (Axis d : {DISC_T, DISC_B}) {
    if (drives.state(d) == MOVING || drives.state(d) == RETURNING) return true;
  }
  return false;
}
}  // namespace

void Sleds::begin() {
  for (int f = 0; f < 2; ++f) drive(f, 0);
  writeOutputs();
}

void Sleds::drive(uint8_t foot, int dir) {
  // DRV8833: IN1 high IN2 low = forward, the reverse = backward, both low = coast, both high = brake
  bool a = false, b = false;
  int d = DEPLOY_REVERSED[foot] ? -dir : dir;
  if (dir == 2) {
    a = b = true;
  } else if (d > 0) {
    a = true;
  } else if (d < 0) {
    b = true;
  }
  olat_ &= ~(mcpBit(IN1[foot]) | mcpBit(IN2[foot]));
  if (a) olat_ |= mcpBit(IN1[foot]);
  if (b) olat_ |= mcpBit(IN2[foot]);
  dir_[foot] = dir;
}

void Sleds::writeOutputs() {
  if (olat_ == olat_sent_) return;
  Wire.beginTransmission(MCP_ADDR);
  Wire.write(MCP_OLATA);
  Wire.write((uint8_t)(olat_ & 0xFF));
  Wire.write((uint8_t)(olat_ >> 8));
  if (Wire.endTransmission() == 0) olat_sent_ = olat_;  // retried on the next update if the bus failed
}

bool Sleds::out(uint8_t foot) {
  if (foot > 1 || moving(foot) || st_[foot] == SLED_OUT) return false;
  if (!bonds.stowed(foot) || !bonds.boltUp(foot) || discTurning()) return false;
  req_[foot] = REQ_OUT;
  return true;
}

bool Sleds::in(uint8_t foot) {
  if (foot > 1 || moving(foot) || st_[foot] == SLED_IN) return false;
  if (!bonds.stowed(foot) || !bonds.boltUp(foot) || fabsf(drives.angle(hingeAxis(foot))) > STOWED_DEG) return false;
  req_[foot] = REQ_IN;
  return true;
}

void Sleds::stop() {
  req_[0] = req_[1] = REQ_STOP;
}

void Sleds::update(uint16_t sw) {
  uint32_t now = millis();
  for (uint8_t f = 0; f < 2; ++f) {
    // Debounce: the pair of switch readings must hold for DEPLOY_DEBOUNCE_MS.
    bool ro = sw & mcpBit(SW_OUT[f]), ri = sw & mcpBit(SW_IN[f]);
    if (ro != raw_out_[f] || ri != raw_in_[f]) {
      raw_out_[f] = ro;
      raw_in_[f] = ri;
      raw_ms_[f] = now;
    }
    if (now - raw_ms_[f] >= DEPLOY_DEBOUNCE_MS) {
      out_sw_[f] = raw_out_[f];
      in_sw_[f] = raw_in_[f];
    }

    Req r = req_[f];
    req_[f] = REQ_NONE;
    if (r == REQ_STOP) {
      drive(f, 0);
      if (moving(f)) st_[f] = SLED_BETWEEN;
    } else if (r == REQ_OUT || r == REQ_IN) {
      fault_[f] = SF_NONE;
      st_[f] = r == REQ_OUT ? SLED_GOING_OUT : SLED_GOING_IN;
      t0_[f] = now;
      drive(f, r == REQ_OUT ? 1 : -1);
    }

    if (brake_until_[f] && (int32_t)(now - brake_until_[f]) >= 0) {
      brake_until_[f] = 0;
      drive(f, 0);
    }

    if (moving(f)) {
      bool going_out = st_[f] == SLED_GOING_OUT;
      bool at_goal = going_out ? out_sw_[f] : in_sw_[f];
      bool at_start = going_out ? in_sw_[f] : out_sw_[f];
      uint32_t run = now - t0_[f];
      SledFault fl = SF_NONE;
      if (out_sw_[f] && in_sw_[f]) fl = SF_BOTH_SWITCHES;
      else if (at_goal) {
        drive(f, 2);  // brief brake so the nut stops at the switch, then coast
        brake_until_[f] = now + DEPLOY_BRAKE_MS;
        stroke_ms_[f] = run;
        st_[f] = going_out ? SLED_OUT : SLED_IN;
      } else if (at_start && run > DEPLOY_LEAVE_MS) fl = SF_STUCK;
      else if (run > DEPLOY_TIMEOUT_MS) fl = SF_TIMEOUT;
      if (fl != SF_NONE) {
        drive(f, 0);
        fault_[f] = fl;
        st_[f] = SLED_FAULT;
      }
    } else if (st_[f] != SLED_FAULT) {
      // At rest the state follows the switches (a sled pushed by hand shows up here).
      if (out_sw_[f] && in_sw_[f]) {
        fault_[f] = SF_BOTH_SWITCHES;
        st_[f] = SLED_FAULT;
      } else {
        st_[f] = out_sw_[f] ? SLED_OUT : (in_sw_[f] ? SLED_IN : SLED_BETWEEN);
      }
    }
  }
  writeOutputs();
}
