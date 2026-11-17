#include "bond.h"

#include <Wire.h>
#include <driver/ledc.h>

#include "config.h"
#include "pins.h"

Bonds bonds;

namespace {
constexpr uint8_t MCP_ADDR = 0x20;
// IOCON.BANK = 0 (reset default): each A register is followed by its B register, and sequential
// addressing is on, so one transaction writes or reads a pair.
constexpr uint8_t MCP_IODIRA = 0x00, MCP_IPOLA = 0x02, MCP_GPPUA = 0x0C, MCP_GPIOA = 0x12, MCP_OLATA = 0x14;
const int SMA_PIN[SMA_CHANNELS] = {PIN_SMA_T_EXT, PIN_SMA_T_RET, PIN_SMA_B_EXT, PIN_SMA_B_RET, PIN_SMA_BOLT_T, PIN_SMA_BOLT_B, PIN_SMA_LOCK_T, PIN_SMA_LOCK_B};
constexpr int LEDC_BASE = 0;   // LEDC channels 0..7: all eight of the ESP32-S3's low-speed channels
constexpr int LEDC_BITS = 10;  // 0..1023 at 1 kHz
// Heating wires get phase slots 0..SMA_MAX_ACTIVE-1, a quarter period apart. With the duty cap of SMA_DUTY_MAX
// (0.40 x 1024 = 410) the last slot ends at 512 + 410 = 922 < 1024, so a pulse never wraps. Typical duties
// are 0.12 to 0.22 (123 to 225 counts), under the 256-count spacing: the pulses do not overlap at all.
constexpr uint32_t LEDC_SLOT_COUNTS = 256;

void mcpWrite16(uint8_t reg_a, uint16_t v) {
  Wire.beginTransmission(MCP_ADDR);
  Wire.write(reg_a);
  Wire.write((uint8_t)(v & 0xFF));  // port A
  Wire.write((uint8_t)(v >> 8));    // port B
  Wire.endTransmission();
}

// switch bit for "this channel's job is done"
int doneBit(SmaChannel c) {
  switch (c) {
    case SMA_T_EXT: return MCP_SW_EXT_T;
    case SMA_T_RET: return MCP_SW_RET_T;
    case SMA_B_EXT: return MCP_SW_EXT_B;
    case SMA_B_RET: return MCP_SW_RET_B;
    default: return -1;  // bolts run for a fixed time, locks until DiscTurn stops them
  }
}
}  // namespace

void Bonds::begin() {
  for (int c = 0; c < SMA_CHANNELS; ++c) {
    ledcSetup(LEDC_BASE + c, 1000, LEDC_BITS);
    ledcAttachPin(SMA_PIN[c], LEDC_BASE + c);
    ledcWrite(LEDC_BASE + c, 0);
  }
  // Outputs low before they become outputs (deploy motors coast). Switches to GND get the pull-up and
  // are inverted so a closed switch reads 1; the disc ring switches close to VBAT through a divider,
  // so they get neither (the first version pulled them up and inverted them too, against the netlist).
  mcpWrite16(MCP_OLATA, 0x0000);
  mcpWrite16(MCP_IODIRA, (uint16_t)~MCP_OUTPUTS);
  mcpWrite16(MCP_GPPUA, MCP_TO_GND);
  mcpWrite16(MCP_IPOLA, MCP_TO_GND);
  readSwitches();
}

bool Bonds::readSwitches() {
  Wire.beginTransmission(MCP_ADDR);
  Wire.write(MCP_GPIOA);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)MCP_ADDR, 2) != 2) return false;
  uint16_t a = Wire.read();
  sw_ = a | ((uint16_t)Wire.read() << 8);
  return true;
}

bool Bonds::heat(SmaChannel c, float ohm, uint32_t max_ms, bool timed) {
  bool used[SMA_MAX_ACTIVE] = {};
  int active = 0;
  for (int k = 0; k < SMA_CHANNELS; ++k) {
    if (op_[k] != OP_HEATING) continue;
    ++active;
    used[slot_[k] % SMA_MAX_ACTIVE] = true;
  }
  if (active >= SMA_MAX_ACTIVE) return false;
  int slot = 0;
  while (used[slot]) ++slot;
  slot_[c] = (uint8_t)slot;
  op_[c] = OP_HEATING;
  t_on_[c] = millis();
  ohm_[c] = ohm;
  max_ms_[c] = max_ms;
  timed_[c] = timed;
  return true;
}

// Duty at a phase offset: the pulse starts at the channel's slot, not at the period's start, so wires that
// heat together do not all draw their pulse current at the same instant.
void Bonds::smaWrite(SmaChannel c, float duty) {
  uint32_t counts = (uint32_t)(duty * ((1 << LEDC_BITS) - 1));
  if (counts == 0) {
    ledcWrite(LEDC_BASE + c, 0);
    return;
  }
  ledc_set_duty_with_hpoint(LEDC_LOW_SPEED_MODE, (ledc_channel_t)(LEDC_BASE + c), counts, slot_[c] * LEDC_SLOT_COUNTS);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, (ledc_channel_t)(LEDC_BASE + c));
}

void Bonds::off() {
  for (int c = 0; c < SMA_CHANNELS; ++c) {
    ledcWrite(LEDC_BASE + c, 0);
    if (op_[c] == OP_HEATING) op_[c] = OP_IDLE;
  }
}

bool Bonds::extend(uint8_t foot) {
  SmaChannel on = foot ? SMA_B_EXT : SMA_T_EXT, other = foot ? SMA_B_RET : SMA_T_RET;
  if (op_[other] == OP_HEATING || millis() - t_off_[on] < SMA_COOL_MS) return false;
  return heat(on, SMA_RING_OHM, SMA_TIMEOUT_MS, false);
}

bool Bonds::retract(uint8_t foot) {
  SmaChannel on = foot ? SMA_B_RET : SMA_T_RET, other = foot ? SMA_B_EXT : SMA_T_EXT;
  if (op_[other] == OP_HEATING || millis() - t_off_[on] < SMA_COOL_MS) return false;
  return heat(on, SMA_RING_OHM, SMA_TIMEOUT_MS, false);
}

bool Bonds::pullBolt(uint8_t foot, uint32_t hold_ms) {
  SmaChannel c = foot ? SMA_BOLT_B : SMA_BOLT_T;
  if (millis() - t_off_[c] < SMA_COOL_MS / 2) return false;
  return heat(c, SMA_BOLT_OHM, hold_ms, true);
}

bool Bonds::lockRelease(uint8_t foot) {
  SmaChannel c = foot ? SMA_LOCK_B : SMA_LOCK_T;
  if (op_[c] == OP_HEATING || millis() - t_off_[c] < SMA_LOCK_COOL_MS) return false;
  return heat(c, SMA_LOCK_OHM, LOCK_RELEASE_TIMEOUT_MS, false);
}

void Bonds::lockStop(uint8_t foot) {
  SmaChannel c = foot ? SMA_LOCK_B : SMA_LOCK_T;
  if (op_[c] != OP_HEATING) return;
  ledcWrite(LEDC_BASE + c, 0);
  t_off_[c] = millis();
  op_[c] = OP_DONE;
}

BondState Bonds::state(uint8_t foot) const {
  bool ext = sw_ & (1 << (foot ? MCP_SW_EXT_B : MCP_SW_EXT_T));
  bool ret = sw_ & (1 << (foot ? MCP_SW_RET_B : MCP_SW_RET_T));
  if (ext && ret) return BOND_UNKNOWN;  // both closed: a wiring fault
  if (ext) return BOND_EXTENDED;
  if (ret) return BOND_RETRACTED;
  return BOND_BETWEEN;
}

bool Bonds::stowed(uint8_t foot) const { return sw_ & (1 << (foot ? MCP_STOW_B : MCP_STOW_T)); }
bool Bonds::boltUp(uint8_t foot) const { return sw_ & (1 << (foot ? MCP_BOLT_B : MCP_BOLT_T)); }

void Bonds::update(float vbat) {
  readSwitches();
  for (int i = 0; i < SMA_CHANNELS; ++i) {
    SmaChannel c = (SmaChannel)i;
    if (op_[c] != OP_HEATING) continue;
    uint32_t on_ms = millis() - t_on_[c];
    int bit = doneBit(c);
    bool done = bit >= 0 && (sw_ & (1 << bit));
    bool timeout = on_ms > max_ms_[c];
    if (done || timeout || vbat < VBAT_CUTOFF_V) {
      ledcWrite(LEDC_BASE + c, 0);
      t_off_[c] = millis();
      op_[c] = done || timed_[c] ? OP_DONE : OP_TIMEOUT;
      continue;
    }
    // constant heating power: duty = P R / V^2
    smaWrite(c, constrain(SMA_TARGET_W * ohm_[c] / (vbat * vbat), 0.0f, SMA_DUTY_MAX));
  }
}
