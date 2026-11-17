// Cube50 firmware: control task at 1 kHz (drives), 200 Hz (bonds, hinge sleds, IMU), radio commands,
// telemetry, pivot sequences, safety, and the bench-test console on USB serial (bench.cpp).
// Changed 2026-11-17 11:24 (+0530): hinge deploy drive (deploy.h) in the pivot, MSG_DEPLOY, interlocks.
// Changed 2026-11-17 (60 mm redesign): disc turns go through the lock sequence (disc.h), MSG_DISC answers twice,
// protocol version 3 is checked on receipt.
// With -DGATEWAY=1 the same source builds the host's USB-to-ESP-NOW bridge instead.
#include <Arduino.h>
#include <WiFi.h>
#include <Wire.h>

#include "bond.h"
#include "config.h"
#include "deploy.h"
#include "disc.h"
#include "drives.h"
#include "imu.h"
#include "pins.h"
#include "protocol.h"
#include "radio.h"

void benchSetup();
void benchPoll();

#ifndef GATEWAY

enum State : uint8_t { ST_BOOT = 0, ST_READY, ST_PIVOT, ST_LOW_BATTERY, ST_FAULT };

namespace {
volatile State state = ST_BOOT;
uint32_t last_cmd_ms = 0, last_tx_ms = 0;
float vbat = 4.0f;
uint8_t faults = 0;

// ------------------------------------------------------------ pivot sequence (MSG_PIVOT)
// Changed 2026-11-17 11:24 (+0530) for the hinge deploy drive (deploy.h).
// 1. check the foot is seated, its bolt is in and both sleds are in; disc to the nearest quarter turn
//    through the lock sequence (disc.h: release, turn, seat, verify; discs turn only while both sleds are in),
// 2. drive the sled out until its OUT switch (timeout: drive it back in, FAULT),
// 3. bond the foot to its support, pull the stow bolt, hinge to the angle (the body turns, the foot stays),
// 4. if release_after: release the support, swing the foot home to 0 and check the bolt seats,
// 5. drive the sled in until its IN switch (timeout: FAULT, the sled stays where it stopped),
// 6. report (ACK with the result and the hinge angle).
// A hinge stall drives back and releases: a harmless failure (RES_RETURNED), and the sled goes back in
// if the foot came home seated. Without release_after the pivot ends with the foot open and its sled
// out; the host brings it home with HINGE 0 and then DEPLOY in.
enum PivotStep : uint8_t {
  PV_IDLE, PV_INDEX, PV_DEPLOY, PV_BOND, PV_UNLATCH, PV_TURN, PV_SETTLE, PV_RELEASE, PV_STOW, PV_RETRACT, PV_BACK_RELEASE
};
struct Pivot {
  PivotStep step = PV_IDLE;
  uint8_t foot = 0;
  float angle = 90.0f;
  bool release_after = true;
  uint16_t seq = 0;
  uint32_t t = 0;
  uint8_t result = RES_ARRIVED;  // reported once the sled is back in
} pv;

Axis hingeOf(uint8_t f) { return f ? HINGE_B : HINGE_T; }
Axis discOf(uint8_t f) { return f ? DISC_B : DISC_T; }

void ack(uint16_t seq, uint8_t type, uint8_t result, int16_t detail = 0) {
  Ack a = {seq, type, result, detail};
  radio::sendToHost(MSG_ACK, seq, &a, sizeof(a));
}

// MSG_DISC answers twice: OK when the turn starts, then ARRIVED, RETURNED or FAULT (detail = DiscResult)
// when the lock sequence ends. A turn started by PIVOT is reported by the pivot instead.
struct DiscAck {
  bool pending = false;
  uint16_t seq = 0;
} disc_ack[2];

void discAckUpdate() {
  for (uint8_t f = 0; f < 2; ++f) {
    if (!disc_ack[f].pending || discTurn.busy(f)) continue;
    DiscResult r = discTurn.result(f);
    ack(disc_ack[f].seq, MSG_DISC, r == DR_OK ? RES_ARRIVED : (r == DR_STALLED ? RES_RETURNED : RES_FAULT), r);
    disc_ack[f].pending = false;
  }
}

void pivotFinish(uint8_t result) {
  ack(pv.seq, MSG_PIVOT, result, (int16_t)(drives.angle(hingeOf(pv.foot)) * 100));
  pv.step = PV_IDLE;
  state = ST_READY;
}

// Bring the sled home, then report `result`; if the foot is not seated and bolted the sled stays out.
void pivotRetract(uint8_t result) {
  pv.result = result;
  if (sleds.isIn(pv.foot)) return pivotFinish(result);
  if (!sleds.in(pv.foot)) return pivotFinish(result == RES_ARRIVED ? RES_FAULT : result);
  pv.step = PV_RETRACT;
}

void pivotUpdate() {
  uint32_t now = millis();
  Axis h = hingeOf(pv.foot);
  switch (pv.step) {
    case PV_IDLE: return;
    case PV_INDEX:
      if (discTurn.busy(pv.foot)) break;
      if (discTurn.result(pv.foot) == DR_OK) {
        if (!sleds.out(pv.foot)) return pivotFinish(RES_REFUSED);
        pv.step = PV_DEPLOY;
      } else {
        pivotFinish(discTurn.result(pv.foot) == DR_STALLED ? RES_RETURNED : RES_FAULT);
      }
      break;
    case PV_DEPLOY:
      if (sleds.isOut(pv.foot)) {
        bonds.extend(pv.foot);
        pv.step = PV_BOND;
        pv.t = now;
      } else if (sleds.state(pv.foot) == SLED_FAULT) {
        pivotRetract(RES_FAULT);  // timeout or stuck: drive it back in, then report
      }
      break;
    case PV_BOND:
      if (bonds.state(pv.foot) == BOND_EXTENDED) {
        bonds.pullBolt(pv.foot, 1200);
        pv.step = PV_UNLATCH;
        pv.t = now;
      } else if (now - pv.t > SMA_TIMEOUT_MS + 200) {
        bonds.retract(pv.foot);
        pivotRetract(RES_FAULT);
      }
      break;
    case PV_UNLATCH:
      if (!bonds.boltUp(pv.foot) || now - pv.t > 1200) {
        if (drives.move(h, pv.angle, HINGE_RATE_DPS, HINGE_CURRENT_LIMIT)) {
          pv.step = PV_TURN;
        } else {  // refused (the sled left its OUT switch, or the hinge is faulted): release and report
          bonds.retract(pv.foot);
          pv.result = RES_FAULT;
          pv.step = PV_BACK_RELEASE;
          pv.t = now;
        }
      }
      break;
    case PV_TURN:
      if (drives.state(h) == ARRIVED) {
        pv.step = PV_SETTLE;
        pv.t = now;
      } else if (drives.state(h) == RETURNED || drives.state(h) == FAULTED) {
        bonds.retract(pv.foot);
        pv.result = drives.state(h) == RETURNED ? RES_RETURNED : RES_FAULT;
        pv.step = PV_BACK_RELEASE;
        pv.t = now;
      }
      break;
    case PV_SETTLE:
      if (now - pv.t > 150) {  // module_c's HOLD_TIME
        if (!pv.release_after) return pivotFinish(RES_ARRIVED);  // foot open, sled out
        bonds.retract(pv.foot);
        pv.step = PV_RELEASE;
        pv.t = now;
      }
      break;
    case PV_RELEASE:
      if (bonds.state(pv.foot) == BOND_RETRACTED) {
        drives.move(h, 0.0f, HINGE_RATE_DPS, HINGE_CURRENT_LIMIT);  // foot swings home; the bolt's chamfer lets it in
        pv.step = PV_STOW;
        pv.t = 0;
      } else if (now - pv.t > SMA_TIMEOUT_MS + 200) {
        pivotFinish(RES_FAULT);  // arrived but still bonded to the old support: the host decides
      }
      break;
    case PV_STOW:
      if (drives.state(h) == RETURNED || drives.state(h) == FAULTED) return pivotFinish(RES_FAULT);
      if (drives.state(h) != ARRIVED) break;
      if (!pv.t) pv.t = now;
      if (bonds.stowed(pv.foot) && bonds.boltUp(pv.foot)) pivotRetract(RES_ARRIVED);
      else if (now - pv.t > 500) pivotFinish(RES_FAULT);  // home but not seated or not bolted: the sled stays out
      break;
    case PV_RETRACT:
      if (sleds.isIn(pv.foot)) pivotFinish(pv.result);
      else if (sleds.state(pv.foot) == SLED_FAULT) pivotFinish(RES_FAULT);
      break;
    case PV_BACK_RELEASE:
      if (bonds.state(pv.foot) == BOND_RETRACTED || now - pv.t > SMA_TIMEOUT_MS + 200) {
        if (drives.state(h) == RETURNED && bonds.stowed(pv.foot) && bonds.boltUp(pv.foot)) pivotRetract(pv.result);
        else pivotFinish(pv.result);
      }
      break;
  }
}

bool startPivot(const PivotCmd& c, uint16_t seq) {
  if (state != ST_READY || c.foot > 1 || vbat < VBAT_LOW_V) return false;
  if (!bonds.stowed(c.foot) || !bonds.boltUp(c.foot) || !sleds.allIn()) return false;  // step 1
  pv.foot = c.foot;
  pv.angle = c.angle_cdeg / 100.0f;
  pv.release_after = c.release_after;
  pv.seq = seq;
  pv.result = RES_ARRIVED;
  float idx = DISC_QUARTER_DEG * roundf(drives.angle(discOf(c.foot)) / DISC_QUARTER_DEG);
  if (!discTurn.start(c.foot, idx, DISC_RATE_DPS, DISC_CURRENT_LIMIT)) return false;
  pv.step = PV_INDEX;
  state = ST_PIVOT;
  return true;
}

// ------------------------------------------------------------ commands
void onFrame(const uint8_t* mac, const uint8_t* data, int len) {
  const Header* h = reinterpret_cast<const Header*>(data);
  if (len < (int)sizeof(Header) || h->magic != PROTO_MAGIC) return;
  if (h->cube != radio::cubeId() && h->cube != CUBE_BROADCAST) return;
  radio::rememberHost(mac);
  last_cmd_ms = millis();
  if (h->version != PROTO_VERSION && h->type != MSG_PING) return;  // a host on another protocol version gets HELLO (with ours) and nothing else
  const uint8_t* p = data + sizeof(Header);
  int n = len - (int)sizeof(Header);
  uint8_t res = RES_OK;
  switch (h->type) {
    case MSG_PING: {
      uint8_t hello[9] = {1, 1};  // firmware version 1.1, then the MAC, then the protocol version this firmware speaks
      WiFi.macAddress(hello + 2);
      hello[8] = PROTO_VERSION;
      radio::sendToHost(MSG_HELLO, h->seq, hello, sizeof(hello));
      return;
    }
    case MSG_SYNC:
      if (n >= (int)sizeof(SyncCmd)) radio::onSync(reinterpret_cast<const SyncCmd*>(p)->host_us);
      return;
    case MSG_STOP:
      discTurn.stop();
      drives.stopAll();
      bonds.off();
      sleds.stop();
      pv.step = PV_IDLE;
      if (state == ST_PIVOT) state = ST_READY;
      break;
    case MSG_HINGE: {
      if (n < (int)sizeof(HingeCmd) || state != ST_READY) { res = RES_REFUSED; break; }
      auto c = *reinterpret_cast<const HingeCmd*>(p);
      // refused unless that hinge's sled is out (Drives::move interlock)
      if (!drives.move(hingeOf(c.foot), c.target_cdeg / 100.0f, c.rate_cdps / 100.0f, min(c.limit_mA / 1000.0f, HINGE_CURRENT_LIMIT))) res = RES_REFUSED;
      break;
    }
    case MSG_DISC: {
      if (n < (int)sizeof(DiscCmd) || state != ST_READY) { res = RES_REFUSED; break; }
      auto c = *reinterpret_cast<const DiscCmd*>(p);
      if (!bonds.stowed(c.foot) || fabsf(drives.angle(hingeOf(c.foot))) > STOWED_DEG) { res = RES_REFUSED; break; }  // the dog only drives a stowed foot
      if (c.foot > 1) { res = RES_REFUSED; break; }
      Axis d = discOf(c.foot);
      float target = c.target_cdeg / 100.0f + (c.relative ? drives.angle(d) : 0.0f);
      // The lock holds only quarter turns, so the target snaps to the nearest one. Refused unless both
      // sleds are in and no disc is turning. If no turn is needed it still answers twice.
      if (!discTurn.start(c.foot, target, c.rate_cdps / 100.0f, min(c.limit_mA / 1000.0f, DISC_CURRENT_LIMIT))) { res = RES_REFUSED; break; }
      disc_ack[c.foot].pending = true;
      disc_ack[c.foot].seq = h->seq;
      break;
    }
    case MSG_BOND: {
      if (n < (int)sizeof(BondCmd)) { res = RES_REFUSED; break; }
      auto c = *reinterpret_cast<const BondCmd*>(p);
      res = (c.extend ? bonds.extend(c.foot) : bonds.retract(c.foot)) ? RES_OK : RES_REFUSED;
      break;
    }
    case MSG_LATCH: {
      if (n < (int)sizeof(LatchCmd)) { res = RES_REFUSED; break; }
      auto c = *reinterpret_cast<const LatchCmd*>(p);
      res = c.open && bonds.pullBolt(c.foot, 1200) ? RES_OK : RES_REFUSED;
      break;
    }
    case MSG_DEPLOY: {
      if (n < (int)sizeof(DeployCmd) || state != ST_READY) { res = RES_REFUSED; break; }
      auto c = *reinterpret_cast<const DeployCmd*>(p);
      res = (c.out ? sleds.out(c.foot) : sleds.in(c.foot)) ? RES_OK : RES_REFUSED;
      break;
    }
    case MSG_PIVOT: {
      if (n < (int)sizeof(PivotCmd)) { res = RES_REFUSED; break; }
      res = startPivot(*reinterpret_cast<const PivotCmd*>(p), h->seq) ? RES_OK : RES_REFUSED;
      break;
    }
    default:
      return;
  }
  ack(h->seq, h->type, res);
}

void sendTelemetry() {
  Telemetry t = {};
  t.t_ms = millis();
  t.vbat_mV = (uint16_t)(vbat * 1000);
  t.state = state;
  for (int f = 0; f < 2; ++f) {
    t.hinge_cdeg[f] = (int16_t)(drives.angle(hingeOf(f)) * 100);
    t.disc_cdeg[f] = (int32_t)(drives.angle(discOf(f)) * 100);
    t.disc_counts[f] = drives.counts(discOf(f));
    t.lock_result[f] = discTurn.result(f);
  }
  for (int a = 0; a < AXES; ++a) t.current_mA[a] = (int16_t)(drives.current((Axis)a) * 1000);
  t.roll_cdeg = (int16_t)(imu.roll() * 100);
  t.pitch_cdeg = (int16_t)(imu.pitch() * 100);
  t.yaw_cdeg = (int16_t)(fmodf(imu.yaw(), 360.0f) * 100);
  for (int i = 0; i < 3; ++i) t.acc_mg[i] = (int16_t)(imu.acc()[i] * 1000);
  t.switches = bonds.switches();
  t.faults = faults;
  for (int f = 0; f < 2; ++f) {
    t.sled[f] = sleds.state(f);
    t.sled_fault[f] = sleds.fault(f);
  }
  radio::sendToHost(MSG_TELEMETRY, 0, &t, sizeof(t));
}

void controlTask(void*) {
  TickType_t last = xTaskGetTickCount();
  uint32_t k = 0;
  for (;;) {
    drives.update();
    if (++k % 5 == 0) {
      vbat = 0.98f * vbat + 0.02f * (2.0f * analogReadMilliVolts(PIN_VBAT_SENSE) / 1000.0f);
      bonds.update(vbat);
      sleds.update(bonds.switches());
      imu.update(0.005f);
      discTurn.update();
      discAckUpdate();
      pivotUpdate();
    }
    vTaskDelayUntil(&last, 1);  // 1 kHz with CONFIG_FREERTOS_HZ=1000 (the Arduino core's default)
  }
}

void safety() {
  faults = 0;
  if (drives.fault()) faults |= 1;
  for (int a = 0; a < AXES; ++a) if (drives.state((Axis)a) == FAULTED) faults |= 2;
  for (int c = 0; c < SMA_CHANNELS; ++c) if (bonds.op((SmaChannel)c) == OP_TIMEOUT) faults |= 4;
  if (sleds.state(0) == SLED_FAULT || sleds.state(1) == SLED_FAULT) faults |= 64;
  if (discTurn.lockFault(0) || discTurn.lockFault(1)) faults |= 128;
  if (vbat < VBAT_LOW_V) faults |= 8;
  if (millis() - last_cmd_ms > LINK_TIMEOUT_MS) faults |= 32;
  if (vbat < VBAT_CUTOFF_V) {
    discTurn.stop();
    drives.stopAll();
    bonds.off();
    sleds.stop();
    delay(20);  // let the control task write the deploy outputs low
    drives.enable(false);
    esp_deep_sleep_start();  // bonds and stow bolts hold without power; joints hold on their gearboxes
  }
  if (vbat < VBAT_LOW_V && state == ST_READY) state = ST_LOW_BATTERY;
}
}  // namespace

void setup() {
  Serial.begin(921600);
  pinMode(PIN_LED, OUTPUT);
  Wire.begin(PIN_SDA, PIN_SCL, 400000);
  drives.begin();
  bonds.begin();
  sleds.begin();
  if (imu.begin()) imu.calibrateGyro();
  radio::begin(onFrame);
  benchSetup();
  xTaskCreatePinnedToCore(controlTask, "control", 4096, nullptr, 5, nullptr, 1);
  state = ST_READY;
}

void loop() {
  safety();
  if (millis() - last_tx_ms >= TELEMETRY_MS) {
    last_tx_ms = millis();
    sendTelemetry();
  }
  digitalWrite(PIN_LED, (millis() / (state == ST_READY ? 1000 : 150)) % 2);
  benchPoll();
  delay(1);
}

#else  // ------------------------------------------------------------ GATEWAY

// Serial lines from the host:  "TX <cube hex> <type hex> <payload hex>\n" -> ESP-NOW broadcast (the
// header is built here; payload is the struct after the header). Frames heard are printed as
// "RX <cube hex> <type hex> <seq> <payload hex>\n".
namespace {
uint16_t seq = 1;
void onFrame(const uint8_t*, const uint8_t* data, int len) {
  const Header* h = reinterpret_cast<const Header*>(data);
  Serial.printf("RX %02X %02X %u ", h->cube, h->type, h->seq);
  for (int i = sizeof(Header); i < len; ++i) Serial.printf("%02X", data[i]);
  Serial.println();
}
int hexval(char c) { return isdigit(c) ? c - '0' : (toupper(c) - 'A' + 10); }
}  // namespace

void benchSetup() {}
void benchPoll() {}

void setup() {
  Serial.begin(921600);
  radio::begin(onFrame);
}

void loop() {
  static String line;
  while (Serial.available()) {
    char c = Serial.read();
    if (c != '\n') { line += c; continue; }
    // TX <cube> <type> <payload>
    int a = line.indexOf(' '), b = line.indexOf(' ', a + 1), c2 = line.indexOf(' ', b + 1);
    if (line.startsWith("TX") && a > 0 && b > 0) {
      uint8_t buf[250];
      Header h = {PROTO_MAGIC, PROTO_VERSION, (uint8_t)strtol(line.substring(b + 1, c2 > 0 ? c2 : line.length()).c_str(), nullptr, 16),
                  (uint8_t)strtol(line.substring(a + 1, b).c_str(), nullptr, 16), seq++, 0};
      memcpy(buf, &h, sizeof(h));
      size_t n = sizeof(h);
      String hex = c2 > 0 ? line.substring(c2 + 1) : "";
      for (size_t i = 0; i + 1 < (size_t)hex.length() && n < sizeof(buf); i += 2) buf[n++] = hexval(hex[i]) * 16 + hexval(hex[i + 1]);
      radio::send(nullptr, buf, n);
      Serial.printf("OK %u\n", h.seq);
    }
    line = "";
  }
}

#endif
