#include "drives.h"

#include <Preferences.h>
#include <driver/mcpwm.h>
#include <driver/pcnt.h>

#include "deploy.h"
#include "pins.h"

Drives drives;

namespace {
struct Bridge { mcpwm_unit_t unit; mcpwm_timer_t timer; int in1, in2; mcpwm_io_signals_t sa, sb; };
// One MCPWM timer per motor, operator A on IN1, B on IN2. 20 kHz, slow decay (brake) between pulses.
const Bridge BRIDGE[AXES] = {
    {MCPWM_UNIT_0, MCPWM_TIMER_0, PIN_HINGE_T_IN1, PIN_HINGE_T_IN2, MCPWM0A, MCPWM0B},
    {MCPWM_UNIT_0, MCPWM_TIMER_1, PIN_HINGE_B_IN1, PIN_HINGE_B_IN2, MCPWM1A, MCPWM1B},
    {MCPWM_UNIT_1, MCPWM_TIMER_0, PIN_DISC_T_IN1, PIN_DISC_T_IN2, MCPWM0A, MCPWM0B},
    {MCPWM_UNIT_1, MCPWM_TIMER_1, PIN_DISC_B_IN1, PIN_DISC_B_IN2, MCPWM1A, MCPWM1B},
};
const int ENC_PIN[2] = {PIN_ENC_HINGE_T, PIN_ENC_HINGE_B};  // hinge AS5600 analog outputs
// Disc quadrature encoders: A and B pins and the PCNT unit of DISC_T and DISC_B (index = axis - DISC_T)
const int ENC_A[2] = {PIN_ENC_DISC_T_A, PIN_ENC_DISC_B_A};
const int ENC_B[2] = {PIN_ENC_DISC_T_B, PIN_ENC_DISC_B_B};
const pcnt_unit_t PCNT_OF[2] = {PCNT_UNIT_0, PCNT_UNIT_1};
constexpr float PUSH_KI = 5.0f;  // duty per ampere of error per second in the constant-current push
const int ISEN_PIN[AXES] = {PIN_ISEN_HINGE_T, PIN_ISEN_HINGE_B, PIN_ISEN_DISC_T, PIN_ISEN_DISC_B};
constexpr float DT = 0.001f;
const char* KEYS[AXES] = {"ht", "hb", "dt", "db"};

bool isHinge(Axis a) { return a == HINGE_T || a == HINGE_B; }
float arriveDeg(Axis a) { return isHinge(a) ? ARRIVE_DEG : DISC_ARRIVE_DEG; }
float nearDeg(Axis a) { return isHinge(a) ? NEAR_DEG : DISC_NEAR_DEG; }

// 4x quadrature on one PCNT unit: channel 0 counts A edges gated by B, channel 1 counts B edges gated
// by A with the opposite sense, so both add up. Which way is "forward" is set with `cal <axis> flip`.
void setupPcnt(pcnt_unit_t unit, int a, int b) {
  pcnt_config_t cfg = {};
  cfg.pulse_gpio_num = a;
  cfg.ctrl_gpio_num = b;
  cfg.channel = PCNT_CHANNEL_0;
  cfg.unit = unit;
  cfg.pos_mode = PCNT_COUNT_INC;
  cfg.neg_mode = PCNT_COUNT_DEC;
  cfg.lctrl_mode = PCNT_MODE_KEEP;
  cfg.hctrl_mode = PCNT_MODE_REVERSE;
  cfg.counter_h_lim = 32767;
  cfg.counter_l_lim = -32768;
  pcnt_unit_config(&cfg);
  cfg.pulse_gpio_num = b;
  cfg.ctrl_gpio_num = a;
  cfg.channel = PCNT_CHANNEL_1;
  cfg.pos_mode = PCNT_COUNT_DEC;
  cfg.neg_mode = PCNT_COUNT_INC;
  pcnt_unit_config(&cfg);
  pcnt_set_filter_value(unit, DISC_PCNT_FILTER);
  pcnt_filter_enable(unit);
  pcnt_counter_pause(unit);
  pcnt_counter_clear(unit);
  pcnt_counter_resume(unit);
}
}  // namespace

float minJerk(float u) {
  u = constrain(u, 0.0f, 1.0f);
  return u * u * u * (10.0f - 15.0f * u + 6.0f * u * u);
}

void Drives::begin() {
  pinMode(PIN_DRV_NSLEEP, OUTPUT);
  pinMode(PIN_DRV_NFAULT, INPUT_PULLUP);
  for (int a = 0; a < AXES; ++a) {
    const Bridge& b = BRIDGE[a];
    mcpwm_gpio_init(b.unit, b.sa, b.in1);
    mcpwm_gpio_init(b.unit, b.sb, b.in2);
    mcpwm_config_t cfg = {};
    cfg.frequency = 20000;
    cfg.cmpr_a = 0;
    cfg.cmpr_b = 0;
    cfg.counter_mode = MCPWM_UP_COUNTER;
    cfg.duty_mode = MCPWM_DUTY_MODE_0;
    mcpwm_init(b.unit, b.timer, &cfg);
    if (isHinge((Axis)a)) analogSetPinAttenuation(ENC_PIN[a], ADC_11db);   // 0..3.1 V
    analogSetPinAttenuation(ISEN_PIN[a], ADC_0db);   // 0..0.95 V; the sense peak is 0.2 V
  }
  for (int k = 0; k < 2; ++k) setupPcnt(PCNT_OF[k], ENC_A[k], ENC_B[k]);
  analogSetPinAttenuation(PIN_VBAT_SENSE, ADC_11db);
  loadCal();
  for (int a = 0; a < AXES; ++a) {
    Axis ax = (Axis)a;
    if (isHinge(ax)) {
      // The AS5600 is on the axle, so one foot turn is one encoder turn and the travel (-2 to 182 degrees)
      // is under 360: the angle is absolute at power-up. Fold the reading into [-20, 340) around the zero.
      raw_prev_[a] = readEncoderDeg(ax);
      float d0 = raw_prev_[a] - cal_[a].zero_deg;
      float d = d0;
      while (d < -20.0f) d += 360.0f;
      while (d >= 340.0f) d -= 360.0f;
      turns_[a] = (d - d0) / 360.0f;  // so that raw + 360 turns - zero = d
      angle_[a] = cal_[a].sign * HINGE_GEAR * d;
    } else {
      // A disc starts at its lock index: the pin is engaged by default, so the disc sits on a quarter turn
      // (which quarter is not known after a power cycle; the angle starts at 0).
      pcnt_prev_[a] = readPcnt(ax);
      counts_[a] = 0;
      zero_counts_[a] = 0;
      angle_[a] = 0;
    }
    hold(ax);
  }
  enable(true);
}

void Drives::enable(bool on) { digitalWrite(PIN_DRV_NSLEEP, on ? HIGH : LOW); }

bool Drives::fault() const { return digitalRead(PIN_DRV_NFAULT) == LOW; }

void Drives::loadCal() {
  Preferences p;
  p.begin("cal", true);
  for (int a = 0; a < AXES; ++a) {
    if (isHinge((Axis)a)) cal_[a].zero_deg = p.getFloat((String(KEYS[a]) + "z").c_str(), 0.0f);
    cal_[a].sign = p.getFloat((String(KEYS[a]) + "s").c_str(), 1.0f);
  }
  p.end();
}

void Drives::calibrateZero(Axis a) {
  Preferences p;
  if (isHinge(a)) {
    cal_[a].zero_deg = readEncoderDeg(a);
    turns_[a] = 0;
    raw_prev_[a] = cal_[a].zero_deg;
    p.begin("cal", false);
    p.putFloat((String(KEYS[a]) + "z").c_str(), cal_[a].zero_deg);
    p.putFloat((String(KEYS[a]) + "s").c_str(), cal_[a].sign);
    p.end();
  } else {
    zero_counts_[a] = counts_[a];  // the disc sits on its lock index 0; nothing to store, the index is found again at each seating
  }
  angle_[a] = 0;
  hold(a);
}

void Drives::flipSign(Axis a) {
  cal_[a].sign = -cal_[a].sign;
  Preferences p;
  p.begin("cal", false);
  p.putFloat((String(KEYS[a]) + "s").c_str(), cal_[a].sign);
  p.end();
  hold(a);
}

void Drives::setLimit(Axis a, float limit_A) { limit_[a] = limit_A; }

// The lock has seated the disc on a quarter turn: that angle is exact, whatever the counts drifted to.
void Drives::reindex(Axis a, float angle_deg) {
  if (isHinge(a)) return;
  float rev = DISC_COUNTS_PER_REV;
  zero_counts_[a] = counts_[a] - (int32_t)lroundf(cal_[a].sign * angle_deg * rev / 360.0f);
  angle_[a] = angle_deg;
  q0_[a] = q1_[a] = angle_deg;
  integ_[a] = 0;
}

bool Drives::push(Axis a, float dir, float limit_A, uint32_t ms) {
  if (st_[a] == FAULTED || push_ms_[a] || ms == 0) return false;
  if (!isHinge(a) && !sleds.allIn()) return false;
  push_dir_[a] = dir >= 0 ? 1.0f : -1.0f;
  push_limit_[a] = limit_A;
  push_duty_[a] = 0;
  push_start_[a] = angle_[a];
  push_moved_[a] = 0;
  push_t0_[a] = millis();
  push_ms_[a] = ms;
  st_[a] = IDLE;
  return true;
}

float Drives::readEncoderDeg(Axis a) {
  // AS5600 OUT in analog mode (hinge axes only): 10 % to 90 % of VDD spans 0..360 degrees (default OUTS setting)
  float mv = analogReadMilliVolts(ENC_PIN[a]);
  float frac = (mv / 3300.0f - 0.1f) / 0.8f;
  return 360.0f * constrain(frac, 0.0f, 1.0f);
}

int16_t Drives::readPcnt(Axis a) {
  int16_t v = 0;
  pcnt_get_counter_value(PCNT_OF[a - DISC_T], &v);
  return v;
}

// One angle update per millisecond. Disc: the PCNT counter is 16 bits and the encoder makes at most
// about 6.6 counts per millisecond (32,800 rpm motor free speed x 12 / 60), so the difference of two
// readings, taken as int16, never wraps wrongly.
void Drives::updateAngle(Axis a) {
  if (isHinge(a)) {
    float raw = readEncoderDeg(a);
    float d = raw - raw_prev_[a];
    if (d > 180.0f) turns_[a] -= 1;
    if (d < -180.0f) turns_[a] += 1;
    raw_prev_[a] = raw;
    angle_[a] = cal_[a].sign * HINGE_GEAR * (raw + 360.0f * turns_[a] - cal_[a].zero_deg);
  } else {
    int16_t now = readPcnt(a);
    counts_[a] += (int16_t)(now - pcnt_prev_[a]);
    pcnt_prev_[a] = now;
    angle_[a] = cal_[a].sign * (float)(counts_[a] - zero_counts_[a]) * (360.0f / DISC_COUNTS_PER_REV);
  }
}

float Drives::readCurrentA(Axis a) {
  float mv = analogReadMilliVolts(ISEN_PIN[a]);
  return (mv / 1000.0f) / (isHinge(a) ? ISENSE_OHM_HINGE : ISENSE_OHM_DISC);
}

void Drives::writeDuty(Axis a, float duty) {
  duty = constrain(duty, -1.0f, 1.0f);
  const Bridge& b = BRIDGE[a];
  // Slow decay: one input held high, the other PWM'd low. |duty| = on fraction.
  float pct = 100.0f * (1.0f - fabsf(duty));
  if (duty >= 0) {
    mcpwm_set_duty(b.unit, b.timer, MCPWM_OPR_A, 100.0f);
    mcpwm_set_duty(b.unit, b.timer, MCPWM_OPR_B, pct);
  } else {
    mcpwm_set_duty(b.unit, b.timer, MCPWM_OPR_A, pct);
    mcpwm_set_duty(b.unit, b.timer, MCPWM_OPR_B, 100.0f);
  }
}

bool Drives::move(Axis a, float target_deg, float rate_dps, float limit_A) {
  if (st_[a] == FAULTED) return false;
  // Hinge deploy interlocks (2026-11-17 11:24 +0530): a disc turns only while both sleds are in (an
  // out sled stands proud of the disc plane and the +x face); a hinge swings only while its sled is
  // out, except a move that stays at 0.
  if (!isHinge(a) && !sleds.allIn()) return false;
  if (isHinge(a) && !sleds.isOut(a == HINGE_B) && (fabsf(target_deg) > STOWED_DEG || fabsf(angle_[a]) > STOWED_DEG)) return false;
  if (isHinge(a)) target_deg = constrain(target_deg, HINGE_MIN_DEG, HINGE_MAX_DEG);
  q0_[a] = angle_[a];
  start_deg_[a] = angle_[a];
  q1_[a] = target_deg;
  span_[a] = max(fabsf(target_deg - angle_[a]) / max(rate_dps, 1.0f), 0.05f);
  t0_[a] = micros() * 1e-6f;
  limit_[a] = limit_A;
  integ_[a] = 0;
  st_[a] = MOVING;
  return true;
}

void Drives::hold(Axis a) {
  push_ms_[a] = 0;  // a stop also ends a push
  q0_[a] = q1_[a] = angle_[a];
  span_[a] = 0.05f;
  t0_[a] = micros() * 1e-6f;
  limit_[a] = isHinge(a) ? HINGE_CURRENT_LIMIT : DISC_CURRENT_LIMIT;
  integ_[a] = 0;
  if (st_[a] != FAULTED) st_[a] = IDLE;
}

void Drives::stopAll() {
  for (int a = 0; a < AXES; ++a) hold((Axis)a);
}

void Drives::coast(Axis a) {
  writeDuty(a, 0);
  st_[a] = IDLE;
  limit_[a] = 0;
}

void Drives::update() {
  float now = micros() * 1e-6f;
  float vbat = 2.0f * analogReadMilliVolts(PIN_VBAT_SENSE) / 1000.0f;
  (void)vbat;  // the 6 V boost holds the motor supply; vbat is used by the safety check in main
  for (int i = 0; i < AXES; ++i) {
    Axis a = (Axis)i;
    updateAngle(a);
    current_[a] = 0.9f * current_[a] + 0.1f * readCurrentA(a);
    peak_[a] = max(peak_[a], current_[a]);
    if (push_ms_[a]) {
      push_moved_[a] = max(push_moved_[a], fabsf(angle_[a] - push_start_[a]));
      if (millis() - push_t0_[a] >= push_ms_[a]) {
        push_ms_[a] = 0;
        writeDuty(a, 0);
        hold(a);
        q0_[a] = q1_[a] = push_start_[a];  // go back to where the push began
      } else {
        // constant current: integrate the error, so the push torque is the current limit and not a duty guess
        push_duty_[a] = constrain(push_duty_[a] + PUSH_KI * (push_limit_[a] - current_[a]) * DT, 0.0f, 1.0f);
        writeDuty(a, push_dir_[a] * push_duty_[a]);
      }
      continue;
    }
    if (limit_[a] <= 0) continue;  // coasting

    float u = (now - t0_[a]) / span_[a];
    float ref = q0_[a] + (q1_[a] - q0_[a]) * minJerk(u);
    float err = ref - angle_[a];
    const Gains& g = isHinge(a) ? HINGE_GAINS : DISC_GAINS;
    integ_[a] = constrain(integ_[a] + err * DT, -20.0f, 20.0f);
    float duty = g.kp * err + g.ki * integ_[a] + g.kd * (err - prev_err_[a]) / DT;
    prev_err_[a] = err;
    if (current_[a] > limit_[a]) duty *= limit_[a] / current_[a];  // current limit (torque limit)
    writeDuty(a, duty);

    if (st_[a] == MOVING || st_[a] == RETURNING) {
      float goal_err = fabsf(q1_[a] - angle_[a]);
      if (goal_err < arriveDeg(a) && u >= 1.0f) {
        st_[a] = (st_[a] == MOVING) ? ARRIVED : RETURNED;
      } else if (u > 1.0f + STALL_S / span_[a]) {
        if (st_[a] == MOVING && goal_err < nearDeg(a)) {
          st_[a] = ARRIVED;  // stopped just short (a neighbour a little into the cell): counts as arrived
        } else if (st_[a] == MOVING) {
          float back = start_deg_[a];
          move(a, back, isHinge(a) ? HINGE_RATE_DPS : DISC_RATE_DPS, limit_[a]);
          st_[a] = RETURNING;
        } else {
          st_[a] = FAULTED;  // could not even get back: stop and report
          writeDuty(a, 0);
          limit_[a] = 0;
        }
      }
    }
  }
}
