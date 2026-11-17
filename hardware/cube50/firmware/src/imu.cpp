#include "imu.h"

#include <Wire.h>

Imu imu;

namespace {
constexpr uint8_t ADDR = 0x6A;
constexpr uint8_t WHO_AM_I = 0x0F, CTRL1_XL = 0x10, CTRL2_G = 0x11, CTRL3_C = 0x12, OUTX_L_G = 0x22;
constexpr float G_PER_LSB = 0.061e-3f;    // +/-2 g
constexpr float DPS_PER_LSB = 17.50e-3f;  // +/-500 dps

void write8(uint8_t reg, uint8_t v) {
  Wire.beginTransmission(ADDR);
  Wire.write(reg);
  Wire.write(v);
  Wire.endTransmission();
}

int read8(uint8_t reg) {
  Wire.beginTransmission(ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return -1;
  if (Wire.requestFrom((int)ADDR, 1) != 1) return -1;
  return Wire.read();
}
}  // namespace

bool Imu::begin() {
  if (read8(WHO_AM_I) != 0x6A) return ok_ = false;
  write8(CTRL3_C, 0x44);   // BDU, auto-increment
  write8(CTRL1_XL, 0x60);  // 416 Hz, +/-2 g
  write8(CTRL2_G, 0x64);   // 416 Hz, 500 dps
  delay(20);
  return ok_ = read();
}

bool Imu::read() {
  Wire.beginTransmission(ADDR);
  Wire.write(OUTX_L_G);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)ADDR, 12) != 12) return false;
  int16_t raw[6];
  for (int i = 0; i < 6; ++i) raw[i] = (int16_t)(Wire.read() | (Wire.read() << 8));
  for (int i = 0; i < 3; ++i) {
    gyro_[i] = raw[i] * DPS_PER_LSB - bias_[i];
    acc_[i] = raw[i + 3] * G_PER_LSB;
  }
  return true;
}

void Imu::calibrateGyro(int samples) {
  float sum[3] = {0};
  for (int k = 0; k < samples; ++k) {
    read();
    for (int i = 0; i < 3; ++i) sum[i] += gyro_[i] + bias_[i];
    delay(3);
  }
  for (int i = 0; i < 3; ++i) bias_[i] = sum[i] / samples;
}

void Imu::update(float dt) {
  if (!ok_ || !read()) return;
  float ar = atan2f(acc_[1], acc_[2]) * RAD_TO_DEG;
  float ap = atan2f(-acc_[0], sqrtf(acc_[1] * acc_[1] + acc_[2] * acc_[2])) * RAD_TO_DEG;
  constexpr float K = 0.98f;
  roll_ = K * (roll_ + gyro_[0] * dt) + (1 - K) * ar;
  pitch_ = K * (pitch_ + gyro_[1] * dt) + (1 - K) * ap;
  yaw_ += gyro_[2] * dt;
}
