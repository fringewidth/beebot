// LSM6DS3TR-C 6-axis IMU over I2C (0x6A): 416 Hz, +/-2 g, +/-500 dps. Roll and pitch from a
// complementary filter; yaw is gyro-integrated and drifts (the overhead camera gives absolute yaw).
#pragma once
#include <Arduino.h>

class Imu {
 public:
  bool begin();
  void update(float dt);  // call at 200-400 Hz
  float roll() const { return roll_; }
  float pitch() const { return pitch_; }
  float yaw() const { return yaw_; }
  const float* acc() const { return acc_; }  // g
  const float* gyro() const { return gyro_; }  // deg/s
  bool ok() const { return ok_; }
  void calibrateGyro(int samples = 400);  // cube at rest

 private:
  bool read();
  bool ok_ = false;
  float acc_[3] = {0, 0, 1}, gyro_[3] = {0}, bias_[3] = {0};
  float roll_ = 0, pitch_ = 0, yaw_ = 0;
};

extern Imu imu;
