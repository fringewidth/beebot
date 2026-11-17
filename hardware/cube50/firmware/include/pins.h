// Cube50 main board pin map (ESP32-S3-MINI-1-N8). Matches hardware/cube50/electronics/ELECTRONICS.md.
// Check every pin against the ESP32-S3-MINI-1 datasheet pin table before the board layout.
// Changed 2026-11-17 11:24 (+0530): MCP23017 now 16 of 16 (hinge deploy drive); GPIO 43/44 not reserved.
// Changed 2026-11-17 (60 mm redesign): the disc AS5600s are gone; GPIO 4 and 5 now carry the top disc's quadrature
// encoder, GPIO 33 and 26 the bottom disc's; GPIO 45 and 46 drive the two disc-lock SMA wires. Spare: 0, 43, 44.
// Pin availability was checked against the ESP32-S3-MINI-1 datasheet v1.7, table 3-1 (IO26 is free on the
// -N8, not on the -N4R2; IO33-IO37 are free on both).
#pragma once
#include <stdint.h>

// ADC1 (GPIO1-10), read with analogReadMilliVolts()
constexpr int PIN_VBAT_SENSE = 1;    // VBAT through 100k/100k: mV x 2
constexpr int PIN_ENC_HINGE_T = 2;   // AS5600 OUT, top hinge (magnet on the axle end, 1:1)
constexpr int PIN_ENC_HINGE_B = 3;   // AS5600 OUT, bottom hinge (GPIO3 is a strapping pin: floating at reset, so the AS5600 output may drive it)
constexpr int PIN_ISEN_HINGE_T = 6;  // DRV8833 #1 AISEN through 1k/1uF
constexpr int PIN_ISEN_HINGE_B = 7;  // DRV8833 #1 BISEN
constexpr int PIN_ISEN_DISC_T = 8;   // DRV8833 #2 AISEN
constexpr int PIN_ISEN_DISC_B = 9;   // DRV8833 #2 BISEN
constexpr int PIN_CELL_NTC = 10;     // 10k NTC on the cell to GND, 10k to 3V3

// Motor bridges (MCPWM unit 0: hinges, unit 1: discs)
constexpr int PIN_HINGE_T_IN1 = 11, PIN_HINGE_T_IN2 = 12;
constexpr int PIN_HINGE_B_IN1 = 13, PIN_HINGE_B_IN2 = 14;
constexpr int PIN_DISC_T_IN1 = 15, PIN_DISC_T_IN2 = 16;
constexpr int PIN_DISC_B_IN1 = 17, PIN_DISC_B_IN2 = 18;
// Disc quadrature encoders (Pololu #4761, 12 CPR, A and B open-drain-like with 10k pull-ups on the encoder board),
// counted by the PCNT peripheral (any GPIO through the matrix). One PCNT unit per disc, two channels (4x).
constexpr int PIN_ENC_DISC_T_A = 4, PIN_ENC_DISC_T_B = 5;
constexpr int PIN_ENC_DISC_B_A = 33, PIN_ENC_DISC_B_B = 26;
constexpr int PIN_DRV_NSLEEP = 21;   // all three DRV8833 and the 6 V boost enable
constexpr int PIN_DRV_NFAULT = 47;   // all three nFAULT, wired-OR, 10k pull-up
// DRV8833 #3 (hinge deploy motors) is driven on/off from the MCP23017 outputs below; GPIO 43 and 44 are free.

// USB (native): GPIO19 D-, GPIO20 D+

// I2C: LSM6DS3TR-C at 0x6A, MCP23017 at 0x20
constexpr int PIN_SDA = 38, PIN_SCL = 39;
constexpr int PIN_IMU_INT1 = 48;
// MCP23017 INTA is not wired: the switches are polled over I2C at 200 Hz (GPIO0 is a strapping pin).

// SMA channels: LEDC PWM on low-side AO3400A gates (100R series, 100k pull-down)
constexpr int PIN_SMA_T_EXT = 40;   // top disc: pins out and turned (bond)
constexpr int PIN_SMA_T_RET = 41;   // top disc: pins back and flush (release)
constexpr int PIN_SMA_B_EXT = 42;
constexpr int PIN_SMA_B_RET = 35;
constexpr int PIN_SMA_BOLT_T = 36;  // stow bolt, top foot (pull = unlatch)
constexpr int PIN_SMA_BOLT_B = 37;
// Disc lock wires (0.15 mm, 52 mm). GPIO45 and 46 are strapping pins read at reset: both default low, which is what
// the gate pull-downs hold, so the wires are off while the chip boots. Keep the 100k pull-down on each gate.
constexpr int PIN_SMA_LOCK_T = 45;  // top disc lock pin released (pull = unlock)
constexpr int PIN_SMA_LOCK_B = 46;

constexpr int PIN_LED = 34;  // status LED (green) to GND through 1k

// MCP23017 (0x20): 16 of 16 pins (changed 2026-11-17 11:24 +0530 for the hinge deploy drive).
// Bit i of the 16-bit switch word is GPA0..7 (0..7) then GPB0..7 (8..15). GPA7 and GPB7 are
// output-only on current silicon (datasheet DS20001952D, 2022), so they carry deploy outputs and the
// bottom bolt switch moved from GPA7 to GPB0.
enum McpPin : uint8_t {
  MCP_SW_EXT_T = 0,     // GPA0 in: top disc ring at "extended" (switch to VBAT through the disc tracks; 47k/100k divider; no pull-up)
  MCP_SW_RET_T = 1,     // GPA1 in: top disc ring at "retracted"
  MCP_SW_EXT_B = 2,     // GPA2 in
  MCP_SW_RET_B = 3,     // GPA3 in
  MCP_STOW_T = 4,       // GPA4 in: top foot seated on the core (tact switch to GND, pressed = stowed)
  MCP_STOW_B = 5,       // GPA5 in
  MCP_BOLT_T = 6,       // GPA6 in: top stow bolt up (latched)
  MCP_DEP_T_IN1 = 7,    // GPA7 out: DRV8833 #3 AIN1, top deploy motor
  MCP_BOLT_B = 8,       // GPB0 in: bottom stow bolt up
  MCP_SLED_OUT_T = 9,   // GPB1 in: top hinge sled at its out end (axle on the edge line)
  MCP_SLED_IN_T = 10,   // GPB2 in: top hinge sled at its in end (nothing proud)
  MCP_SLED_OUT_B = 11,  // GPB3 in
  MCP_SLED_IN_B = 12,   // GPB4 in
  MCP_DEP_T_IN2 = 13,   // GPB5 out: DRV8833 #3 AIN2
  MCP_DEP_B_IN1 = 14,   // GPB6 out: DRV8833 #3 BIN1, bottom deploy motor
  MCP_DEP_B_IN2 = 15,   // GPB7 out: DRV8833 #3 BIN2
};
constexpr uint16_t mcpBit(McpPin p) { return (uint16_t)(1u << p); }
// Outputs (IODIR 0), switches to VBAT (no pull-up, not inverted), switches to GND (pull-up, inverted: closed = 1)
constexpr uint16_t MCP_OUTPUTS = mcpBit(MCP_DEP_T_IN1) | mcpBit(MCP_DEP_T_IN2) | mcpBit(MCP_DEP_B_IN1) | mcpBit(MCP_DEP_B_IN2);
constexpr uint16_t MCP_TO_VBAT = mcpBit(MCP_SW_EXT_T) | mcpBit(MCP_SW_RET_T) | mcpBit(MCP_SW_EXT_B) | mcpBit(MCP_SW_RET_B);
constexpr uint16_t MCP_TO_GND = (uint16_t)~(MCP_OUTPUTS | MCP_TO_VBAT);
