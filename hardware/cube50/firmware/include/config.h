// Cube50 limits and gains. Numbers marked (bench) must be set from the bench tests in ASSEMBLY.md.
// Changed 2026-11-17 11:24 (+0530): hinge deploy drive constants (DEPLOY_*, STOWED_DEG).
// Changed 2026-11-17 (60 mm redesign): disc drive is a Pololu #3079 with a #4761 quadrature encoder and a
// quarter-turn lock (disc AS5600s removed); hinge AS5600 reads the axle 1:1; 8 SMA channels (2 lock wires);
// deploy timeout for a 6 s stroke. Each number says where it comes from: (source), (derived) or (guess).
#pragma once
#include <stdint.h>

// Drive trains (DESIGN.md "Torque budget")
// Hinge: the AS5600 sits on the axle end, so the encoder angle is the foot angle. The train is 2.083:1
// (12:25 then 14:16:14), 0.438 N m at the foot, 14.9 rpm = 89 deg/s free (derived, DESIGN.md).
constexpr float HINGE_GEAR = 1.0f;               // foot degrees per encoder degree
constexpr float HINGE_MAX_DEG = 182.0f;          // mechanical stop at 185
constexpr float HINGE_MIN_DEG = -2.0f;           // stowed is 0
constexpr float HINGE_RATE_DPS = 50.0f;          // reference rate for pivots: 56 % of the 89 deg/s free speed (guess; 90 was the old train's)

// Disc: Pololu #3079 (HPCB 6 V, 297.92:1, extended back shaft) with a #4761 magnetic encoder, 12 counts per
// motor revolution counting both edges of both channels (source: pololu.com/product/4761, "multiply the
// gear ratio by 12"), then the external 14.29:1 train. The PCNT counts all four edges, so
// counts per disc turn = 12 x 297.92 x 14.29 = 51,087 (derived), 0.00705 deg per count. The task text's
// "12 x 4 x 298 x 14.29" would double count: 12 CPR is already the 4x figure.
constexpr float DISC_ENC_CPR_MOTOR = 12.0f;      // counts per motor revolution, 4x (source)
constexpr float DISC_GEARBOX_RATIO = 297.92f;    // source: pololu.com/product/3079 ("297.92:1")
constexpr float DISC_EXTERNAL_RATIO = 14.29f;    // source: mechanical design (set the exact tooth ratio when the CAD freezes)
constexpr float DISC_COUNTS_PER_REV = DISC_ENC_CPR_MOTOR * DISC_GEARBOX_RATIO * DISC_EXTERNAL_RATIO;
constexpr float DISC_FREE_DPS = 46.0f;           // 0.80 rad/s free speed at the disc (derived: 110 rpm / 14.29)
constexpr float DISC_RATE_DPS = 30.0f;           // default disc rate: 65 % of free speed (guess)
constexpr float DISC_QUARTER_DEG = 90.0f;        // the lock pin holds the disc at multiples of this
constexpr uint32_t DISC_PCNT_FILTER = 250;       // PCNT glitch filter, APB cycles (3 us at 80 MHz; the encoder edges are >= 150 us apart at free speed) (guess)

// Quarter-turn disc lock, one per foot (spring pin, engaged by default; released by a 0.15 mm SMA wire).
// Turn sequence: heat the wire, start turning, stop heating once the disc has moved LOCK_RELEASE_DEG (the
// pin rides on the ring), turn to the target quarter, where the spring snaps the pin in; then check by
// pushing at LOCK_TEST_CURRENT_A and requiring no motion.
constexpr float LOCK_RELEASE_DEG = 5.0f;         // source: task spec
constexpr uint32_t LOCK_RELEASE_TIMEOUT_MS = 2000;  // the disc must have moved LOCK_RELEASE_DEG by then, else abort (guess, as SMA_TIMEOUT_MS)
constexpr float DISC_LOCK_RELEASE_LIMIT_A = 0.60f;  // current limit until the pin has cleared: 1.4 N m at the disc, below the 2.4 N m cap, so a stuck pin is not rammed (guess)
constexpr uint32_t LOCK_SEAT_MS = 400;           // wait at the target for the wire to cool and the spring to seat the pin (guess)
constexpr float LOCK_CAPTURE_DEG = 6.0f;         // at the end the disc must be within this of the quarter, else DR_MISSED (guess)
constexpr bool LOCK_VERIFY_AFTER_TURN = true;    // push check after every turn (costs LOCK_TEST_MS)
// 0.3 N m at the disc: torque = 3.04 N m/A x (I - 0.15 A), from 2.40 N m at 0.94 A and the 0.15 A free-run current (derived)
constexpr float LOCK_TEST_TORQUE_NM = 0.30f;
constexpr float LOCK_TEST_CURRENT_A = 0.15f + LOCK_TEST_TORQUE_NM / 3.04f;  // 0.249 A
constexpr uint32_t LOCK_TEST_MS = 400;
constexpr float LOCK_TEST_MAX_DEG = 1.0f;        // the encoder is on the motor side: gear backlash shows up here, so this is set from the bench (guess)

// Current limits (A at the motor). DRV8833 chopping sets a hard ceiling via the sense resistors
// (I_trip = 0.2 V / R: hinge 0.33 R -> 0.6 A, disc 0.215 R -> 0.93 A); these are the firmware limits under it.
constexpr float HINGE_CURRENT_LIMIT = 0.40f;  // 2.5 kg cm at the 1000:1 gearbox (Pololu's limit): 0.1 + 1.5 x 0.245 / 1.18 A; x 2.083 x 0.86 = 0.438 N m at the foot (derived)
constexpr float DISC_CURRENT_LIMIT = 0.94f;   // 2.40 N m at the disc (derived: Pololu 298:1 stall 3.4 kg cm at 1.5 A, 0.15 A free-run, x 14.29 x 0.86)
constexpr float ISENSE_OHM_HINGE = 0.33f;
constexpr float ISENSE_OHM_DISC = 0.215f;     // two 0.43 R 0805 in parallel: chop at 0.2 V / 0.215 R = 0.93 A (the DRV8833 trip voltage has a tolerance)

// Position loop, 1 kHz, output is PWM duty in [-1, 1]
struct Gains { float kp, ki, kd; };
constexpr Gains HINGE_GAINS = {0.04f, 0.02f, 0.0008f};  // per degree (bench)
constexpr Gains DISC_GAINS = {0.03f, 0.01f, 0.0006f};   // per degree (bench; the disc loop is slower now: 46 deg/s free)
constexpr float ARRIVE_DEG = 3.0f;      // hinge arrived (cube_hinge's tolerance)
constexpr float NEAR_DEG = 10.0f;       // a hinge stall this close to the goal counts as arrived
constexpr float DISC_ARRIVE_DEG = 1.0f; // the lock pin needs the disc close to the quarter to seat (guess)
constexpr float DISC_NEAR_DEG = 3.0f;   // a disc stall this close counts as arrived; not looser, a disc must seat its pin (guess)
constexpr float STALL_S = 0.5f;         // past the reference's end before driving back

// Motor supply: 6.0 V boost; PWM caps keep the average at or under the motors' 6 V rating.
constexpr float MOTOR_SUPPLY_V = 6.0f;

// SMA (0.15 mm wire, 55 ohm/m): heating power target per wire, duty is computed from VBAT and wire resistance.
// Eight channels: 4 ring wires, 2 stow bolts, 2 disc locks.
constexpr float SMA_TARGET_W = 0.55f;          // about 0.34 A in a 4.7 ohm wire (bench)
constexpr float SMA_RING_OHM = 4.7f;           // 85 mm at 55 ohm/m
constexpr float SMA_BOLT_OHM = 2.8f;           // 50 mm
constexpr float SMA_LOCK_OHM = 2.9f;           // 52 mm (2.86 ohm derived), source: task spec
constexpr float SMA_DUTY_MAX = 0.40f;          // PWM duty cap: with the staggered phases below the pulses cannot wrap past the LEDC period (guess)
constexpr int SMA_MAX_ACTIVE = 3;              // wires heating at once: bounds the cell's pulse current (ELECTRONICS.md "Power budget"); more are refused (guess)
constexpr uint32_t SMA_LOCK_COOL_MS = 1500;    // minimum off time before a lock wire heats again (0.15 mm wire cools in about 1 s in air; guess)
constexpr uint32_t SMA_TIMEOUT_MS = 2000;      // give up and report
constexpr uint32_t SMA_COOL_MS = 3000;         // minimum off time before the same wire heats again

// Battery (1S LiPo)
constexpr float VBAT_LOW_V = 3.45f;     // finish the current move, then refuse new ones
constexpr float VBAT_CUTOFF_V = 3.30f;  // stop all drives, deep sleep
constexpr float CELL_MAX_C = 50.0f;

// Hinge deploy drive (changed 2026-11-17 11:24 +0530): one N20 1:150 gearmotor with an M3 lead screw
// per hinge drives its sled 7.9 mm between two end switches, on/off through DRV8833 #3 (chop limit
// 0.47 A = the motor's stall current, set by 0.43 R). The screw self-locks, so nothing is powered at rest.
constexpr uint32_t DEPLOY_TIMEOUT_MS = 15000;  // one stroke takes 6 s at a 0.7 mm lead, 9.7 s at 0.5 mm (derived, 98 rpm free, 7.9 mm); 1.5x the slower case. Bench: set from test 3b
constexpr uint32_t DEPLOY_LEAVE_MS = 2000;     // the start switch must open within this, else the motor is stalled or wired backwards
constexpr uint32_t DEPLOY_DEBOUNCE_MS = 15;    // a sled switch counts once it reads the same for this long (3 polls at 200 Hz)
constexpr uint32_t DEPLOY_BRAKE_MS = 60;       // brake after the end switch closes, then coast
constexpr bool DEPLOY_REVERSED[2] = {false, false};  // top, bottom: set if "out" drives the sled in (motor leads swapped)
constexpr float STOWED_DEG = 2.0f;             // a hinge within this of 0 counts as home

// Telemetry
constexpr uint32_t TELEMETRY_MS = 50;   // 20 Hz
constexpr uint32_t LINK_TIMEOUT_MS = 1000;  // no command for this long while moving: stop and hold
