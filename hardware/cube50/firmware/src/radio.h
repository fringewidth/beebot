// ESP-NOW link to the host. Cubes and the gateway share channel 1. Frames are protocol.h structs.
#pragma once
#include <Arduino.h>

#include "protocol.h"

using FrameHandler = void (*)(const uint8_t* mac, const uint8_t* data, int len);

namespace radio {
bool begin(FrameHandler handler);
uint8_t cubeId();                                     // low byte of the station MAC (unique per chip in a small fleet; check for clashes)
bool send(const uint8_t* mac, const void* frame, size_t len);  // mac nullptr = broadcast
bool sendToHost(uint8_t type, uint16_t seq, const void* payload, size_t len);
void rememberHost(const uint8_t* mac);
uint64_t syncedMicros();                              // host clock estimate (MSG_SYNC)
void onSync(uint64_t host_us);
}  // namespace radio
