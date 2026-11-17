#include "radio.h"

#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

namespace {
uint8_t host_mac[6] = {0};
bool have_host = false;
FrameHandler handler_ = nullptr;
int64_t clock_offset_us = 0;
const uint8_t BROADCAST[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

void addPeer(const uint8_t* mac) {
  if (esp_now_is_peer_exist(mac)) return;
  esp_now_peer_info_t p = {};
  memcpy(p.peer_addr, mac, 6);
  p.channel = 1;
  p.encrypt = false;
  esp_now_add_peer(&p);
}

void onRecv(const uint8_t* mac, const uint8_t* data, int len) {
  if (len < (int)sizeof(Header)) return;
  const Header* h = reinterpret_cast<const Header*>(data);
  if (h->magic != PROTO_MAGIC || h->version != PROTO_VERSION) return;
  if (handler_) handler_(mac, data, len);
}
}  // namespace

namespace radio {

bool begin(FrameHandler handler) {
  handler_ = handler;
  WiFi.mode(WIFI_STA);
  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
  if (esp_now_init() != ESP_OK) return false;
  esp_now_register_recv_cb(onRecv);
  addPeer(BROADCAST);
  return true;
}

uint8_t cubeId() {
  uint8_t mac[6];
  WiFi.macAddress(mac);
  return mac[5] == 0xFF ? 0xFE : mac[5];
}

void rememberHost(const uint8_t* mac) {
  memcpy(host_mac, mac, 6);
  have_host = true;
  addPeer(mac);
}

bool send(const uint8_t* mac, const void* frame, size_t len) {
  if (len > 250) return false;
  return esp_now_send(mac ? mac : BROADCAST, (const uint8_t*)frame, len) == ESP_OK;
}

bool sendToHost(uint8_t type, uint16_t seq, const void* payload, size_t len) {
  uint8_t buf[250];
  Header h = {PROTO_MAGIC, PROTO_VERSION, type, cubeId(), seq, 0};
  if (sizeof(h) + len > sizeof(buf)) return false;
  memcpy(buf, &h, sizeof(h));
  if (len) memcpy(buf + sizeof(h), payload, len);
  return send(have_host ? host_mac : nullptr, buf, sizeof(h) + len);
}

void onSync(uint64_t host_us) { clock_offset_us = (int64_t)host_us - (int64_t)esp_timer_get_time(); }

uint64_t syncedMicros() { return (uint64_t)((int64_t)esp_timer_get_time() + clock_offset_us); }

}  // namespace radio
