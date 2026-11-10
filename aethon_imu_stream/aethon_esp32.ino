/* ============================================================
 *  AETHON-X  ·  IMU BLE STREAMER  (ESP32 BLE GATT notify)
 *  ESP32 sketch — feeds vyuha_bridge.py / aethon_visualizer via BLE.
 *
 *  WHY BLE INSTEAD OF CLASSIC SPP:
 *    This sketch used to advertise Classic Bluetooth SPP (macOS saw it
 *    as /dev/tty.AETHON-X). On macOS Tahoe (26.5) that SPP session
 *    opens fine but silently delivers zero bytes — a bug in macOS's
 *    RFCOMM stack, confirmed by the firmware's telemetry being clean
 *    over USB Serial the whole time. BLE goes through CoreBluetooth, a
 *    completely different stack on macOS, and needs no manual "pair in
 *    System Settings" step — the bridge connects to it by name
 *    programmatically. The wire *format* (13-field CSV) is unchanged;
 *    only how the bytes get from this board to the Mac has changed.
 *
 *  HARDWARE WIRING (breadboard) — unchanged:
 *
 *     MPU6050  -->  ESP32
 *       VCC   ->  3.3V   (MPU6050 runs on 3.3V — do NOT use 5V)
 *       GND   ->  GND
 *       SCL   ->  GPIO 22
 *       SDA   ->  GPIO 21
 *       INT   ->  (not connected)
 *
 *     NO external Bluetooth module needed — ESP32 has BT built-in.
 *     Power the ESP32 from a USB power bank via its USB-C / micro-USB port.
 *
 *  FIRST-TIME USE:
 *    1. Flash this sketch.
 *    2. No pairing step needed. Run the bridge with BLE mode:
 *         python3 vyuha_bridge.py --ble
 *       (it scans for a device advertising the name "AETHON-X" and
 *       connects automatically — nothing to do in System Settings.)
 *    3. To sanity-check the firmware alone before touching the bridge,
 *       any generic BLE scanner app (nRF Connect, LightBlue) will show
 *       "AETHON-X" advertising the Nordic UART Service and let you
 *       subscribe to the TX characteristic to watch raw CSV frames.
 *
 *  ARDUINO IDE SETUP:
 *    Board   : "ESP32 Dev Module" (or your specific ESP32 board variant)
 *    Library : BLEDevice / BLEServer / BLEUtils / BLE2902 — included in
 *              the ESP32 Arduino core, no install needed (same core as
 *              BluetoothSerial was).
 *    Partition scheme: Use "Default" or "Huge APP" if you get low-memory errors.
 *
 *  OUTPUT FORMAT (13 CSV fields):
 *    angleX,angleY,angleZ,gxR,gyR,gzR,axg,ayg,waz,thrX,thrY,thrZ,landed\n
 *    • angleX/Y/Z : complementary-filter fused angles (degrees)
 *    • gxR/gyR/gzR: bias-corrected gyro rates (deg/s)
 *    • axg/ayg    : raw body-frame accelerometer (g)
 *    • waz        : world-frame vertical acceleration (g); 1.0 = hover
 *    • thrX/Y/Z   : body-frame thrust vector components (g)
 *    • landed     : 1 if still for >1.5 s, else 0
 *
 *  BAUD NOTE:
 *    BluetoothSerial baud is irrelevant for the BT link — SPP tunnels
 *    data over the radio at full speed regardless. The 115200 baud below
 *    only applies to the USB Serial monitor. Set --baud 115200 in Python.
 * ============================================================ */

#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ── BLE GATT setup ────────────────────────────────────────────
// Nordic UART Service (NUS) UUIDs — a widely-used de facto standard
// for streaming text over a BLE notify characteristic. Using the
// well-known UUIDs (rather than inventing new ones) means generic
// tools like nRF Connect / LightBlue recognize this as a UART-style
// service out of the box, which is handy for bench-testing this
// sketch on its own before vyuha_bridge.py is even involved.
// *** These UUIDs must match BLE_SERVICE_UUID / BLE_TX_CHAR_UUID in
// *** vyuha_bridge.py exactly.
#define SERVICE_UUID        "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define TX_CHARACTERISTIC_UUID "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

const char* BT_NAME = "AETHON-X";   // BLE advertised device name

BLEServer* bleServer = nullptr;
BLECharacteristic* txCharacteristic = nullptr;
volatile bool deviceConnected = false;

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* server) override {
    deviceConnected = true;
    Serial.println(F("AETHON-X: BLE central connected."));
  }
  void onDisconnect(BLEServer* server) override {
    deviceConnected = false;
    Serial.println(F("AETHON-X: BLE central disconnected — resuming advertising."));
    // The ESP32 BLE library stops advertising once a central connects;
    // without restarting it here, a dropped connection would never be
    // reconnectable without a manual reboot of the board.
    delay(200);
    server->startAdvertising();
  }
};

// ── Serial baud (USB monitor only — BT speed is automatic) ──
const uint32_t USB_BAUD = 115200;

// ── MPU6050 ─────────────────────────────────────────────────
const uint8_t MPU_ADDR  = 0x68;
const float   DT_TARGET = 0.01f;   // 100 Hz loop target
const float   ALPHA     = 0.98f;   // complementary filter weight (gyro trust)

// ── I²C pins (ESP32 default) ─────────────────────────────────
// GPIO 21 = SDA, GPIO 22 = SCL — Wire uses these automatically on ESP32.
// If your board variant differs, call Wire.begin(SDA_PIN, SCL_PIN) instead.

float angleX = 0, angleY = 0, angleZ = 0;
float gyroXoff = 0, gyroYoff = 0, gyroZoff = 0;
uint32_t lastMicros = 0;
uint32_t stillFrames = 0;    // counts consecutive still frames for landed detection


// ── MPU6050 helpers ─────────────────────────────────────────
void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

void mpuRead(int16_t &ax, int16_t &ay, int16_t &az,
             int16_t &gx, int16_t &gy, int16_t &gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);                         // ACCEL_XOUT_H
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)14);
  ax = (Wire.read() << 8) | Wire.read();
  ay = (Wire.read() << 8) | Wire.read();
  az = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read();                 // skip temperature bytes
  gx = (Wire.read() << 8) | Wire.read();
  gy = (Wire.read() << 8) | Wire.read();
  gz = (Wire.read() << 8) | Wire.read();
}


// ── Gyro calibration (keep board still for ~1 s on power-up) ─
void calibrateGyro() {
  const int N = 500;
  long sx = 0, sy = 0, sz = 0;
  int16_t ax, ay, az, gx, gy, gz;
  for (int i = 0; i < N; i++) {
    mpuRead(ax, ay, az, gx, gy, gz);
    sx += gx; sy += gy; sz += gz;
    delay(2);
  }
  gyroXoff = sx / (float)N;
  gyroYoff = sy / (float)N;
  gyroZoff = sz / (float)N;
}


// ── Setup ────────────────────────────────────────────────────
void setup() {
  Serial.begin(USB_BAUD);

  // Start BLE — advertises as "AETHON-X" with one notify characteristic
  BLEDevice::init(BT_NAME);
  bleServer = BLEDevice::createServer();
  bleServer->setCallbacks(new ServerCallbacks());

  BLEService* service = bleServer->createService(SERVICE_UUID);
  txCharacteristic = service->createCharacteristic(
      TX_CHARACTERISTIC_UUID,
      BLECharacteristic::PROPERTY_NOTIFY
  );
  // Required for notify to actually work with most BLE centrals
  // (including bleak/CoreBluetooth) — without a CCCD descriptor the
  // central has nothing to write to enable notifications.
  txCharacteristic->addDescriptor(new BLE2902());
  service->start();

  BLEAdvertising* advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->setScanResponse(true);
  BLEDevice::startAdvertising();

  Serial.print(F("AETHON-X: BLE advertising started, device name = "));
  Serial.println(BT_NAME);

  // I²C — ESP32 default pins: SDA=21, SCL=22
  Wire.begin();
  Wire.setClock(400000);        // 400 kHz fast-mode I2C

  mpuWrite(0x6B, 0x00);  // PWR_MGMT_1  — wake MPU6050
  mpuWrite(0x1B, 0x08);  // GYRO_CONFIG — ±500 dps  (65.5 LSB/dps)
  mpuWrite(0x1C, 0x10);  // ACCEL_CONFIG— ±8 g      (4096 LSB/g)
  delay(100);

  Serial.println(F("AETHON-X: calibrating gyro — keep board still..."));
  calibrateGyro();
  Serial.println(F("AETHON-X: calibration done. Advertising over BLE as 'AETHON-X'."));
  Serial.println(F("          No pairing needed — run: python3 vyuha_bridge.py --ble"));

  lastMicros = micros();
}


// ── Main loop ────────────────────────────────────────────────
void loop() {
  int16_t ax, ay, az, gx, gy, gz;
  uint32_t now = micros();      // timestamp BEFORE I²C read for accurate dt
  mpuRead(ax, ay, az, gx, gy, gz);

  float dt = (now - lastMicros) * 1e-6f;
  if (dt <= 0 || dt > 0.5f) dt = DT_TARGET;   // guard: micros() wrap or stall
  lastMicros = now;

  // ── Accelerometer in g ───────────────────────────────────
  float axg = ax / 4096.0f;
  float ayg = ay / 4096.0f;
  float azg = az / 4096.0f;

  // ── Accel magnitude guard ────────────────────────────────
  // Only trust the accelerometer for tilt correction when |a| ≈ 1g.
  // Linear motion (horizontal push, lift jerk) shifts |a| away from 1g —
  // freezing the accel blend during those moments prevents false tilt readings.
  float aMag = sqrt(axg*axg + ayg*ayg + azg*azg);

  // ── Accel-derived pitch & roll ───────────────────────────
  float accPitch = atan2(-axg, sqrt(ayg * ayg + azg * azg)) * 57.2957795f;
  float accRoll  = atan2(-ayg, azg)                          * 57.2957795f;

  // ── Gyro rates (deg/s, bias-corrected) ──────────────────
  float gxR = (gx - gyroXoff) / 65.5f;
  float gyR = (gy - gyroYoff) / 65.5f;
  float gzR = -((gz - gyroZoff) / 65.5f);

  // ── Complementary filter ─────────────────────────────────
  // blend = 0 during linear motion (|a| outside 0.85–1.15 g band).
  float blend = (aMag > 0.85f && aMag < 1.15f) ? (1.0f - ALPHA) : 0.0f;
  angleX = ALPHA * (angleX + gxR * dt) + blend * accPitch;
  angleY = ALPHA * (angleY + gyR * dt) + blend * accRoll;
  angleZ += gzR * dt;   // yaw: gyro-only (no magnetometer on MPU6050)

  // ── World-frame vertical acceleration (waz) ──────────────
  // Rotates the body-frame accel into world frame using fused pitch & roll,
  // then projects onto the world vertical axis.
  // waz = 1.0 g at hover/rest (gravity baseline), > 1.0 = ascending thrust,
  // < 1.0 = descending. Algebraically proved to always equal exactly 1.0
  // under pure gravity regardless of attitude — linear motion is the only
  // thing that shifts it, which is exactly what we want to detect.
  float pRad    = angleX * 0.017453f;
  float rRad    = angleY * 0.017453f;
  float sinP    = sin(pRad), cosP = cos(pRad);
  float sinR    = sin(rRad), cosR = cos(rRad);
  float waz     = -sinP * axg
                  + cosP * sinR * (-ayg)
                  + cosP * cosR * azg;

  // ── Thrust vector (body-frame, scaled by waz) ───────────────
  // Projects the drone's "up" axis into world frame components,
  // then scales by waz so magnitude reflects actual thrust level.
  float thrX =  sinP         * waz;   // forward/back component
  float thrY = -cosP * sinR  * waz;   // left/right component
  float thrZ =  cosP * cosR  * waz;   // vertical component

  // ── Landed detection ─────────────────────────────────────────
  // Drone is "landed" when gyro rates AND angles are all near-zero
  // for more than 150 consecutive frames (~1.5 s at 100 Hz).
  bool isStill = (abs(gxR)    < 2.0f && abs(gyR)    < 2.0f && abs(gzR)    < 2.0f
               && abs(angleX) < 3.0f && abs(angleY) < 3.0f);
  stillFrames  = isStill ? stillFrames + 1 : 0;
  uint8_t landed = (stillFrames > 150) ? 1 : 0;

  // ── Transmit 13-field CSV over BLE notify ─────────────────────
  // Same format as before (see OUTPUT FORMAT above) — only the
  // transport changed. One notify() call per frame; a single CSV
  // line here is ~70 bytes, comfortably inside the MTU BLE
  // negotiates with bleak/CoreBluetooth (typically 185+ bytes), so
  // no chunking is needed.
  if (deviceConnected) {
    char frame[96];
    snprintf(frame, sizeof(frame),
             "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%u\n",
             angleX, angleY, angleZ, gxR, gyR, gzR,
             axg, ayg, waz, thrX, thrY, thrZ, landed);
    txCharacteristic->setValue((uint8_t*)frame, strlen(frame));
    txCharacteristic->notify();
  }

  // ── Mirror to USB serial (for Arduino IDE Serial Monitor) ────
  Serial.print(angleX, 2); Serial.print(',');
  Serial.print(angleY, 2); Serial.print(',');
  Serial.print(angleZ, 2); Serial.print(',');
  Serial.print(gxR,    2); Serial.print(',');
  Serial.print(gyR,    2); Serial.print(',');
  Serial.print(gzR,    2); Serial.print(',');
  Serial.print(axg,    3); Serial.print(',');
  Serial.print(ayg,    3); Serial.print(',');
  Serial.print(waz,    3); Serial.print(',');
  Serial.print(thrX,   3); Serial.print(',');
  Serial.print(thrY,   3); Serial.print(',');
  Serial.print(thrZ,   3); Serial.print(',');
  Serial.println(landed);

  // ── Pace loop to ~100 Hz ─────────────────────────────────
  // delay() yields to FreeRTOS between iterations — the BLE stack's
  // own tasks (connection event handling, notify queueing) run during
  // this window, same reasoning as when this was Classic SPP. A bare
  // spin-wait would burn 100% CPU for the full 10 ms slice and starve
  // those tasks.
  uint32_t elapsed_ms = (micros() - now) / 1000;
  if (elapsed_ms < 10) delay(10 - elapsed_ms);
}
