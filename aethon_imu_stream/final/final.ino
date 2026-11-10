/* ============================================================
 *  AETHON-X  ·  IMU SERIAL STREAMER
 *  Arduino sketch — feeds the Processing/Python visualiser.
 *
 *  HARDWARE (breadboard wiring):
 *     MPU6050  -->  Arduino UNO / Nano
 *       VCC   ->  3.3V  (or 5V if module has regulator)
 *       GND   ->  GND
 *       SCL   ->  A5  (UNO) / D21 (Mega)
 *       SDA   ->  A4  (UNO) / D20 (Mega)
 *       INT   ->  (not used)
 *
 *  OUTPUT FORMAT  (matches AETHON_X_DroneDemo_P3D_v6.pde parser):
 *     "<angleX>,<angleY>,<angleZ>,<gxR>,<gyR>,<gzR>\n"  @ 115200 baud
 *     angles in degrees (complementary-filter fused);
 *     gyro rates in deg/s (bias-corrected, not integrated).
 *
 *  LIBRARY: none required — talks to MPU6050 over raw Wire (I2C).
 * ============================================================ */

#include <Wire.h>

const uint8_t  MPU_ADDR = 0x68;
const uint32_t BAUD     = 115200;
const float    DT_TARGET= 0.01f;     // 100 Hz loop
const float    ALPHA    = 0.98f;     // complementary filter

float angleX = 0, angleY = 0, angleZ = 0;
float gyroXoff = 0, gyroYoff = 0, gyroZoff = 0;
uint32_t lastMicros = 0;

void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg); Wire.write(val);
  Wire.endTransmission();
}

void mpuRead(int16_t &ax, int16_t &ay, int16_t &az,
             int16_t &gx, int16_t &gy, int16_t &gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);                      // ACCEL_XOUT_H
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)14);
  ax = (Wire.read() << 8) | Wire.read();
  ay = (Wire.read() << 8) | Wire.read();
  az = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read();              // skip temperature
  gx = (Wire.read() << 8) | Wire.read();
  gy = (Wire.read() << 8) | Wire.read();
  gz = (Wire.read() << 8) | Wire.read();
}

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

void setup() {
  Serial.begin(BAUD);
  Wire.begin();
  Wire.setClock(400000);

  mpuWrite(0x6B, 0x00);   // PWR_MGMT_1   wake up
  mpuWrite(0x1B, 0x08);   // GYRO_CONFIG  ±500 dps  (65.5 LSB/dps)
  mpuWrite(0x1C, 0x10);   // ACCEL_CONFIG ±8 g     (4096 LSB/g)
  delay(100);

  calibrateGyro();
  lastMicros = micros();
}

void loop() {
  // FIX 1: capture timestamp AFTER the I2C read so that the ~400–500 µs
  // bus transaction is included in dt, eliminating the ~5 % angle drift
  // that occurred when "now" was sampled before mpuRead().
  int16_t ax, ay, az, gx, gy, gz;
  mpuRead(ax, ay, az, gx, gy, gz);
  uint32_t now = micros();             // post-read timestamp

  float dt = (now - lastMicros) * 1e-6f;
  if (dt <= 0) dt = DT_TARGET;
  lastMicros = now;

  // Accelerometer-derived pitch / roll (degrees)
  float axg = ax / 4096.0f;
  float ayg = ay / 4096.0f;
  float azg = az / 4096.0f;
  float accPitch = atan2(-axg, sqrt(ayg * ayg + azg * azg)) * 57.2957795f;  // Bug 2 fixed: was ayg, correct axis is -axg
  float accRoll  = atan2(ayg, azg) * 57.2957795f;                           // Bug 2 fixed: was -axg, correct axis is ayg

  // Gyro rates (deg/s)
  float gxR = (gx - gyroXoff) / 65.5f;
  float gyR = (gy - gyroYoff) / 65.5f;
  float gzR = (gz - gyroZoff) / 65.5f;

  // Complementary filter
  angleX = ALPHA * (angleX + gxR * dt) + (1.0f - ALPHA) * accPitch;
  angleY = ALPHA * (angleY + gyR * dt) + (1.0f - ALPHA) * accRoll;
  angleZ += gzR * dt;     // yaw — gyro integration only (no magnetometer)

  // FIX 2: stream real gyro rates (deg/s) alongside fused angles so the
  // Python visualiser can plot actual sensor data instead of a fake
  // numerical derivative.  Format: "aX,aY,aZ,gxR,gyR,gzR\n"
  Serial.print(angleX, 2); Serial.print(',');
  Serial.print(angleY, 2); Serial.print(',');
  Serial.print(angleZ, 2); Serial.print(',');
  Serial.print(gxR, 2);    Serial.print(',');
  Serial.print(gyR, 2);    Serial.print(',');
  Serial.println(gzR, 2);

  // Pace the loop to ~100 Hz
  while ((micros() - now) < (uint32_t)(DT_TARGET * 1e6f)) { /* spin */ }
}