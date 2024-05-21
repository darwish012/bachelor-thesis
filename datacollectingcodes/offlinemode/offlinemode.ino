#include <Wire.h>
#include <SD_MMC.h>
#include <MPU6050.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
MPU6050 mpu;
#define SDA_PIN 16 // GPIO14 for SDA 
#define SCL_PIN 13 // GPIO13 for SCL
#define TOUCH_PIN 12 // GPIO pin for touch sensor
File dataFile;
int x = 0; // State variable
unsigned long lastTouchTime = -1000;

void setup() {
  Serial.begin(115200);
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  Wire.begin(SDA_PIN, SCL_PIN); // Initialize SDA and SCL pins
  mpu.initialize();// Initialize MPU
  if (!SD_MMC.begin("/sdcard", true)) {
    Serial.println("SD Card initialization failed!");
    return;}
  Serial.println("SD Card initialized successfully!");
  pinMode(TOUCH_PIN, INPUT);// Set up touch pin
  }

void loop() { int touchState = digitalRead(TOUCH_PIN);unsigned long currentTime = millis();
 switch (x) {  case 0: // Idle state
if (touchState == HIGH && (currentTime - lastTouchTime) >= 1000) {
  Serial.println("Touch detected, starting recording.");   dataFile = SD_MMC.open("/data.csv", FILE_WRITE);
  if (!dataFile) {Serial.println("Error opening data.csv for writing!");return;} 
  dataFile.println("Time,Accel X,Accel Y,Accel Z,Gyro X,Gyro Y,Gyro Z"); 
  x++; // Move to the next state 
   lastTouchTime = currentTime;}  break;
                case 1: // Recording state    
if (touchState == HIGH && (currentTime - lastTouchTime) >= 2000) {        
        Serial.println("Touched again, stopping recording.");
        dataFile.close(); // Close the file
        x++; // Move to the next state
         lastTouchTime = currentTime; 
      } else {
int16_t ax, ay, az;int16_t gx, gy, gz;
mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);unsigned long currentTime = millis();
dataFile.print(currentTime); dataFile.print(",");dataFile.print(ax);
 dataFile.print(",");dataFile.print(ay); dataFile.print(",");dataFile.print(az); 
dataFile.print(",");dataFile.print(gx); dataFile.print(",");
dataFile.print(gy); dataFile.print(",");dataFile.println(gz);}
break;
             case 2: // File closed state
      if (touchState == HIGH && (currentTime - lastTouchTime) >= 1000) {
 Serial.println("Touch detected again, resetting.");  x = 0; // Reset state variable
  lastTouchTime = currentTime;delay(500); ESP.restart();}   break;} 
   delay(10); }
