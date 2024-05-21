#include <WiFi.h>
#include <Wire.h>
#include <MPU6050.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
MPU6050 mpu;
const char* ssid = "ab";
const char* password = "09876542";
const char* serverIPAddress = "192.168.130.83"; 
const uint16_t serverPort = 12345; 
#define SDA_PIN 16 // GPIO14 for SDA
#define SCL_PIN 13 // GPIO13 for SCL
#define TOUCH_PIN 12 // GPIO pin for touch sensor
WiFiClient client;
int x = 0; // State variable
unsigned long lastTouchTime = 0;

void setup() {
  Serial.begin(115200);
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  Wire.begin(SDA_PIN, SCL_PIN); // Initialize SDA and SCL pins
  mpu.initialize();
  WiFi.begin(ssid, password); // Connect to Wi-Fi
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);Serial.println("Connecting to WiFi...");}
  Serial.println("Connected to WiFi");
  pinMode(TOUCH_PIN, INPUT); // Set up touch pin
  }

void loop() {
  int touchState = digitalRead(TOUCH_PIN);unsigned long currentTime = millis();
    switch (x) {
      case 0: // Idle state
      if (touchState == HIGH && (currentTime - lastTouchTime) >= 1000) {
        Serial.println("Touch detected, setting up TCP.");
        if (client.connect(serverIPAddress, serverPort)) {
  Serial.println("TCP setup successful.");x++; // Move to the next state
  lastTouchTime = currentTime; } 
  else {Serial.println("TCP setup failed!");} }
      break;  
     case 1: // Recording state
      if (touchState == HIGH && (currentTime - lastTouchTime) >= 2000) {
 Serial.println("Touch released, stopping recording.");x++; // Move to the next state
        lastTouchTime = currentTime;} 
        else {  
  int16_t ax, ay, az; int16_t gx, gy, gz;
 mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
 String data = String(currentTime) + "," + String(ax) + "," + String(ay) + "," 
 + String(az) + ","+ String(gx) + "," + String(gy) + "," + String(gz);
        if (client.connected()) {
          client.println(data);   Serial.println("Data sent to server: " + data); } 
      else {Serial.println("Connection to server lost!");} } break;
      
    case 2: // File closed state
      if (touchState == HIGH && (currentTime - lastTouchTime) >= 1000) {
  Serial.println("Touch detected again, resetting."); x = 0; // Reset state variable
  lastTouchTime = currentTime;client.stop(); // Close TCP connection
        delay(500);ESP.restart();}  break; }
  delay(50); }
