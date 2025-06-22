#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>


const int A1A = 3;
const int A1B = 2;
char state = 's';
int speedValue = 150;
bool keepRunning = true; 

Adafruit_ADXL345_Unified accel(12345);
const unsigned long SAMPLE_INTERVAL_MS = 10;

unsigned long previousMillis = 0;


void setup() {
  Serial.begin(115200);
  if (!accel.begin()) {
    Serial.println("ADXL345 연결 실패!");
    while(1);
  }
  accel.setRange(ADXL345_RANGE_16_G);
  accel.setDataRate(ADXL345_DATARATE_100_HZ);
  Serial.println("CSV_HEADER,Timestamp(ms),X,Y,Z");

  pinMode(A1A, OUTPUT);
  pinMode(A1B, OUTPUT);
  analogWrite(A1A, 0);
  digitalWrite(A1B, LOW);
}

void loop() {
  if (!keepRunning) { // 'q' 입력 시 데이터 수집 및 모터 동작 중단
    analogWrite(A1A, 0);
    digitalWrite(A1B, LOW);
    return;
  }

  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= SAMPLE_INTERVAL_MS) {
    previousMillis = currentMillis;
    sensors_event_t event;
    accel.getEvent(&event);
    Serial.print(millis());
    Serial.print(",");
    Serial.print(event.acceleration.x, 3);
    Serial.print(",");
    Serial.print(event.acceleration.y, 3);
    Serial.print(",");
    Serial.println(event.acceleration.z, 3);
  }

  handleMotorControl();
}

void handleMotorControl() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.equalsIgnoreCase("s")) {
      state = 's';
      analogWrite(A1A, 0);
      digitalWrite(A1B, LOW);
      Serial.println("Motor stopped");
    }
    else if (input.equalsIgnoreCase("c")) {
      state = 'c';
      analogWrite(A1A, speedValue);
      digitalWrite(A1B, LOW);
      Serial.println("Motor continuous mode");
    }
    else if (input.equalsIgnoreCase("q")) { // 'q' 입력 시 루프 중단
      keepRunning = false;
      analogWrite(A1A, 0);
      digitalWrite(A1B, LOW);
      Serial.println("Data collection and motor stopped by 'q'");
    }
    else {
      int val = input.toInt();
      if (val >= 0 && val <= 255) {
        speedValue = val;
        Serial.print("Speed set to: ");
        Serial.println(speedValue);
        if (state == 'c') {
          analogWrite(A1A, speedValue);
        }
      } else {
        Serial.println("Invalid speed value (0-255)");
      }
    }
  }
}
