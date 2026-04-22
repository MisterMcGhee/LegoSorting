/*
 * LegoSorter_Servo.ino
 *
 * Main sketch file for the LEGO Sorting Machine Arduino.
 * Handles servo control (letter A) and delegates motor commands
 * (letters B/C/D/E) to LegoSorter_MotorControl.ino, which lives in
 * the same sketch folder and is compiled together by the Arduino IDE.
 *
 * PROTOCOL:
 *   Receives "LETTER,VALUE\n" over Serial at 57600 baud.
 *   A  → Servo angle (0–180°)
 *   B  → Conveyor motor  (signed PWM -255..255)
 *   C  → Feeder C motor  (signed PWM -255..255)
 *   D  → Feeder D motor  (signed PWM -255..255)
 *   E  → Future feeder   (signed PWM -255..255, reserved)
 *
 * HARDWARE:
 *   Servo signal → SERVO_PIN (default pin 3, a PWM pin).
 *
 *   NOTE: The original servo sketch used pin 9.  Pin 9 is also used as
 *   the Motor C enable (EN) pin.  If your servo is still wired to pin 9
 *   you MUST resolve this conflict — either move the servo wire to pin 3
 *   (or another free PWM pin) or reroute Motor C EN to a different PWM
 *   pin and update MOTOR_C_EN in LegoSorter_MotorControl.ino.
 *
 * DESIGN PHILOSOPHY:
 *   - Python side does all validation; sketch trusts the input.
 *   - Minimal logic, fast execution.
 *   - Signed PWM convention: positive = forward, negative = reverse.
 */

#include <Servo.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

// Servo pin — must be a PWM-capable pin; default moved to 3 to free pin 9
// for Motor C EN.  Change back to 9 only if you have resolved the conflict.
const int SERVO_PIN     = 3;    // !! VERIFY against your wiring !!

const int HOME_POSITION = 90;   // Home/default servo angle
const long BAUD_RATE    = 57600;

// Command buffer — "D,-255\n" is 8 chars; 20 is safe headroom
const int BUFFER_SIZE = 20;
char commandBuffer[BUFFER_SIZE];
int bufferIndex = 0;

// ============================================================================
// GLOBAL OBJECTS
// ============================================================================

Servo sorterServo;
int currentAngle = HOME_POSITION;

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  Serial.begin(BAUD_RATE);

  // Initialise DC motors (defined in LegoSorter_MotorControl.ino)
  setupMotors();

  // Initialise servo
  sorterServo.attach(SERVO_PIN);
  sorterServo.write(HOME_POSITION);
  currentAngle = HOME_POSITION;
  delay(500);

  // Flush any garbage already in the serial buffer
  while (Serial.available() > 0) {
    Serial.read();
  }
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  if (Serial.available() > 0) {
    char inChar = Serial.read();

    if (inChar == '\n' || inChar == '\r') {
      if (bufferIndex > 0) {
        commandBuffer[bufferIndex] = '\0';
        processCommand(commandBuffer);
        bufferIndex = 0;
      }
    } else if (bufferIndex < BUFFER_SIZE - 1) {
      commandBuffer[bufferIndex++] = inChar;
    } else {
      // Buffer overflow — discard and reset
      bufferIndex = 0;
    }
  }
}

// ============================================================================
// COMMAND ROUTING
// ============================================================================

void processCommand(const char* cmd) {
  // Route motor commands (B/C/D/E) to the motor handler first.
  // processMotorCommand() returns true if it consumed the command.
  if (processMotorCommand(cmd)) {
    return;
  }

  // Letter A — servo angle
  if (cmd[0] == 'A') {
    int angle = atoi(&cmd[2]);
    sorterServo.write(angle);
    currentAngle = angle;
    Serial.flush();
  }
  // Unknown letters are silently ignored.
}
