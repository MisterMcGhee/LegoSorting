/*
 * LegoSorter_MotorControl.ino
 *
 * Arduino sketch for LEGO Sorting Machine — DC motor control (L298N)
 *
 * This sketch handles motor letters B, C, D, and E.
 * It is intended to run on the SAME Arduino as LegoSorter_Servo.ino.
 * Both files live in the same sketch folder; the Arduino IDE compiles
 * them together.  setup() and loop() live in LegoSorter_Servo.ino.
 *
 * PROTOCOL:
 *   Receives "LETTER,VALUE\n" over Serial (57600 baud).
 *   VALUE is a signed PWM integer in the range [-255, 255].
 *     Positive  → forward  (IN1 HIGH / IN2 LOW  on L298N)
 *     Negative  → reverse  (IN1 LOW  / IN2 HIGH on L298N)
 *     Zero      → stop     (both LOW, EN = 0)
 *
 * MOTOR LAYOUT:
 *   Letter  Role              L298N Board  EN pin (PWM)
 *   ──────  ────────────────  ───────────  ────────────
 *   B       Conveyor belt     Board #2     see MOTOR_B_EN
 *   C       Rotary feeder C   Board #1     see MOTOR_C_EN
 *   D       Rotary feeder D   Board #1     see MOTOR_D_EN
 *   E       Future feeder     Board #2     see MOTOR_E_EN
 *
 * =============================================================================
 * !! IMPORTANT — VERIFY ALL PIN ASSIGNMENTS BEFORE FLASHING !!
 *
 * The EN (enable/PWM) pins come from the Python module documentation:
 *   Motor B EN = 10,  Motor C EN = 9,  Motor D EN = 6,  Motor E EN = 5
 *
 * POTENTIAL CONFLICT: The servo sketch uses SERVO_PIN = 9.  If Motor C's
 * EN wire is also connected to Arduino pin 9 you cannot run both on the
 * same Arduino without moving one of them.  Suggested fix: move the servo
 * to pin 3 (also a PWM pin on Uno/Mega) and update SERVO_PIN in the servo
 * sketch accordingly.
 *
 * The IN (direction) pin assignments below are PLACEHOLDERS.  Replace them
 * with the actual Arduino pin numbers that your IN1/IN2 wires are connected
 * to.  Any available digital output pins will work for direction control.
 * =============================================================================
 */

// ============================================================================
// PIN DEFINITIONS  — !! VERIFY AGAINST YOUR WIRING BEFORE FLASHING !!
// ============================================================================

// Motor B  —  Conveyor belt  (L298N Board #2, Channel A)
const int MOTOR_B_EN  = 10;   // ENA → Arduino PWM pin 10  (VERIFY)
const int MOTOR_B_IN1 = 22;   // IN1 → Arduino digital pin  (VERIFY / PLACEHOLDER)
const int MOTOR_B_IN2 = 23;   // IN2 → Arduino digital pin  (VERIFY / PLACEHOLDER)

// Motor C  —  Rotary feeder C  (L298N Board #1, Channel A)
// !! If your servo is on pin 9, move it to pin 3 and update SERVO_PIN !!
const int MOTOR_C_EN  =  9;   // ENA → Arduino PWM pin 9   (VERIFY — see conflict note above)
const int MOTOR_C_IN1 = 24;   // IN1 → Arduino digital pin  (VERIFY / PLACEHOLDER)
const int MOTOR_C_IN2 = 25;   // IN2 → Arduino digital pin  (VERIFY / PLACEHOLDER)

// Motor D  —  Rotary feeder D  (L298N Board #1, Channel B)
const int MOTOR_D_EN  =  6;   // ENB → Arduino PWM pin 6   (VERIFY)
const int MOTOR_D_IN1 = 26;   // IN3 → Arduino digital pin  (VERIFY / PLACEHOLDER)
const int MOTOR_D_IN2 = 27;   // IN4 → Arduino digital pin  (VERIFY / PLACEHOLDER)

// Motor E  —  Future feeder  (L298N Board #2, Channel B)
const int MOTOR_E_EN  =  5;   // ENB → Arduino PWM pin 5   (VERIFY)
const int MOTOR_E_IN1 = 28;   // IN3 → Arduino digital pin  (VERIFY / PLACEHOLDER)
const int MOTOR_E_IN2 = 29;   // IN4 → Arduino digital pin  (VERIFY / PLACEHOLDER)


// ============================================================================
// MOTOR SETUP  —  called from setup() in LegoSorter_Servo.ino
// ============================================================================

void setupMotors() {
  // Configure all motor pins as outputs and leave motors stopped
  int enPins[]  = { MOTOR_B_EN,  MOTOR_C_EN,  MOTOR_D_EN,  MOTOR_E_EN  };
  int in1Pins[] = { MOTOR_B_IN1, MOTOR_C_IN1, MOTOR_D_IN1, MOTOR_E_IN1 };
  int in2Pins[] = { MOTOR_B_IN2, MOTOR_C_IN2, MOTOR_D_IN2, MOTOR_E_IN2 };

  for (int i = 0; i < 4; i++) {
    pinMode(enPins[i],  OUTPUT);
    pinMode(in1Pins[i], OUTPUT);
    pinMode(in2Pins[i], OUTPUT);
    // Start stopped: EN = 0, both direction pins LOW
    analogWrite(enPins[i], 0);
    digitalWrite(in1Pins[i], LOW);
    digitalWrite(in2Pins[i], LOW);
  }
}


// ============================================================================
// CORE MOTOR DRIVER
// ============================================================================

/*
 * driveMotor — set one motor's speed and direction.
 *
 *   enPin   PWM-capable enable pin
 *   in1Pin  Direction pin 1 (HIGH = forward side)
 *   in2Pin  Direction pin 2 (HIGH = reverse side)
 *   value   Signed PWM [-255, 255].  Sign encodes direction; magnitude is duty.
 */
void driveMotor(int enPin, int in1Pin, int in2Pin, int value) {
  if (value == 0) {
    // Coast to stop
    digitalWrite(in1Pin, LOW);
    digitalWrite(in2Pin, LOW);
    analogWrite(enPin, 0);
  } else if (value > 0) {
    // Forward
    int pwm = constrain(value, 0, 255);
    digitalWrite(in1Pin, HIGH);
    digitalWrite(in2Pin, LOW);
    analogWrite(enPin, pwm);
  } else {
    // Reverse  (value is negative)
    int pwm = constrain(-value, 0, 255);
    digitalWrite(in1Pin, LOW);
    digitalWrite(in2Pin, HIGH);
    analogWrite(enPin, pwm);
  }
}


// ============================================================================
// COMMAND DISPATCH  —  called from processCommand() in LegoSorter_Servo.ino
// ============================================================================

/*
 * processMotorCommand — handle a motor command string.
 *
 *   cmd  Null-terminated string, e.g. "C,-153" or "B,200"
 *
 * Returns true if the letter was a known motor letter (B/C/D/E),
 * false if it should be handled by another dispatcher (e.g. servo).
 */
bool processMotorCommand(const char* cmd) {
  char letter = cmd[0];

  // cmd[2] is the start of the value (after "X,")
  // atoi handles negative numbers natively
  int value = atoi(&cmd[2]);

  switch (letter) {
    case 'B':
      driveMotor(MOTOR_B_EN, MOTOR_B_IN1, MOTOR_B_IN2, value);
      return true;

    case 'C':
      driveMotor(MOTOR_C_EN, MOTOR_C_IN1, MOTOR_C_IN2, value);
      return true;

    case 'D':
      driveMotor(MOTOR_D_EN, MOTOR_D_IN1, MOTOR_D_IN2, value);
      return true;

    case 'E':
      driveMotor(MOTOR_E_EN, MOTOR_E_IN1, MOTOR_E_IN2, value);
      return true;

    default:
      return false;  // Not a motor command — let the caller handle it
  }
}
