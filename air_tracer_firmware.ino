#include <Servo.h>
#include <AccelStepper.h>
#include <MultiStepper.h>

 /*
  AirTracer Plotter Firmware - Programmed by Sparks33
  --------------------------
  Receives G-code over serial from the Python app and drives a 2-axis
  (X/Y) pen plotter built on a CNC Shield V3 + A4988 drivers, with a servo
  lifting the pen up/down (mounted where the Z-axis would normally be,
  wired to a spare digital pin - see SERVO_PIN note below).

  PROTOCOL (one line per command, newline-terminated, a small G-code
  subset - this is exactly what the Python app's "Generate G-code" step
  produces):
    G21              -> units = millimeters (accepted, no-op)
    G90              -> absolute positioning (accepted, no-op)
    G0 X<x> Y<y>     -> rapid move to x,y in MILLIMETERS (used for pen-up travel)
    G1 X<x> Y<y>     -> controlled move to x,y in MILLIMETERS (used while drawing)
    G1 F<rate>       -> sets feed rate for subsequent G1 moves (rate itself
                        is not used for real feed-rate control here - both
                        G0 and G1 use their own fixed speed below - but the
                        line is accepted so it doesn't error out)
    M3               -> lower the pen (pen down / "spindle on")
    M5               -> raise the pen (pen up / "spindle off")

  After each line is fully executed, this sketch sends back "OK\n" so the
  Python side knows it's safe to send the next line. This handshake is
  required - without it, commands could overflow the Arduino's small
  serial buffer while it's still physically moving.

  There is deliberately NO homing routine. Before each drawing, manually
  position the pen carriage at the starting point on the paper first -
  the very first move command just starts from wherever the pen already
  is, which is exactly how this project is meant to be used.

  REQUIRED LIBRARY:
    AccelStepper - install via Arduino IDE: Tools > Manage Libraries >
    search "AccelStepper" by Mike McCauley > Install

  ============================================================
  WIRING / PIN NOTES
  ============================================================
  Standard CNC Shield V3 pinout:
    X.STEP = D2   X.DIR = D5
    Y.STEP = D3   Y.DIR = D6
    Enable = D8   (LOW = drivers enabled)

  SERVO_PIN is set to D11 below.

  ============================================================
  CALIBRATION - YOU MUST DO THIS BEFORE IT WILL DRAW ACCURATELY
  ============================================================

  To calibrate:
    1. Upload this sketch.
    2. Open the Arduino IDE's Serial Monitor, set baud to 115200, line
       ending to "Newline".
    3. Type:  G0 X50 Y0   and press Enter (moves the X carriage toward 50mm)
    4. Physically measure how far the pen carriage actually moved with a
       ruler - call this measured_mm.
    5. New value = OLD_STEPS_PER_MM_X * (50.0 / measured_mm)
    6. Update STEPS_PER_MM_X below, re-upload, and re-test until 50mm
       commanded = 50mm actual movement.
    7. Repeat the same process for Y using "G0 X0 Y50".
*/

// ---- pins ----
#define X_STEP_PIN 2
#define X_DIR_PIN 5
#define Y_STEP_PIN 3
#define Y_DIR_PIN 6
#define ENABLE_PIN 8
#define SERVO_PIN 11 

// ---- calibration (see instructions above) ----
float STEPS_PER_MM_X = 80.0;
float STEPS_PER_MM_Y = 80.0;

// ---- servo angles for pen up/down ----
#define PEN_UP_ANGLE 90
#define PEN_DOWN_ANGLE 40
#define PEN_MOVE_DELAY_MS 150  

// ---- motion tuning: separate speeds for rapid travel (G0, pen up) vs
// controlled drawing moves (G1, pen down) - drawing moves are slower for
// accuracy and cleaner lines ----
#define TRAVEL_SPEED_STEPS_PER_SEC 2000
#define DRAW_SPEED_STEPS_PER_SEC   900
#define ACCEL_STEPS_PER_SEC2       1200

AccelStepper stepperX(AccelStepper::DRIVER, X_STEP_PIN, X_DIR_PIN);
AccelStepper stepperY(AccelStepper::DRIVER, Y_STEP_PIN, Y_DIR_PIN);
Servo pen;

String inputLine = "";
float currentXmm = 0.0;
float currentYmm = 0.0;

void setup() {
  Serial.begin(115200);

  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW); 

  stepperX.setMaxSpeed(TRAVEL_SPEED_STEPS_PER_SEC);
  stepperX.setAcceleration(ACCEL_STEPS_PER_SEC2);
  stepperY.setMaxSpeed(TRAVEL_SPEED_STEPS_PER_SEC);
  stepperY.setAcceleration(ACCEL_STEPS_PER_SEC2);

  pen.attach(SERVO_PIN);
  pen.write(PEN_UP_ANGLE);
  delay(400);

  Serial.println("READY");
}

void penUp() {
  pen.write(PEN_UP_ANGLE);
  delay(PEN_MOVE_DELAY_MS);
}

void penDown() {
  pen.write(PEN_DOWN_ANGLE);
  delay(PEN_MOVE_DELAY_MS);
}

void moveToMM(float xmm, float ymm) {
  long targetX = (long)(xmm * STEPS_PER_MM_X);
  long targetY = (long)(ymm * STEPS_PER_MM_Y);
  stepperX.moveTo(targetX);
  stepperY.moveTo(targetY);
  while (stepperX.distanceToGo() != 0 || stepperY.distanceToGo() != 0) {
    stepperX.run();
    stepperY.run();
  }
  currentXmm = xmm;
  currentYmm = ymm;
}

// Pulls the value following 'axis' (e.g. 'X') out of a G-code line, up to
// the next space or end of string. Returns true if that axis was present.
bool extractAxisValue(const String &line, char axis, float &outVal) {
  int idx = line.indexOf(axis);
  if (idx == -1) return false;
  int end = line.indexOf(' ', idx + 1);
  String tok = (end == -1) ? line.substring(idx + 1) : line.substring(idx + 1, end);
  outVal = tok.toFloat();
  return true;
}

void handleGMove(const String &line, bool isTravel) {
  float x = currentXmm;
  float y = currentYmm;
  bool hasX = extractAxisValue(line, 'X', x);
  bool hasY = extractAxisValue(line, 'Y', y);

  if (!hasX && !hasY) {
    // A line like "G1 F800" that only sets feed rate, no coordinates -
    // nothing to move, just acknowledge it.
    Serial.println("OK");
    return;
  }

  stepperX.setMaxSpeed(isTravel ? TRAVEL_SPEED_STEPS_PER_SEC : DRAW_SPEED_STEPS_PER_SEC);
  stepperY.setMaxSpeed(isTravel ? TRAVEL_SPEED_STEPS_PER_SEC : DRAW_SPEED_STEPS_PER_SEC);
  moveToMM(x, y);
  Serial.println("OK");
}

void handleLine(String line) {
  line.trim();
  if (line.length() == 0) return;
  line.toUpperCase();

  if (line.startsWith("G21") || line.startsWith("G90")) {
    Serial.println("OK");
    return;
  }
  if (line.startsWith("M3")) {
    penDown();
    Serial.println("OK");
    return;
  }
  if (line.startsWith("M5")) {
    penUp();
    Serial.println("OK");
    return;
  }
  if (line.startsWith("G0")) {
    handleGMove(line, true);
    return;
  }
  if (line.startsWith("G1")) {
    handleGMove(line, false);
    return;
  }

  // ---- legacy simple protocol (still supported for manual testing via
  // the Serial Monitor): U / D / M <x> <y> ----
  char cmd = line.charAt(0);
  if (cmd == 'U') {
    penUp();
    Serial.println("OK");
    return;
  }
  if (cmd == 'D') {
    penDown();
    Serial.println("OK");
    return;
  }
  if (cmd == 'M') {
    int sp1 = line.indexOf(' ');
    int sp2 = line.indexOf(' ', sp1 + 1);
    if (sp1 == -1 || sp2 == -1) {
      Serial.println("ERR bad M command");
      return;
    }
    float x = line.substring(sp1 + 1, sp2).toFloat();
    float y = line.substring(sp2 + 1).toFloat();
    stepperX.setMaxSpeed(DRAW_SPEED_STEPS_PER_SEC);
    stepperY.setMaxSpeed(DRAW_SPEED_STEPS_PER_SEC);
    moveToMM(x, y);
    Serial.println("OK");
    return;
  }

  Serial.println("ERR unknown command");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handleLine(inputLine);
      inputLine = "";
    } else if (c != '\r') {
      inputLine += c;
    }
  }
}
