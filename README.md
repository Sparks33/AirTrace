# AirTrace
This project is a simple painting application developed using OpenCV and MediaPipe Hands. It allows users to draw on a canvas using hand gestures captured by a webcam.

## Features

- **Hand Gesture Control:** Hand gestures detected by MediaPipe Hands are used to control drawing actions such as selecting colors and tools.
- **Drawing Tools:** Users can select drawing tools from the tool package, including circles, lines, and rectangles..
- **Fill Mode:** Users can toggle fill mode for shapes such as rectangles and circles.
- **Undo Functionality:** Allows users to undo the last drawing action.
- **Eraser Size:** Can adjust the Eraser size according to your need.
- **Preview mode:** Can switch between SIM or LIVE mode.
- **Clear Page:** User can clear the entire canvas using one button.
- **Real-time Preview:** Drawing actions are displayed in real-time on the canvas.

## Requirements

- Python 3
- OpenCV (cv2)
- MediaPipe Hands
- NumPy

Install the required dependencies using pip:

```bash
pip install opencv-python mediapipe numpy
```

## Controls
- **Click:** When the distance between two selected points is less than 55 pixels, it's considered a click. However, the click is registered only if the time elapsed since the last click is greater than 0.8 (variable cooldown_period) seconds.

## Preview
<img width="1470" height="956" alt="image" src="https://github.com/user-attachments/assets/629bcca0-24eb-4c0f-aec6-1677edd18bbc" />

