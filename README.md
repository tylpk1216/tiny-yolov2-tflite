## tiny-yolov2-tflite
Run Tiny-YOLOv2 model on TensorFlow Lite.

## Dataset
COCO, 80 classes

## Test Steps
1. Download 'tiny_yolo.h5' from https://drive.google.com/file/d/14-5ZojD1HSgMKnv6_E3WUcBPxaVm52X2/view?usp=sharing
2. Convert TF model to TF Lite mode.
3. Run Tiny-YOLOv2-tflite.py

## Test Environment
Windows 10 + TensorFlow 1.15.0

## Todo
Do NMS by every class.
(I am not sure this is necessary step?)

## Acknowledgement
- https://medium.com/@amrokamal_47691/yolo-yolov2-and-yolov3-all-you-want-to-know-7e3e92dc4899 (Good document)
- https://pjreddie.com/darknet/yolov2/ (Developer)
- https://blog.francium.tech/real-time-object-detection-on-mobile-with-flutter-tensorflow-lite-and-yolo-android-part-a0042c9b62c6 (Reference Code)
- https://github.com/kaka-lin/object-detection (Reference Code)
- https://github.com/thtrieu/darkflow (TensorFlow Version)
