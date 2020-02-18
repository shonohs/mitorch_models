# mitorch-models
A collection of vision-related PyTorch models.

## Readability first
This library is optimized for readability. We think it will help you understand the core ideas of each models. Reproducing the results from the papers is not the goal of this project.

## Implemented models

### Classification

* VGG-A, VGG-B, VGG-C, VGG-D(VGG16), VGG-E
* SqueezeNet
* MobileNetV2
* MobileNetV3
* MobileNetV3Small
* ShuffleNet
* ShuffleNetV2
* ResNext14, ResNext26, ResNext50, ResNext101
* SEResNext14, SEResNext26, SEResNext50, SEResNext101
* EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7

### Object Detection

* SSDLite
* RetinaNet (in preview)

## Benchmarks

Name | Model size (1 class)
:--- | :-------------------
SqueezeNet | 2.9MB
MobileNetV2 | 8.7MB
MobileNetV3 | 16.2MB
MobileNetV3Small | 5.9MB
ShuffleNet | 3.5MB
ShuffleNetV2 | 4.9MB
ResNext50 | 90MB
SEResNext50 | 100MB
EfficientNetB0 | 17.8MB
EfficientNetB1 | 26.6MB
EfficientNetB2 | 31.3MB
EfficientNetB3 | 43.0MB
EfficientNetB4 | 69.0MB
EfficientNetB5 | 107.7MB
EfficientNetB6 | 153.5MB
EfficientNetB7 | 237.3MB

Name | Model size (1 class)
:--- | :-------------------
MobileNetV2-SSDLite | 13MB
MobileNetV3-SSDLite | 20.9MB
MobileNetV3Small-SSDLite | 9.7MB
SEResNext50-SSDLite | 106MB
MobileNetV2-FPN-SSDLite | 32MB
MobileNetV3-FPN-SSDLite | 31MB
MobileNetV3Small-FPN-SSDLite | 20MB
EfficientNetB0-FPNLite-SSDLite | 22.7MB
EfficientNetB1-FPNLite-SSDLite | 31.4MB
EfficientNetB2-FPNLite-SSDLite | 36.5MB
EfficientNetB3-FPNLite-SSDLite | 48.5MB
MobileNetV2-FPNLite-SSDLite | 14MB
MobileNetV3-FPNLite-SSDLite | 20.9MB
MobileNetV3Small-FPNLite-SSDLite | 9.0MB
MobileNetV2-FPN-RetinaNet | 51MB
MobileNetV3-FPN-RetinaNet | 49MB
MobileNetV3Small-FPN-RetinaNet | 39MB
EfficientNetB0-FPNLite-RetinaNet | 40.8MB
EfficientNetB1-FPNLite-RetinaNet | 49.5MB
EfficientNetB2-FPNLite-RetinaNet | 54.6MB
EfficientNetB3-FPNLite-RetinaNet | 66.7MB
MobileNetV2-FPNLite-RetinaNet | 32MB
MobileNetV3-FPNLite-RetinaNet | 38.5MB
MobileNetV3Small-FPNLite-RetinaNet | 26.9MB