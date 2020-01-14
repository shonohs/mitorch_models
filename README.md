# shtorch-models
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
MobileNetV2 | 8.9MB
MobileNetV3 | 11MB
MobileNetV3Small | 4.3MB
ShuffleNet | 3.5MB
ShuffleNetV2 | 5.0MB
ResNext50 | 90MB
SEResNext50 | 100MB
EfficientNetB0 | 28MB
MobileNetV2-SSDLite | 13MB
MobileNetV3-SSDLite | 15MB
MobileNetV3Small-SSDLite | 8.0MB
SEResNext50-SSDLite | 106MB
MobileNetV2-FPN-RetinaNet | 51MB
MobileNetV2-FPN-SSDLite | 32MB
MobileNetV3-FPN-SSDLite | 31MB
MobileNetV3Small-FPN-SSDLite | 20MB