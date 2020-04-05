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

Name | Model size (1 class) | input = 224 | input = 500 | input = 500 (gpu) |
:--- | :------------------- | :---------- | :---------- | :---------------- |
SqueezeNet | 2.9MB | 45ms | 105ms | 5ms |
MobileNetV2 | 8.7MB | 34ms | 113ms | 9ms |
MobileNetV3 | 16.2MB | 100ms | 100ms | 13ms |
MobileNetV3Small | 5.9MB | 41ms | 79ms | 12ms |
ShuffleNet | 3.5MB | 59ms | 81ms | 11ms |
ShuffleNetV2 | 4.9MB | 56ms | 88ms | 12ms |
ResNext50 | 90MB | 104ms | 324ms | 24ms |
SEResNext50 | 100MB | 114ms | 351ms | 26ms |
EfficientNetB0 | 17.8 MB | 100ms | 121ms | 16ms |
EfficientNetB1 | 26.6 MB | 139ms | 171ms | 24ms |
EfficientNetB2 | 31.3 MB | 142ms | 178ms | 24 ms |
EfficientNetB3 | 43.0 MB | 163 ms | 211 ms | 27 ms |
EfficientNetB4 | 69.0 MB | 208 ms | 308 ms | 33 ms |
EfficientNetB5 | 107.7 MB | 251 ms | 406 ms | 41 ms |
EfficientNetB6 | 153.5 MB | 276 ms | 507 ms | 46 ms |
EfficientNetB7 | 237.3 MB | 334 ms | 685 ms | 59 ms |

Name | Model size (1 class) | input = 224 | input = 500 | input = 500 (gpu) |
:--- | :------------------- | :---------- | :---------- | :---------------- |
MobileNetV2-SSDLite | 12.9 MB | 57 ms | 147 ms | 39 ms |
MobileNetV3-SSDLite | 19.7 MB | 120 ms | 127 ms | 44 ms |
MobileNetV3Small-SSDLite | 8.5 MB | 66 ms | 112 ms | 42 ms |
SEResNext50-SSDLite | 103.7 MB | 130 ms | 371 ms | 56 ms |
MobileNetV2-FPN-SSDLite | 31.3 MB | 65 ms | 204 ms | 71 ms |
MobileNetV3-FPN-SSDLite | 35.8 MB | 130 ms | 189 ms | 77 ms |
MobileNetV3Small-FPN-SSDLite | 21.2 MB | 75ms | 180 ms | 74 ms |
EfficientNetB0-FPNLite-SSDLite | 22.7 MB | 127 ms | 212 ms | 80 ms |
EfficientNetB1-FPNLite-SSDLite | 31.4 MB | 164 ms | 263 ms | 89 ms |
EfficientNetB2-FPNLite-SSDLite | 36.5 MB | 179 ms | 267 ms | 87 ms |
EfficientNetB3-FPNLite-SSDLite | 48.5 MB | 199 ms | 308 ms | 91 ms |
MobileNetV2-FPNLite-SSDLite | 13.4 MB | 64 ms | 207 ms | 72 ms |
MobileNetV3-FPNLite-SSDLite | 20.4 MB | 131 ms | 189 ms | 77 ms |
MobileNetV3Small-FPNLite-SSDLite | 8.8 MB | 75 ms | 177 ms | 75 ms |
MobileNetV2-FPN-RetinaNet | 49.4 MB | 109 ms | 357 ms | 71 ms |
MobileNetV3-FPN-RetinaNet | 53.9 MB | 173 ms | 336 ms | 76 ms |
MobileNetV3Small-FPN-RetinaNet | 39.3 MB | 117 ms | 328 ms | 74 ms |
EfficientNetB0-FPNLite-RetinaNet | 40.8 MB | 169 ms | 356 ms | 81 ms |
EfficientNetB1-FPNLite-RetinaNet | 49.5 MB | 214 ms | 407 ms | 88 ms |
EfficientNetB2-FPNLite-RetinaNet | 54.6 MB | 219 ms | 418 ms | 89 ms |
EfficientNetB3-FPNLite-RetinaNet | 66.7 MB | 240 ms | 449 ms | 90 ms |
MobileNetV2-FPNLite-RetinaNet | 31.5 MB | 105 ms | 363 ms | 74 ms |
MobileNetV3-FPNLite-RetinaNet | 38.5 MB | 171 ms | 335 ms | 78 ms |
MobileNetV3Small-FPNLite-RetinaNet | 26.9 MB | 113 ms | 313 ms | 76 ms |
EfficientNetB0-FPNLite-RetinaNetLite | 24.5 MB | 177 ms | 320 ms | 88 ms |
EfficientNetB1-FPNLite-RetinaNetLite | 33.3 MB | 215 ms | 359 ms | 95 ms |
EfficientNetB2-FPNLite-RetinaNetLite | 38.4 MB | 220 ms | 362 ms | 99 ms |
EfficientNetB3-FPNLite-RetinaNetLite | 50.4 MB | 242 ms | 393 ms | 103 ms |
MobileNetV2-FPNLite-RetinaNetLite | 15.3 MB | 109 ms | 300 ms | 79 ms |
MobileNetV3-FPNLite-RetinaNetLite | 22.3 MB | 180 ms | 281 ms | 85 ms |
MobileNetV3Small-FPNLite-RetinaNetLite | 10.7 MB | 125 ms | 261 ms | 82 ms |
MobileNetV2-MnasFPN-RetinaNetLite | 11.8 MB | 129 ms | 299 ms | 88 ms |
MobileNetV3-MnasFPN-RetinaNetLite | 19.2 MB | 195 ms | 281 ms | 94 ms |
MobileNetV3Small-MnasFPN-RetinaNetLite | 8.6 MB | 137 ms | 262 ms | 91 ms |
EfficientNetB0-MnasFPN-RetinaNetLite | 20.9 MB | 197 ms | 304 ms | 98 ms |
EfficientNetB1-MnasFPN-RetinaNetLite | 29.7 MB | 237 ms | 357 ms | 105 ms |
EfficientNetB2-MnasFPN-RetinaNetLite | 34.4 MB | 244 ms | 366 ms | 103 ms |
EfficientNetB3-MnasFPN-RetinaNetLite | 46.3 MB | 258 ms | 404 ms | 111 ms |