from . import *

class ModelFactory:
    PREDEFINED_MODELS = {
        'EfficientNetB0': lambda num_classes: Classifier(EfficientNetB0(), num_classes),
        'EfficientNetB1': lambda num_classes: Classifier(EfficientNetB1(), num_classes),
        'EfficientNetB2': lambda num_classes: Classifier(EfficientNetB2(), num_classes),
        'EfficientNetB3': lambda num_classes: Classifier(EfficientNetB3(), num_classes),
        'EfficientNetB4': lambda num_classes: Classifier(EfficientNetB4(), num_classes),
        'EfficientNetB5': lambda num_classes: Classifier(EfficientNetB5(), num_classes),
        'EfficientNetB6': lambda num_classes: Classifier(EfficientNetB6(), num_classes),
        'EfficientNetB7': lambda num_classes: Classifier(EfficientNetB7(), num_classes),
        'MobileNetV2': lambda num_classes: Classifier(MobileNetV2(), num_classes),
        'MobileNetV3': lambda num_classes: Classifier(MobileNetV3(), num_classes),
        'MobileNetV3Small': lambda num_classes: Classifier(MobileNetV3Small(), num_classes),
        'SEResNext50': lambda num_classes: Classifier(SEResNext50(), num_classes),
        'ShuffleNet': lambda num_classes: Classifier(ShuffleNet(), num_classes),
        'ShuffleNetV2': lambda num_classes: Classifier(ShuffleNetV2(), num_classes),
        'SqueezeNet': lambda num_classes: Classifier(SqueezeNet(), num_classes),
        'VGG16': lambda num_classes: Classifier(VGG16(), num_classes),
        'EfficientNetB0-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(EfficientNetB0()), num_classes),
        'EfficientNetB1-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(EfficientNetB1()), num_classes),
        'EfficientNetB2-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(EfficientNetB2()), num_classes),
        'EfficientNetB3-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(EfficientNetB3()), num_classes),
        'EfficientNetB4-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(EfficientNetB4()), num_classes),
        'EfficientNetB5-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(EfficientNetB5()), num_classes),
        'EfficientNetB6-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(EfficientNetB6()), num_classes),
        'EfficientNetB7-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(EfficientNetB7()), num_classes),
        'MobileNetV2-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(MobileNetV2()), num_classes),
        'MobileNetV3-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(MobileNetV3()), num_classes),
        'MobileNetV3Small-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(MobileNetV3Small()), num_classes),
        'SEResNext50-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(SEResNext50()), num_classes),
        'SEResNext101-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(SEResNext101()), num_classes),
        'ShuffleNetV2-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(ShuffleNetV2()), num_classes),
        'VGG16-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(VGG16ForSSD()), num_classes),

        'MobileNetV2-FPN-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetwork(MobileNetV2()), num_classes),
        'MobileNetV3-FPN-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetwork(MobileNetV3()), num_classes),
        'MobileNetV3Small-FPN-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetwork(MobileNetV3Small()), num_classes),

        'MobileNetV2-FPN-RetinaNet': lambda num_classes: RetinaNet(FeaturePyramidNetwork(MobileNetV2()), num_classes),
        'MobileNetV3-FPN-RetinaNet': lambda num_classes: RetinaNet(FeaturePyramidNetwork(MobileNetV3()), num_classes),
        'MobileNetV3Small-FPN-RetinaNet': lambda num_classes: RetinaNet(FeaturePyramidNetwork(MobileNetV3Small()), num_classes)
    }


    @staticmethod
    def create(model_name, num_classes):
        if model_name not in ModelFactory.PREDEFINED_MODELS:
            raise NotImplementedError(f"Unknown model name: {model_name}")

        model = ModelFactory.PREDEFINED_MODELS[model_name](num_classes)
        model.reset_parameters()
        return model
