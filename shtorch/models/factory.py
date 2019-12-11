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
        'MobileNetV2SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(MobileNetV2()), num_classes),
        'ShuffleNet': lambda num_classes: Classifier(ShuffleNet(), num_classes),
        'ShuffleNetV2': lambda num_classes: Classifier(ShuffleNetV2(), num_classes),
        'SqueezeNet': lambda num_classes: Classifier(SqueezeNet(), num_classes),
        'VGG16': lambda num_classes: Classifier(VGG16(), num_classes),
    }


    @staticmethod
    def create(model_name, num_classes):
        if model_name not in ModelFactory.PREDEFINED_MODELS:
            raise NotImplementedError(f"Unknown model name: {model_name}")

        return ModelFactory.PREDEFINED_MODELS[model_name](num_classes)
