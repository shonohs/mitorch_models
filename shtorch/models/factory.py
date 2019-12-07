from .mobilenetv2 import MobileNetV2

class ModelFactory:
    @staticmethod
    def create(model_name, num_classes):
        if model_name == 'MobileNetV2':
            return Classifier(MobileNetV2, num_classes)
        else:
            raise NotImplementedError()
