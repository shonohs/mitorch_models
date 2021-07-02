from . import *
from .heads import *


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
        'ResNext14': lambda num_classes: Classifier(ResNext14(), num_classes),
        'ResNext26': lambda num_classes: Classifier(ResNext26(), num_classes),
        'ResNext50': lambda num_classes: Classifier(ResNext50(), num_classes),
        'ResNext101': lambda num_classes: Classifier(ResNext101(), num_classes),
        'SEResNext14': lambda num_classes: Classifier(SEResNext14(), num_classes),
        'SEResNext26': lambda num_classes: Classifier(SEResNext26(), num_classes),
        'SEResNext50': lambda num_classes: Classifier(SEResNext50(), num_classes),
        'SEResNext101': lambda num_classes: Classifier(SEResNext101(), num_classes),
        'ShuffleNet': lambda num_classes: Classifier(ShuffleNet(), num_classes),
        'ShuffleNetV2': lambda num_classes: Classifier(ShuffleNetV2(), num_classes),
        'SqueezeNet': lambda num_classes: Classifier(SqueezeNet(), num_classes),
        'VGG16': lambda num_classes: Classifier(VGG16(), num_classes),

        # SSDLite
        'EfficientNetB0-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(EfficientNetB0()), num_classes),
        'EfficientNetB1-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(EfficientNetB1()), num_classes),
        'EfficientNetB2-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(EfficientNetB2()), num_classes),
        'EfficientNetB3-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(EfficientNetB3()), num_classes),
        'EfficientNetB4-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(EfficientNetB4()), num_classes),
        'EfficientNetB5-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(EfficientNetB5()), num_classes),
        'EfficientNetB6-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(EfficientNetB6()), num_classes),
        'EfficientNetB7-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(EfficientNetB7()), num_classes),
        'MobileNetV2-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(MobileNetV2()), num_classes),
        'MobileNetV2-Sigmoid-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(MobileNetV2()), num_classes, use_sigmoid=True),
        'MobileNetV3-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(MobileNetV3()), num_classes),
        'MobileNetV3Small-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(MobileNetV3Small()), num_classes),
        'SEResNext50-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(SEResNext50()), num_classes),
        'SEResNext101-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(SEResNext101()), num_classes),
        'ShuffleNetV2-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(ShuffleNetV2()), num_classes),
        'VGG16-SSDLite': lambda num_classes: SSDLite(SSDLiteExtraLayers(VGG16ForSSD()), num_classes),

        'EfficientNetB0-FPN-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetwork(EfficientNetB0()), num_classes),
        'EfficientNetB1-FPN-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetwork(EfficientNetB1()), num_classes),
        'EfficientNetB2-FPN-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetwork(EfficientNetB2()), num_classes),
        'EfficientNetB3-FPN-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetwork(EfficientNetB3()), num_classes),
        'EfficientNetB4-FPN-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetwork(EfficientNetB4()), num_classes),
        'EfficientNetB5-FPN-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetwork(EfficientNetB5()), num_classes),
        'EfficientNetB6-FPN-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetwork(EfficientNetB6()), num_classes),
        'EfficientNetB7-FPN-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetwork(EfficientNetB7()), num_classes),
        'MobileNetV2-FPN-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetwork(MobileNetV2()), num_classes),
        'MobileNetV3-FPN-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetwork(MobileNetV3()), num_classes),
        'MobileNetV3Small-FPN-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetwork(MobileNetV3Small()), num_classes),

        'EfficientNetB0-FPNLite-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetworkLite(EfficientNetB0()), num_classes),
        'EfficientNetB1-FPNLite-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetworkLite(EfficientNetB1()), num_classes),
        'EfficientNetB2-FPNLite-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetworkLite(EfficientNetB2()), num_classes),
        'EfficientNetB3-FPNLite-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetworkLite(EfficientNetB3()), num_classes),
        'EfficientNetB4-FPNLite-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetworkLite(EfficientNetB4()), num_classes),
        'EfficientNetB5-FPNLite-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetworkLite(EfficientNetB5()), num_classes),
        'EfficientNetB6-FPNLite-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetworkLite(EfficientNetB6()), num_classes),
        'EfficientNetB7-FPNLite-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetworkLite(EfficientNetB7()), num_classes),
        'MobileNetV2-FPNLite-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetworkLite(MobileNetV2()), num_classes),
        'MobileNetV3-FPNLite-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetworkLite(MobileNetV3()), num_classes),
        'MobileNetV3Small-FPNLite-SSDLite': lambda num_classes: SSDLite(FeaturePyramidNetworkLite(MobileNetV3Small()), num_classes),

        'EfficientNetB0-MnasFPN-SSDLite': lambda num_classes: SSDLite(MnasFPN(EfficientNetB0()), num_classes),
        'EfficientNetB1-MnasFPN-SSDLite': lambda num_classes: SSDLite(MnasFPN(EfficientNetB1()), num_classes),
        'EfficientNetB2-MnasFPN-SSDLite': lambda num_classes: SSDLite(MnasFPN(EfficientNetB2()), num_classes),
        'EfficientNetB3-MnasFPN-SSDLite': lambda num_classes: SSDLite(MnasFPN(EfficientNetB3()), num_classes),
        'EfficientNetB4-MnasFPN-SSDLite': lambda num_classes: SSDLite(MnasFPN(EfficientNetB4()), num_classes),
        'EfficientNetB5-MnasFPN-SSDLite': lambda num_classes: SSDLite(MnasFPN(EfficientNetB5()), num_classes),
        'EfficientNetB6-MnasFPN-SSDLite': lambda num_classes: SSDLite(MnasFPN(EfficientNetB6()), num_classes),
        'EfficientNetB7-MnasFPN-SSDLite': lambda num_classes: SSDLite(MnasFPN(EfficientNetB7()), num_classes),
        'MobileNetV2-MnasFPN-SSDLite': lambda num_classes: SSDLite(MnasFPN(MobileNetV2()), num_classes),
        'MobileNetV3-MnasFPN-SSDLite': lambda num_classes: SSDLite(MnasFPN(MobileNetV3()), num_classes),
        'MobileNetV3Small-MnasFPN-SSDLite': lambda num_classes: SSDLite(MnasFPN(MobileNetV3Small()), num_classes),

        # RetinaNet
        'MobileNetV2-FPN-RetinaNet': lambda num_classes: RetinaNet(FeaturePyramidNetwork(MobileNetV2()), num_classes),
        'MobileNetV3-FPN-RetinaNet': lambda num_classes: RetinaNet(FeaturePyramidNetwork(MobileNetV3()), num_classes),
        'MobileNetV3Small-FPN-RetinaNet': lambda num_classes: RetinaNet(FeaturePyramidNetwork(MobileNetV3Small()), num_classes),

        'EfficientNetB0-FPNLite-RetinaNet': lambda num_classes: RetinaNet(FeaturePyramidNetworkLite(EfficientNetB0()), num_classes),
        'EfficientNetB1-FPNLite-RetinaNet': lambda num_classes: RetinaNet(FeaturePyramidNetworkLite(EfficientNetB1()), num_classes),
        'EfficientNetB2-FPNLite-RetinaNet': lambda num_classes: RetinaNet(FeaturePyramidNetworkLite(EfficientNetB2()), num_classes),
        'EfficientNetB3-FPNLite-RetinaNet': lambda num_classes: RetinaNet(FeaturePyramidNetworkLite(EfficientNetB3()), num_classes),
        'EfficientNetB4-FPNLite-RetinaNet': lambda num_classes: RetinaNet(FeaturePyramidNetworkLite(EfficientNetB4()), num_classes),
        'EfficientNetB5-FPNLite-RetinaNet': lambda num_classes: RetinaNet(FeaturePyramidNetworkLite(EfficientNetB5()), num_classes),
        'EfficientNetB6-FPNLite-RetinaNet': lambda num_classes: RetinaNet(FeaturePyramidNetworkLite(EfficientNetB6()), num_classes),
        'EfficientNetB7-FPNLite-RetinaNet': lambda num_classes: RetinaNet(FeaturePyramidNetworkLite(EfficientNetB7()), num_classes),
        'MobileNetV2-FPNLite-RetinaNet': lambda num_classes: RetinaNet(FeaturePyramidNetworkLite(MobileNetV2()), num_classes),
        'MobileNetV3-FPNLite-RetinaNet': lambda num_classes: RetinaNet(FeaturePyramidNetworkLite(MobileNetV3()), num_classes),
        'MobileNetV3Small-FPNLite-RetinaNet': lambda num_classes: RetinaNet(FeaturePyramidNetworkLite(MobileNetV3Small()), num_classes),

        'EfficientNetB0-MnasFPN-RetinaNet': lambda num_classes: RetinaNet(MnasFPN(EfficientNetB0()), num_classes),
        'EfficientNetB1-MnasFPN-RetinaNet': lambda num_classes: RetinaNet(MnasFPN(EfficientNetB1()), num_classes),
        'EfficientNetB2-MnasFPN-RetinaNet': lambda num_classes: RetinaNet(MnasFPN(EfficientNetB2()), num_classes),
        'EfficientNetB3-MnasFPN-RetinaNet': lambda num_classes: RetinaNet(MnasFPN(EfficientNetB3()), num_classes),
        'EfficientNetB4-MnasFPN-RetinaNet': lambda num_classes: RetinaNet(MnasFPN(EfficientNetB4()), num_classes),
        'EfficientNetB5-MnasFPN-RetinaNet': lambda num_classes: RetinaNet(MnasFPN(EfficientNetB5()), num_classes),
        'EfficientNetB6-MnasFPN-RetinaNet': lambda num_classes: RetinaNet(MnasFPN(EfficientNetB6()), num_classes),
        'EfficientNetB7-MnasFPN-RetinaNet': lambda num_classes: RetinaNet(MnasFPN(EfficientNetB7()), num_classes),
        'MobileNetV2-MnasFPN-RetinaNet': lambda num_classes: RetinaNet(MnasFPN(MobileNetV2()), num_classes),
        'MobileNetV3-MnasFPN-RetinaNet': lambda num_classes: RetinaNet(MnasFPN(MobileNetV3()), num_classes),
        'MobileNetV3Small-MnasFPN-RetinaNet': lambda num_classes: RetinaNet(MnasFPN(MobileNetV3Small()), num_classes),

        # RetinaNetLite
        'MobileNetV2-FPN-RetinaNetLite': lambda num_classes: RetinaNetLite(FeaturePyramidNetwork(MobileNetV2()), num_classes),
        'MobileNetV3-FPN-RetinaNetLite': lambda num_classes: RetinaNetLite(FeaturePyramidNetwork(MobileNetV3()), num_classes),
        'MobileNetV3Small-FPN-RetinaNetLite': lambda num_classes: RetinaNetLite(FeaturePyramidNetwork(MobileNetV3Small()), num_classes),

        'EfficientNetB0-FPNLite-RetinaNetLite': lambda num_classes: RetinaNetLite(FeaturePyramidNetworkLite(EfficientNetB0()), num_classes),
        'EfficientNetB1-FPNLite-RetinaNetLite': lambda num_classes: RetinaNetLite(FeaturePyramidNetworkLite(EfficientNetB1()), num_classes),
        'EfficientNetB2-FPNLite-RetinaNetLite': lambda num_classes: RetinaNetLite(FeaturePyramidNetworkLite(EfficientNetB2()), num_classes),
        'EfficientNetB3-FPNLite-RetinaNetLite': lambda num_classes: RetinaNetLite(FeaturePyramidNetworkLite(EfficientNetB3()), num_classes),
        'EfficientNetB4-FPNLite-RetinaNetLite': lambda num_classes: RetinaNetLite(FeaturePyramidNetworkLite(EfficientNetB4()), num_classes),
        'EfficientNetB5-FPNLite-RetinaNetLite': lambda num_classes: RetinaNetLite(FeaturePyramidNetworkLite(EfficientNetB5()), num_classes),
        'EfficientNetB6-FPNLite-RetinaNetLite': lambda num_classes: RetinaNetLite(FeaturePyramidNetworkLite(EfficientNetB6()), num_classes),
        'EfficientNetB7-FPNLite-RetinaNetLite': lambda num_classes: RetinaNetLite(FeaturePyramidNetworkLite(EfficientNetB7()), num_classes),
        'MobileNetV2-FPNLite-RetinaNetLite': lambda num_classes: RetinaNetLite(FeaturePyramidNetworkLite(MobileNetV2()), num_classes),
        'MobileNetV3-FPNLite-RetinaNetLite': lambda num_classes: RetinaNetLite(FeaturePyramidNetworkLite(MobileNetV3()), num_classes),
        'MobileNetV3Small-FPNLite-RetinaNetLite': lambda num_classes: RetinaNetLite(FeaturePyramidNetworkLite(MobileNetV3Small()), num_classes),

        'EfficientNetB0-MnasFPN-RetinaNetLite': lambda num_classes: RetinaNetLite(MnasFPN(EfficientNetB0()), num_classes),
        'EfficientNetB1-MnasFPN-RetinaNetLite': lambda num_classes: RetinaNetLite(MnasFPN(EfficientNetB1()), num_classes),
        'EfficientNetB2-MnasFPN-RetinaNetLite': lambda num_classes: RetinaNetLite(MnasFPN(EfficientNetB2()), num_classes),
        'EfficientNetB3-MnasFPN-RetinaNetLite': lambda num_classes: RetinaNetLite(MnasFPN(EfficientNetB3()), num_classes),
        'EfficientNetB4-MnasFPN-RetinaNetLite': lambda num_classes: RetinaNetLite(MnasFPN(EfficientNetB4()), num_classes),
        'EfficientNetB5-MnasFPN-RetinaNetLite': lambda num_classes: RetinaNetLite(MnasFPN(EfficientNetB5()), num_classes),
        'EfficientNetB6-MnasFPN-RetinaNetLite': lambda num_classes: RetinaNetLite(MnasFPN(EfficientNetB6()), num_classes),
        'EfficientNetB7-MnasFPN-RetinaNetLite': lambda num_classes: RetinaNetLite(MnasFPN(EfficientNetB7()), num_classes),
        'MobileNetV2-MnasFPN-RetinaNetLite': lambda num_classes: RetinaNetLite(MnasFPN(MobileNetV2()), num_classes),
        'MobileNetV3-MnasFPN-RetinaNetLite': lambda num_classes: RetinaNetLite(MnasFPN(MobileNetV3()), num_classes),
        'MobileNetV3Small-MnasFPN-RetinaNetLite': lambda num_classes: RetinaNetLite(MnasFPN(MobileNetV3Small()), num_classes),

        'EfficientDetD0': lambda num_classes: RetinaNet(BidirectionalFeaturePyramidNetwork(EfficientNetB0(), 64, 2), num_classes, num_blocks=3),
        'EfficientDetD1': lambda num_classes: RetinaNet(BidirectionalFeaturePyramidNetwork(EfficientNetB1(), 88, 3), num_classes, num_blocks=3),
        'EfficientDetD2': lambda num_classes: RetinaNet(BidirectionalFeaturePyramidNetwork(EfficientNetB2(), 112, 4), num_classes, num_blocks=3),
        'EfficientDetD3': lambda num_classes: RetinaNet(BidirectionalFeaturePyramidNetwork(EfficientNetB3(), 160, 5), num_classes, num_blocks=4),
        'EfficientDetD4': lambda num_classes: RetinaNet(BidirectionalFeaturePyramidNetwork(EfficientNetB4(), 224, 6), num_classes, num_blocks=4),
        'EfficientDetD5': lambda num_classes: RetinaNet(BidirectionalFeaturePyramidNetwork(EfficientNetB5(), 288, 7), num_classes, num_blocks=4),
        'EfficientDetD6': lambda num_classes: RetinaNet(BidirectionalFeaturePyramidNetwork(EfficientNetB6(), 384, 8), num_classes, num_blocks=5),
        'EfficientDetD7': lambda num_classes: RetinaNet(BidirectionalFeaturePyramidNetwork(EfficientNetB6(), 384, 8), num_classes, num_blocks=5),
    }

    RECOMMENDED_INPUT_SIZES = {
        'EfficientNetB0': 224,
        'EfficientNetB1': 240,
        'EfficientNetB2': 260,
        'EfficientNetB3': 300,
        'EfficientNetB4': 380,
        'EfficientNetB5': 456,
        'EfficientNetB6': 528,
        'EfficientNetB7': 600,
        'MobileNetV2': 224,
        'MobileNetV3': 224,
        'MobileNetV3Small': 224,
        'ResNext14': 224,
        'ResNext26': 224,
        'ResNext50': 224,
        'ResNext101': 224,
        'SEResNext14': 224,
        'SEResNext26': 224,
        'SEResNext50': 224,
        'SEResNext101': 224,
        'ShuffleNet': 224,
        'ShuffleNetV2': 224,
        'SqueezeNet': 224,
        'VGG16': 224,
        'MobileNetV2-SSDLite': 320,
        'MobileNetV2-Sigmoid-SSDLite': 320,
        'MobileNetV3-SSDLite': 320,
        'MobileNetV3Small-SSDLite': 320,
        'SEResNext50-SSDLite': 320,
        'SEResNext101-SSDLite': 320,
        'ShuffleNetV2-SSDLite': 320,
        'VGG16-SSDLite': 320,
        'EfficientDetD0': 512,
        'EfficientDetD1': 640,
        'EfficientDetD2': 768,
        'EfficientDetD3': 896,
        'EfficientDetD4': 1024,
        'EfficientDetD5': 1280,
        'EfficientDetD6': 1480,
        'EfficientDetD7': 1536
    }

    @staticmethod
    def create(model_name, num_classes, options=[]):
        model = None
        creator = ModelFactory.PREDEFINED_MODELS.get(model_name)
        if not creator:
            raise NotImplementedError(f"Unknown model name: {model_name}")

        if options:
            raise NotImplementedError("Options are deprecated.")

        model = creator(num_classes)
        model.reset_parameters()

        if model_name in ModelFactory.RECOMMENDED_INPUT_SIZES:
            model.INPUT_SIZE = ModelFactory.RECOMMENDED_INPUT_SIZES[model_name]

        return model
