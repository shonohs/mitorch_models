from ..model import Model


class Head(Model):
    """Networks to extract features from base model.
    Input and output is single or multi scale features.
    """
    BASE_FEATURE_MAPS = {'EfficientNetB0': {5: 'features.conv1', 4: 'features.block5_0.conv0', 3: 'features.block3_0.conv0'},
                         'EfficientNetB1': {5: 'features.conv1', 4: 'features.block5_0.conv0', 3: 'features.block3_0.conv0'},
                         'EfficientNetB2': {5: 'features.conv1', 4: 'features.block5_0.conv0', 3: 'features.block3_0.conv0'},
                         'EfficientNetB3': {5: 'features.conv1', 4: 'features.block5_0.conv0', 3: 'features.block3_0.conv0'},
                         'EfficientNetB4': {5: 'features.conv1', 4: 'features.block5_0.conv0', 3: 'features.block3_0.conv0'},
                         'EfficientNetB5': {5: 'features.conv1', 4: 'features.block5_0.conv0', 3: 'features.block3_0.conv0'},
                         'EfficientNetB6': {5: 'features.conv1', 4: 'features.block5_0.conv0', 3: 'features.block3_0.conv0'},
                         'EfficientNetB7': {5: 'features.conv1', 4: 'features.block5_0.conv0', 3: 'features.block3_0.conv0'},
                         'MobileNetV2': {5: 'features.conv1', 4: 'features.block5_0.conv0', 3: 'features.block3_0.conv0'},
                         'MobileNetV3': {5: 'features.conv1', 4: 'features.block4_0.conv0', 3: 'features.block3_0.conv0'},
                         'MobileNetV3Small': {5: 'features.conv1', 4: 'features.block3_0.conv0', 3: 'features.block2_0.conv0'},
                         'SEResNext50': {5: 'features.block3_2', 4: 'features.block3_0.conv0', 3: 'features.block2_0.conv0'},
                         'SEResNext101': {5: 'features.block3_2', 4: 'features.block3_0.conv0', 3: 'features.block2_0.conv0'},
                         'ShuffleNetV2': {5: 'features.conv1', 4: 'features.block2_0.conv0', 3: 'features.block1_0.conv0'},
                         'VGG16ForSSD': {5: 'features.conv5_1', 4: 'features.conv3_2', 3: 'features.conv2_2'}}

    def __init__(self, base_model, input_scales, output_channels):
        super().__init__(output_channels)
        self.base_model = base_model
        self.base_feature_names = Head.get_base_feature_names(base_model, input_scales)

    def get_base_features(self, input):
        return self.base_model.forward(input, self.base_feature_names)

    @staticmethod
    def get_base_feature_names(base_model, input_scales):
        base_feature_maps = Head.BASE_FEATURE_MAPS[type(base_model).__name__]
        if not base_feature_maps:
            raise NotImplementedError(f"Unsupported model {type(base_model).__name__}")

        base_feature_names = [base_feature_maps.get(i, None) for i in input_scales]
        if None in base_feature_names:
            raise NotImplementedError(f"Unsupported input scales {input_scales}for the model {type(base_model).__name__}")

        return base_feature_names

    @classmethod
    def get_base_output_shapes(cls, base_model, input_scales):
        base_feature_names = Head.get_base_feature_names(base_model, input_scales)
        return base_model.get_output_shapes(base_feature_names)
