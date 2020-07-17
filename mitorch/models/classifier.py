import torch
from .model import Model
from .modules import Linear, default_module_settings, get_module_settings


class Classifier(Model):
    @default_module_settings(multilabel=False)
    def __init__(self, model, num_classes):
        super().__init__(num_classes)
        settings = get_module_settings()
        is_multiclass = not settings['multilabel']

        self.base_model = model
        self.classifier = Linear(model.output_dim, num_classes)
        self.loss = torch.nn.CrossEntropyLoss() if is_multiclass else torch.nn.BCEWithLogitsLoss()
        self.predictor = torch.nn.Softmax(dim=1) if is_multiclass else torch.nn.Sigmoid()

    def forward(self, input):
        return self.classifier(self.base_model(input))
