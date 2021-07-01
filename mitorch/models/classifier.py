import torch
from .model import Model
from .modules import Linear


class Classifier(Model):
    def __init__(self, model, num_classes, multilabel=False):
        super().__init__(num_classes)
        is_multiclass = not multilabel

        self.base_model = model
        self.classifier = Linear(model.output_dim, num_classes)
        self.loss = torch.nn.CrossEntropyLoss() if is_multiclass else torch.nn.BCEWithLogitsLoss()
        self.predictor = torch.nn.Softmax(dim=1) if is_multiclass else torch.nn.Sigmoid()

    def forward(self, input):
        return self.classifier(self.base_model(input))
