from .model import Model
from .modules import Linear


class Classifier(Model):
    def __init__(self, model, num_classes):
        super().__init__(num_classes)
        self.base_model = model
        self.classifier = Linear(model.output_dim, num_classes)

    def forward(self, input):
        return self.classifier(self.base_model(input))
