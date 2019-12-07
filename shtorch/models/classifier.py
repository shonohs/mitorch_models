class Classifier(Model):
    def __init__(self, model, num_classes):
        self.classifier = torch.nn.Linear(model.output_dim, num_classes)

    def forward(self, input):
        return self.classifier(input)
