import argparse
import pathlib
import torch
from mitorch.models import ModelFactory


class PredictionModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        if hasattr(self.model, 'predictor'):
            x = self.model.predictor(x)
        return x


def export_onnx(model_name, num_classes, weights_filepath, with_predictor, output_filepath):
    model = ModelFactory.create(model_name, num_classes)
    if weights_filepath:
        print(f"Loading {weights_filepath}")
        weights = torch.load(weights_filepath, map_location=torch.device('cpu'))
        model.load_state_dict(weights)

    dummy_input = torch.zeros(1, 3, model.INPUT_SIZE, model.INPUT_SIZE, dtype=torch.float32)
    model = PredictionModel(model) if with_predictor else model
    torch.onnx.export(model, dummy_input, output_filepath, input_names=['input'])

    print(f"Saved to {output_filepath}")


def main():
    parser = argparse.ArgumentParser(description="Export a model in ONNX format.")
    parser.add_argument('model_name')
    parser.add_argument('num_classes', nargs='?', type=int, default=1)
    parser.add_argument('--weights', '-w')
    parser.add_argument('--with_predictor', '-p', action='store_true', help="Include a predictor")
    parser.add_argument('--output_filepath', '-o', type=pathlib.Path, default=None)

    args = parser.parse_args()
    if not args.output_filepath:
        args.output_filepath = pathlib.Path(f'{args.model_name}_{args.num_classes}.onnx')

    if args.output_filepath.exists():
        parser.error(f"Output filepath already exists: {args.output_filepath}")

    export_onnx(args.model_name, args.num_classes, args.weights, args.with_predictor, args.output_filepath)


if __name__ == '__main__':
    main()
