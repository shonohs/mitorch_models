import argparse
import os
import tempfile
import torch
from shtorch.models import ModelFactory

def get_model_size(model):
    state_dict = model.state_dict()
    with tempfile.NamedTemporaryFile() as file:
        torch.save(state_dict, file.name)
        return os.path.getsize(file.name) // 1024

def benchmark_model(model_name, verbose):
    model = ModelFactory.create(model_name, 1)
    model_size_1 = get_model_size(model)

    model = ModelFactory.create(model_name, 100)
    model_size_100 = get_model_size(model)

    print(f"Model size: 1 class: {model_size_1}KB, 100 classes: {model_size_100}KB")

    if verbose:
        state_dict = model.state_dict()
        print("=== state_dict shapes ===")
        for key in state_dict:
            print(f"{key}: shape: {state_dict[key].shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    benchmark_model(args.model_name, args.verbose)


if __name__ == '__main__':
    main()
