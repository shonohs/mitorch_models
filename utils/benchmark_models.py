import argparse
import os
import tempfile
import time
import torch
from mitorch.models import ModelFactory


def get_model_size(model):
    state_dict = model.state_dict()
    with tempfile.NamedTemporaryFile() as file:
        torch.save(state_dict, file.name)
        return os.path.getsize(file.name) // 1024 / 1024


def get_inference_time(model, input_size, batch_size, num_repeats=10, with_gpu=False):
    device = torch.device('cuda' if with_gpu else 'cpu')
    input = torch.randn((batch_size, 3, input_size, input_size), dtype=torch.float, device=device)
    model = model.to(device)

    start_time = time.time()
    for i in range(num_repeats):
        features = model(input)
        if hasattr(model, 'predictor'):
            model.predictor(features)
        if with_gpu:
            torch.cuda.synchronize()
    end_time = time.time()

    return (end_time - start_time) / num_repeats


def get_and_print_inference_time(model):
    for input_size in [224, 500]:
        for batch_size in [1, 10, 32]:
            cpu_time = get_inference_time(model, input_size, batch_size, with_gpu=False)
            if torch.cuda.is_available():
                gpu_time = get_inference_time(model, input_size, batch_size, with_gpu=True, num_repeats=100)
            print(f"Inference: input_size={input_size}, batch_size={batch_size}: cpu: {cpu_time}s" + (f"gpu: {gpu_time}s" if torch.cuda.is_available() else ""))


def benchmark_model_per_num_classes(model_name, num_classes, measure_inference):
    print(f"==== {num_classes} class ====")
    model = ModelFactory.create(model_name, num_classes)
    model_size = get_model_size(model)
    print(f"Model size: {model_size:.1f}MB")

    if measure_inference:
        get_and_print_inference_time(model)
    return model


def benchmark_model(model_name, measure_inference, verbose):
    model = benchmark_model_per_num_classes(model_name, 1, measure_inference)
    benchmark_model_per_num_classes(model_name, 100, measure_inference)
    benchmark_model_per_num_classes(model_name, 1000, measure_inference)

    if verbose:
        state_dict = model.state_dict()
        print("=== state_dict shapes ===")
        for key in state_dict:
            print(f"{key}: shape: {state_dict[key].shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('--measure_inference', '-i', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    benchmark_model(args.model_name, args.measure_inference, args.verbose)


if __name__ == '__main__':
    main()
