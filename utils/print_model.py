import argparse
from mitorch.models import ModelFactory


def print_model(model_name, num_classes):
    model = ModelFactory.create(model_name, num_classes)
    print(model)


def main():
    parser = argparse.ArgumentParser(description="Print a model")
    parser.add_argument('model_name')
    parser.add_argument('num_classes', nargs='?', type=int, default=1)

    args = parser.parse_args()
    print_model(args.model_name, args.num_classes)


if __name__ == '__main__':
    main()
