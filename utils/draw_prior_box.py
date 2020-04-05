import argparse
import random
import sys
import torch
import PIL.Image
import PIL.ImageDraw
import mitorch.models.modules  # noqa: F401


class BoundingBoxDrawer(object):
    COLOR_CODES = ["black", "brown", "red", "orange", "yellow", "green", "blue", "violet", "grey", "white"]

    def __init__(self, image):
        self.width, self.height = image.size
        self.draw = PIL.ImageDraw.Draw(image)

    def draw_predictions(self, class_id, probability, x0, y0, x1, y1):
        color = self.COLOR_CODES[class_id % len(self.COLOR_CODES)]
        self.draw_rectangle(x0, y0, x1, y1, color)

    def draw_rectangle(self, x0, y0, x1, y1, color="red"):
        self.draw.rectangle(((x0 * self.width, y0 * self.height), (x1 * self.width, y1 * self.height)), outline=color)


def draw_prior_box(prior_box, size, output_filename, show_random):
    input = [[torch.zeros((1, 1, s, s))] for s in size]
    boxes = prior_box(input)

    image = PIL.Image.new('RGB', (512, 512))
    drawer = BoundingBoxDrawer(image)
    draw_boxes = random.sample(boxes.tolist(), 10) if show_random else boxes
    for box in draw_boxes:
        drawer.draw_predictions(random.randint(0, 10), 0, *box)

    image.save(output_filename)
    print(f"{len(draw_boxes)} out of {len(boxes)} boxes are drawn")


def main():
    parser = argparse.ArgumentParser("Draw prior boxes on an image")
    parser.add_argument('prior_box', type=str, help='PriorBox class name')
    parser.add_argument('--size', type=int, nargs='+', default=[64, 32, 16, 8, 4], help="The size of the prior boxes")
    parser.add_argument('--output', type=str, default='prior_box.png', help="Output image filename")
    parser.add_argument('--random', action='store_true', help="Show ramdonly picked boxes")
    args = parser.parse_args()
    prior_box_class = getattr(sys.modules['mitorch.models.modules'], args.prior_box)
    prior_box = prior_box_class(len(args.size))

    draw_prior_box(prior_box, args.size, args.output, args.random)


if __name__ == '__main__':
    main()
