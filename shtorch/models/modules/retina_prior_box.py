import math
import torch
from .prior_box import PriorBox


class RetinaPriorBox(PriorBox):
    def __init__(self, num_scales, aspect_ratios=[0.5, 1, 2], anchor_scales=[1, 1.25999, 1.58740]):
        super(RetinaPriorBox, self).__init__(num_scales)
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.anchor_scales = anchor_scales
        self.min_size = 32 / 600

    def get_num_priors(self):
        return [len(self.aspect_ratios) * len(self.anchor_scales)] * self.num_scales

    def forward(self, input):
        assert len(input) == self.num_scales

        # Get spatial shape of each features.
        shapes = [i[0].shape[2:] for i in input]

        prior_boxes = []
        for i, shape in enumerate(shapes):
            for y in range(shape[0]):
                for x in range(shape[1]):
                    cx = (x + 0.5) / shape[1]
                    cy = (y + 0.5) / shape[0]

                    size = self.min_size * (2 ** i)

                    for aspect_ratio in self.aspect_ratios:
                        sqrt_ar = math.sqrt(aspect_ratio)
                        for scale in self.anchor_scales:
                            prior_boxes.append(PriorBox._to_corners(cx, cy, size * scale * sqrt_ar, size * scale / sqrt_ar))

        prior_boxes = torch.Tensor(prior_boxes).detach().to(input[0][0].device)
        return torch.clamp(prior_boxes, min=0, max=1)
