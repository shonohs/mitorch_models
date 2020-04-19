import math
import torch
from .base import ModuleBase


class PriorBox(ModuleBase):
    def __init__(self, num_scales, aspect_ratios=[2, 3]):
        super().__init__()
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.min_size = 0.1
        self.max_size = 0.9

    def get_num_priors(self):
        return [len(self.aspect_ratios) * 2 + 2] * self.num_scales

    def forward(self, input):
        assert len(input) == self.num_scales
        # Get spatial shape of each features.
        shapes = [i[0].shape[2:] for i in input]
        assert [s[0] for s in shapes] == sorted([s[0] for s in shapes], reverse=True)

        prior_boxes = []

        for i, shape in enumerate(shapes):
            for y in range(shape[0]):
                for x in range(shape[1]):
                    cx = (x + 0.5) / shape[1]
                    cy = (y + 0.5) / shape[0]
                    prior_boxes.extend(self._make_boxes(cx, cy, i))

        prior_boxes = torch.Tensor(prior_boxes).detach().to(input[0][0].device)
        return torch.clamp(prior_boxes, min=0, max=1)

    def _make_boxes(self, cx, cy, index):
        if self.num_scales == 1:
            s_k = self.max_size
            s_k2 = self.max_size * 2 - self.min_size
        else:
            s_k = self.min_size + (self.max_size - self.min_size) / (self.num_scales - 1) * index
            s_k2 = self.min_size + (self.max_size - self.min_size) / (self.num_scales - 1) * (index + 1)

        s_k_prime = math.sqrt(s_k * s_k2)
        prior_boxes = [PriorBox._to_corners(cx, cy, s_k, s_k),
                       PriorBox._to_corners(cx, cy, s_k_prime, s_k_prime)]

        for aspect_ratio in self.aspect_ratios:
            sqrt_ar = math.sqrt(aspect_ratio)
            prior_boxes.append(PriorBox._to_corners(cx, cy, s_k * sqrt_ar, s_k / sqrt_ar))
            prior_boxes.append(PriorBox._to_corners(cx, cy, s_k / sqrt_ar, s_k * sqrt_ar))

        return prior_boxes

    @staticmethod
    def _to_corners(cx, cy, w, h):
        x = cx - w / 2
        y = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return ([x, y, x2, y2])
