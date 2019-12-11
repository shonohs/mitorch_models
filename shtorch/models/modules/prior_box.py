import math
import torch
from .base import ModuleBase


class PriorBox(ModuleBase):
    def __init__(self, aspect_ratios=[[2], [2,3], [2,3], [2,3], [2,3], [2]]):
        super(PriorBox, self).__init__()
        self.aspect_ratios = aspect_ratios
        self.min_size = 0.1
        self.max_size = 0.9

    def get_num_priors(self):
        return [len(ar) * 2 + 2 for ar in self.aspect_ratios]

    def forward(self, input):
        assert len(input) == len(self.aspect_ratios)

        # Get spatial shape of each features.
        shapes = [i[0].shape[2:] for i in input]

        prior_boxes = []
        for i, shape in enumerate(shapes):
            for y in range(shape[0]):
                for x in range(shape[1]):
                    cx = (x + 0.5) / shape[1]
                    cy = (y + 0.5) / shape[0]

                    if len(shapes) == 1:
                        s_k = self.max_size
                        s_k2 = self.max_size * 2 - self.min_size
                    else:
                        s_k = self.min_size + (self.max_size - self.min_size) / (len(shapes) - 1) * i
                        s_k2 = self.min_size + (self.max_size - self.min_size) / (len(shapes) - 1) * (i + 1)

                    s_k_prime = math.sqrt(s_k * s_k2)
                    prior_boxes.append(PriorBox._to_corners(cx, cy, s_k, s_k))
                    prior_boxes.append(PriorBox._to_corners(cx, cy, s_k_prime, s_k_prime))

                    for aspect_ratio in self.aspect_ratios[i]:
                        sqrt_ar = math.sqrt(aspect_ratio)
                        prior_boxes.append(PriorBox._to_corners(cx, cy, s_k * sqrt_ar, s_k / sqrt_ar))
                        prior_boxes.append(PriorBox._to_corners(cx, cy, s_k / sqrt_ar, s_k * sqrt_ar))

        prior_boxes = torch.Tensor(prior_boxes).detach().to(input[0][0].device)
        return torch.clamp(prior_boxes, min=0, max=1)

    @staticmethod
    def _to_corners(cx, cy, w, h):
        x = cx - w / 2
        y = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return ([x, y, x2, y2])
