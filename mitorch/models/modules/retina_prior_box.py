import math
from .prior_box import PriorBox


class RetinaPriorBox(PriorBox):
    def __init__(self, num_scales, aspect_ratios=[0.5, 1, 2], anchor_scales=[1, 1.25999, 1.58740]):
        super().__init__(num_scales)
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.anchor_scales = anchor_scales
        self.min_size = 32 / 600

    def get_num_priors(self):
        return [len(self.aspect_ratios) * len(self.anchor_scales)] * self.num_scales

    def _make_boxes(self, cx, cy, index):
        size = self.min_size * (2 ** index)

        prior_boxes = []
        for aspect_ratio in self.aspect_ratios:
            sqrt_ar = math.sqrt(aspect_ratio)
            for scale in self.anchor_scales:
                prior_boxes.append(PriorBox._to_corners(cx, cy, size * scale * sqrt_ar, size * scale / sqrt_ar))

        return prior_boxes
