#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import print_function

import numpy as np
from PIL import Image


# Define the trigger appending transformation
class TriggerAppending(object):
    """
    Args:
         trigger: the trigger pattern (image size)
         alpha: the blended hyper-parameter (image size)
         x_poisoned = (1-alpha)*x_benign + alpha*trigger
    """

    def __init__(self, trigger, alpha):
        self.trigger = np.array(trigger.clone().detach().permute(
            1, 2, 0) * 255)  # trigger in [0,1]^d
        self.alpha = np.array(alpha.clone().detach().permute(1, 2, 0))

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        img_ = np.array(img).copy()
        img_ = (1 - self.alpha) * img_ + self.alpha * self.trigger

        return Image.fromarray(img_.astype('uint8')).convert('RGB')
