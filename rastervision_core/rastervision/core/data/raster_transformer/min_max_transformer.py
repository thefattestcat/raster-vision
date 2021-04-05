import cv2
import numpy as np

from rastervision.core.data.raster_transformer.raster_transformer \
    import RasterTransformer


class MinMaxRasterTransformer(RasterTransformer):
    def __init__(self):
        from rastervision.pytorch_learner.utils import MinMaxNormalize
        self.normalize = MinMaxNormalize(0, 255, cv2.CV_8U)

    def transform(self, chip, channel_order=None):
        return self.normalize(image=chip)['image']
