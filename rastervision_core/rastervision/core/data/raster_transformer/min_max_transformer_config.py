from rastervision.pipeline.config import register_config, Field
from rastervision.core.data.raster_transformer.raster_transformer_config import (  # noqa
    RasterTransformerConfig)
from rastervision.core.data.raster_transformer.min_max_transformer import (  # noqa
    MinMaxRasterTransformer)

@register_config('min_max_raster_transformer')
class MinMaxRasterTransformerConfig(RasterTransformerConfig):
    def build(self):
        return MinMaxRasterTransformer()
