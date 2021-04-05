# flake8: noqa

from typing import Optional
import re
import random
import os
from abc import abstractmethod

import pandas as pd

from rastervision.pipeline.file_system import list_paths
from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *


def build_scene(image_uri, label_uri,
                id: str,
                channel_order: Optional[list] = None) -> SceneConfig:
    raster_source = RasterioSourceConfig(
        uris=[image_uri], channel_order=channel_order,
        transformers=[MinMaxRasterTransformerConfig()])

    # Set a line buffer to convert line strings to polygons.
    vector_source = GeoJSONVectorSourceConfig(
        uri=label_uri,
        default_class_id=0,
        ignore_crs_field=True,
        line_bufs={0: 15})
    label_source = SemanticSegmentationLabelSourceConfig(
        raster_source=RasterizedSourceConfig(
            vector_source=vector_source,
            rasterizer_config=RasterizerConfig(background_class_id=1)))

    label_store = SemanticSegmentationLabelStoreConfig(
        vector_output=[PolygonVectorOutputConfig(class_id=0, denoise=3)])

    return SceneConfig(
        id=id,
        raster_source=raster_source,
        label_source=label_source,
        label_store=label_store)


def get_config(runner,
               raw_uri: str,
               spacenet_csv_uri: str,
               root_uri: str,
               test_aoi: str = 'all',
               train_sz: float = 1.0,
               test: bool = False) -> SemanticSegmentationConfig:
    channel_order = [0, 1, 2]
    class_config = ClassConfig(
        names=['building', 'background'], colors=['orange', 'black'])
    train_sz = float(train_sz)

    df = pd.read_csv(spacenet_csv_uri)
    train_scenes = []
    val_scenes = []
    if test:
        df = df.subset(5)
        df['splits'] = ['train'] * 4 + ['val']

    for _, row in df.iterrows():
        scene_id = row['scene_id']
        image_uri = row['image_uri']
        label_uri = row['label_uri']
        scene = build_scene(image_uri, label_uri, scene_id, channel_order)
        if row['split'] == 'train':
            train_scenes.append(scene)
        else:
            val_scenes.append(scene)

    scene_dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=train_scenes,
        validation_scenes=val_scenes)

    chip_sz = 217
    img_sz = chip_sz

    chip_options = SemanticSegmentationChipOptions(
        window_method=SemanticSegmentationWindowMethod.sliding, stride=chip_sz)
    data = SemanticSegmentationImageDataConfig(
        img_sz=img_sz, num_workers=4, group_train_sz_rel=train_sz)
    backend = PyTorchSemanticSegmentationConfig(
        data=data,
        model=SemanticSegmentationModelConfig(backbone=Backbone.resnet50),
        solver=SolverConfig(
            lr=1e-4,
            num_epochs=5,
            test_num_epochs=2,
            batch_sz=8,
            one_cycle=True),
        log_tensorboard=False,
        run_tensorboard=False,
        test_mode=test)

    return SemanticSegmentationConfig(
        root_uri=root_uri,
        dataset=scene_dataset,
        backend=backend,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz,
        chip_options=chip_options,
        img_format='png')
