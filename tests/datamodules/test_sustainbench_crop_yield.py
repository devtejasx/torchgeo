# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import torch

from torchgeo.datamodules import SustainBenchCropYieldDataModule


def test_already_scaled_images_are_not_divided_by_255() -> None:
    root = os.path.join('tests', 'data', 'sustainbench_crop_yield')
    dm = SustainBenchCropYieldDataModule(root=root, batch_size=2)
    dm.setup('fit')
    batch = next(iter(dm.train_dataloader()))
    image = batch['image'].clone()
    out = dm.aug(batch)
    torch.testing.assert_close(out['image'], image)
