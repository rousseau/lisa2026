"""nnU-Net v2 exact preprocessor wrapper for LISA 2026.

This module replicates the ``DefaultPreprocessor`` used by the native
nnU-Net trainer (RUN_0003a) so that our multi-task pipeline sees exactly the
same inputs.  It handles:
* transpose (no-op for this dataset)
* crop_to_nonzero
* ZScoreNormalization (use_mask_for_norm=False)
* resample to isotropic 1 mm spacing
* label remapping via LabelManager
"""
from __future__ import annotations

import os
import warnings
from typing import Sequence, Hashable, Mapping

import numpy as np
import torch
from monai.transforms.transform import MapTransform

# silence nnU-Net path warnings until we actually need them
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

from .nnunetv2_plan import NNUNET_PLANS, NNUNET_DATASET_JSON

# Lazy-initialised singletons so that the heavy import is done once.
_plans_manager: PlansManager | None = None
_cfg_manager: ConfigurationManager | None = None
_preprocessor: DefaultPreprocessor | None = None


def _init():
    """Initialise plans, configuration manager and preprocessor."""
    global _plans_manager, _cfg_manager, _preprocessor
    if _plans_manager is not None:
        return
    _plans_manager = PlansManager(NNUNET_PLANS)
    _cfg_manager = ConfigurationManager(NNUNET_PLANS["configurations"]["3d_fullres"])
    _preprocessor = DefaultPreprocessor(verbose=False)


def preprocess_volume(image_path: str, seg_path: str | None = None):
    """Run exact nnU-Net preprocessing for one case.

    Parameters
    ----------
    image_path: absolute path to the NIfTI image (single channel).
    seg_path: absolute path to the corresponding label NIfTI (or *None* for
        inference / image-only tasks).

    Returns
    -------
    data : np.ndarray, shape (1, H, W, D), float32
        Pre-processed image volume.
    seg : np.ndarray | None
        Pre-processed segmentation (or *None* when ``seg_path`` is *None*).
    props : dict
        Properties dict containing ``spacing``, ``bbox_used_for_cropping``,
        ``shape_before_cropping``, etc.
    """
    _init()
    # nnU-Net expects a list of image files (one per modality / channel).
    image_files = [image_path]
    data, seg, props = _preprocessor.run_case(
        image_files,
        seg_file=seg_path,
        plans_manager=_plans_manager,
        configuration_manager=_cfg_manager,
        dataset_json=NNUNET_DATASET_JSON,
    )
    return data, seg, props


def resample_probabilities_to_shape(probs: np.ndarray, target_shape: Sequence[int],
                                   current_spacing: Sequence[float],
                                   target_spacing: Sequence[float]) -> np.ndarray:
    """Resample probability/logit maps from current shape to target shape.

    Uses the exact same function and parameters as nnU-Net inference
    (``resample_data_or_seg_to_shape``, order=1, order_z=0).
    """
    _init()
    # nnU-Net stores the resampling function reference in the configuration manager.
    fn = _cfg_manager.resampling_fn_probabilities
    kwargs = _cfg_manager.resampling_fn_probabilities_kwargs
    # The nnUNet function expects ``(C, H, W, D)`` arrays.
    return fn(probs, target_shape, current_spacing, target_spacing, **kwargs)


class LoadAndPreprocessNnunetd(MapTransform):
    """MONAI-style transform that runs the full nnU-Net ``DefaultPreprocessor``.

    Replaces ``LoadImaged`` + ``EnsureChannelFirstd`` + ``NormalizeIntensityd``.
    The output ``img`` is a ``(1, H, W, D)`` float32 numpy array and ``label``
    is ``(1, H, W, D)`` long np array (when available).
    Additional key ``nnunet_properties`` stores the properties dict so that
    the evaluator can inverse-resample predictions.
    """

    def __init__(self, keys: Sequence[str] = ("img", "label"), allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        # We only use "img" and "label" keys.
        self._img_key = "img"
        self._label_key = "label"

    def __call__(self, data: Mapping[Hashable, object]) -> dict:
        d = dict(data)
        img_path = d[self._img_key]
        seg_path = d.get(self._label_key)
        pp_img, pp_seg, props = preprocess_volume(str(img_path), str(seg_path) if seg_path is not None else None)
        # nnU-Net returns shape (C, H, W, D) – already channel-first.
        d[self._img_key] = pp_img.astype(np.float32)
        if pp_seg is not None:
            # DefaultPreprocessor sets cropped background voxels to -1; convert back to 0
            # so one-hot losses do not crash on negative indices.
            pp_seg = np.where(pp_seg < 0, 0, pp_seg).astype(np.int16)
            d[self._label_key] = pp_seg
        d["nnunet_properties"] = props
        return d
