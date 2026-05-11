import logging
import os
import os.path as osp

import cv2
import imageio
import numpy as np
import torch


def _cfg_get(cfg, key, default=None):
    return getattr(cfg, key, default) if cfg is not None else default


def _empty_dirname(dirname):
    return dirname is None or str(dirname).lower() in {"", "none", "null", "false"}


def load_external_dynamic_masks(
    ws,
    frame_names,
    H,
    W,
    mask_dirname="sam3_dymask",
    threshold=0.5,
    required=True,
    device=None,
    erode_ksize=0,
    dilate_ksize=0,
):
    """Load per-frame binary dynamic masks from a scene-local mask folder."""
    if _empty_dirname(mask_dirname):
        return None

    mask_root = osp.join(ws, str(mask_dirname))
    mask_dir = osp.join(mask_root, "mask") if osp.isdir(osp.join(mask_root, "mask")) else mask_root
    if not osp.isdir(mask_dir):
        msg = f"External dynamic mask dir not found: {mask_dir}"
        if required:
            raise FileNotFoundError(msg)
        logging.warning(msg)
        return None

    mask_fns = [
        osp.join(mask_dir, fn)
        for fn in os.listdir(mask_dir)
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    ]
    mask_fns.sort()
    by_stem = {osp.splitext(osp.basename(fn))[0]: fn for fn in mask_fns}
    ordered = [by_stem.get(name) for name in frame_names]

    if any(fn is None for fn in ordered):
        if len(mask_fns) == len(frame_names):
            missing = [name for name, fn in zip(frame_names, ordered) if fn is None][:5]
            logging.warning(
                "External mask names do not fully match frame names; "
                f"falling back to sorted order. Example missing stems: {missing}"
            )
            ordered = mask_fns
        else:
            missing = [name for name, fn in zip(frame_names, ordered) if fn is None][:10]
            raise FileNotFoundError(
                f"Cannot align {len(mask_fns)} masks to {len(frame_names)} frames in {mask_dir}. "
                f"Example missing stems: {missing}"
            )

    masks = []
    for fn in ordered:
        mask = imageio.imread(fn)
        if mask.ndim == 3:
            mask = mask[..., 3] if mask.shape[-1] == 4 else mask.max(axis=-1)
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        if threshold <= 1.0 and mask.max() > 1:
            mask = mask.astype(np.float32) / 255.0
        dyn = mask > threshold

        dyn_u8 = dyn.astype(np.uint8)
        if erode_ksize and erode_ksize > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize))
            dyn_u8 = cv2.erode(dyn_u8, kernel, iterations=1)
        if dilate_ksize and dilate_ksize > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
            dyn_u8 = cv2.dilate(dyn_u8, kernel, iterations=1)
        masks.append(dyn_u8 > 0)

    ret = torch.from_numpy(np.stack(masks, axis=0)).bool()
    if device is not None:
        ret = ret.to(device)
    logging.info(
        f"Loaded external dynamic masks from {mask_dir}: "
        f"shape={tuple(ret.shape)}, dynamic_pixels={ret.sum().item()}"
    )
    return ret


def load_external_dynamic_masks_from_cfg(ws, s2d, cfg, device=None, prefix=""):
    dirname = _cfg_get(cfg, f"{prefix}dynamic_mask_dirname", None)
    if _empty_dirname(dirname):
        dirname = _cfg_get(cfg, "dynamic_mask_dirname", None)
    if _empty_dirname(dirname):
        return None

    threshold = _cfg_get(
        cfg,
        f"{prefix}dynamic_mask_threshold",
        _cfg_get(cfg, "dynamic_mask_threshold", 0.5),
    )
    required = _cfg_get(
        cfg,
        f"{prefix}dynamic_mask_required",
        _cfg_get(cfg, "dynamic_mask_required", True),
    )
    erode_ksize = _cfg_get(
        cfg,
        f"{prefix}dynamic_mask_erode_ksize",
        _cfg_get(cfg, "dynamic_mask_erode_ksize", 0),
    )
    dilate_ksize = _cfg_get(
        cfg,
        f"{prefix}dynamic_mask_dilate_ksize",
        _cfg_get(cfg, "dynamic_mask_dilate_ksize", 0),
    )
    return load_external_dynamic_masks(
        ws=ws,
        frame_names=s2d.frame_names,
        H=s2d.H,
        W=s2d.W,
        mask_dirname=dirname,
        threshold=threshold,
        required=required,
        device=device,
        erode_ksize=erode_ksize,
        dilate_ksize=dilate_ksize,
    )


def combine_dynamic_masks(base_mask, external_mask, mode="replace"):
    if external_mask is None:
        return base_mask
    mode = str(mode).lower()
    external_mask = external_mask.to(base_mask.device).bool()
    base_mask = base_mask.bool()
    if mode == "replace":
        return external_mask
    if mode in {"or", "union"}:
        return base_mask | external_mask
    if mode in {"and", "intersection", "intersect"}:
        return base_mask & external_mask
    raise ValueError(f"Unknown dynamic mask mode: {mode}")
