# Planned Modification: SAM3-Guided Dynamic Masks for DyCheck

## Motivation

MoSca currently separates static and dynamic regions mainly through epipolar inconsistency and track-level motion cues. This is useful for discovering motion, but the derived pixel-level dynamic masks can be noisy:

- Static background such as floor regions may be included as dynamic due to depth, optical-flow, camera-pose, or occlusion errors.
- Moving objects can be incomplete because track coverage is sparse and pixel masks are expanded from nearest dynamic curves.
- Thin structures, occlusions, and reappearing object parts are especially unstable.

DyCheck scenes usually have clear dynamic subjects. This makes them suitable for prompt-based video segmentation, where SAM3 can be used to produce cleaner object-level masks for the moving subjects.

## Proposed Direction

Use SAM3 as an object-boundary refinement and propagation module, not as the only source of motion reasoning.

The intended workflow is:

1. For each DyCheck sequence, choose one or more keyframes.
2. Provide prompts for the dynamic subject or subjects in those keyframes.
3. Run SAM3 video segmentation to propagate masks across the sequence.
4. Save the propagated masks under each sequence, for example:

   ```text
   <sequence>/
     dynamic_mask_sam3/
       00000.png
       00001.png
       ...
   ```

5. Modify MoSca data loading or reconstruction code so `dynamic_mask_sam3` can replace or refine the default `s2d.dyn_mask`.

## Integration Options

### Option A: Replace Pixel-Level Dynamic Mask

Use SAM3 masks directly as `s2d.dyn_mask` during Gaussian initialization:

- Static GS initialization uses `~sam3_mask * dep_mask`.
- Dynamic GS initialization uses `sam3_mask * dep_mask`.

This is simple and likely improves object boundaries, but it relies heavily on prompt quality.

### Option B: SAM3 Mask Plus EPI Sanity Filter

Combine SAM3 and MoSca's EPI-derived mask:

- Conservative mode: `dynamic = sam3_mask & epi_dyn_mask`
- Complete mode: `dynamic = sam3_mask | epi_dyn_mask`
- Preferred initial experiment: use `sam3_mask` as the main mask and keep EPI masks for visualization and diagnostics.

This preserves motion evidence while allowing SAM3 to provide cleaner object boundaries.

### Option C: Soft Auxiliary Mask

Instead of hard replacing `s2d.dyn_mask`, keep MoSca's original mask but add a soft consistency loss encouraging the rendered dynamic/static separation to agree with SAM3.

This is safer for training but requires more code changes.

## Expected Benefits

- More complete dynamic object masks.
- Less leakage from floor/background into the dynamic component.
- Cleaner dynamic Gaussian initialization.
- Potentially better novel-view synthesis on dynamic regions, especially for scenes with clear foreground subjects.

## Risks

- SAM3 segments objects, not motion. A prompted static object can be incorrectly treated as dynamic.
- Missing prompts can miss secondary moving objects or carried objects.
- Mask errors can harm reconstruction if used as hard ground truth.
- Occlusion and reappearance still need manual checking or temporal post-processing.

## First Experiment

For DyCheck, generate `dynamic_mask_sam3` for a small set of scenes first:

- `apple`
- `block`
- `paper-windmill`
- `space-out`

Then run MoSca with a minimal code path that allows selecting:

```bash
dynamic_mask_mode=sam3
```

Compare against the current origin results using:

- `mPSNR`
- `mSSIM`
- `mLPIPS`
- `PCK@0.05`
- qualitative rendered videos

The first implementation should avoid changing camera estimation. It should only affect static/dynamic pixel masks used for Gaussian initialization and photometric reconstruction.
