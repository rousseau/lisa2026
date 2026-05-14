# RUN_0004 – Notes

## Status

**Planned** – code is implemented, not yet executed.

---

## Design decisions

### Why self-supervised denoising?

The challenge does not guarantee paired (degraded, clean) acquisitions for Task 1b. A self-supervised approach avoids this dependency entirely: the model sees (noisy_input → clean_target) pairs constructed on-the-fly, using only the clean volumes that are already available. This is a well-established technique (Noise2Clean, N2N-style) and constitutes a valid baseline even without any paired data.

### Why BasicUNet (not DynUNet)?

DynUNet is optimised for segmentation (large multi-scale receptive field, learnable strides). For image-to-image reconstruction, BasicUNet with symmetric encoder-decoder and skip connections is the standard choice. It is lighter and better suited to reconstruction losses (L1, SSIM).

### Why 96³ crops (vs 128³ in Task 2)?

Memory budget: at batch_size=2 with AMP, a 96³ BasicUNet with features up to 256 fits comfortably in 16 GB VRAM. The 96³ size still captures the global anatomical context of the low-field volumes.

### Split reuse strategy

`task1a_fixed.pkl` stores `train_indices` / `val_indices` (row indices into the Task1a CSV). Since Task 1b uses a filesystem scan (no CSV), the `_subjects_from_split` helper replicates the train fraction on the sorted subject list. This ensures the same ≈80/20 patient split without requiring the original CSV at runtime.

### Loss: L1 + SSIM

L1 provides pixel-accurate reconstruction; SSIM adds structural perceptual quality. Both are equally weighted (1.0) as a starting point. Future runs may tune the ratio or add a VGG perceptual term.

---

## Known limitations

- Synthetic noise (Gaussian only) may not fully replicate all real 0.064T artifact types (Rician noise, motion, banding, etc.).
- The FID metric computed on the validation set is sensitive to sample size; with few validation subjects, the covariance estimate may be noisy.
- PSNR is computed using the model's own range normalisation, which may not match the challenge's exact normalisation protocol.

---

## To-do before execution

- [ ] Verify that `_ciso.nii.gz` files exist in `/home/rousseau/Data/LISA2026`.
- [ ] Confirm `task1a_fixed.pkl` or `task2_fixed.pkl` is available.
- [ ] Run smoke test: `bash scripts/run_0004.sh --smoke-test`.
- [ ] Launch full training on GPU node.
- [ ] Update `metrics.json` with actual results.
- [ ] Update `RUNS_INDEX.md`.
