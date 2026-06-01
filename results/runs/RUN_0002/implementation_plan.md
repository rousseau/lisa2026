# RUN_0002 — Implementation plan

## Architecture

### Generator3D (G_AB, G_BA)
- U-Net 3D avec 4 niveaux d'échantillonnage (down: 96→48→24→12)
- Filtres: 32 → 64 → 128 → 256
- Bottleneck: 6 ResBlocks identitaires
- Skip connections (concaténation + conv)
- Output: Tanh (range [-1, 1])

### Discriminator3D (D_A, D_B)
- PatchGAN 3D (4 couches conv + LeakyReLU)
- Base filters: 64 (×2 à chaque niveau → 64/128/256/512)
- Output: Patch de taille ~14×14×14 pour input 96³
- LSGAN (MSE loss)

### Optimisation
- Adam: lr=2e-4, β1=0.5, β2=0.999 (séparé G/D)
- LR: 50 epochs constantes, puis décroissance linéaire à 0
- Image buffer: 50 images (discriminator stabilisation)

## Train/val split
- Domain A: 27 sujets (train), 5 sujets (val)
- Domain B: 27 sujets (train), 8 sujets (val)
- Total: 54 train, 13 val sujets

## Losses
```
Adv loss (LSGAN): 0.5 * (D(real) - 1)² + 0.5 * (D(fake))²
Cycle loss (L1): λ_cyc * |x - G_BA(G_AB(x))|₁
Identity loss (L1): λ_id * |x - G_BA(x)|₁
Total G: Adv + Cycle + Identity
Total D: Mean of D_A and D_B losses
```

## Execution plan
1. Pull data from Jean Zay
2. Review plots: results/runs/RUN_0002/plots/
3. Evaluate: `python evaluate.py --run 0002`
4. Submit to challenge using G_AB_best.pt

## Outputs
- Checkpoints: `outputs/checkpoints/RUN_0002/`
  - G_AB_best.pt
  - cyclegan_full_best.pt
- Metrics: `results/runs/RUN_0002/metrics.json`
- Plots: `results/runs/RUN_0002/plots/`
