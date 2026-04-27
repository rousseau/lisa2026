#!/usr/bin/env python3
"""
Statistiques complètes du jeu de données LISA 2026.

Usage :
    python src/dataset_stats.py [--data-root /path/to/LISA2026]
"""

import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

DATA_ROOT_DEFAULT = "/home/rousseau/Data/LISA2026"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "stats"

# Regex pour les fichiers LF et seg
FILE_RE = re.compile(
    r"^(LISA_(?:VALIDATION_)?(\d+))_LF_(axi|cor|sag)\.nii\.gz$"
)
SEG_LF_RE = re.compile(r"^(LISA_\d+)_LF_seg\.nii\.gz$")
SEG_HF_RE = re.compile(r"^(LISA_\d+)_seg\.nii\.gz$")
CISO_RE   = re.compile(r"^(LISA_\d+)_ciso\.nii\.gz$")

ORIENTATIONS = ["axi", "cor", "sag"]

# Structures de segmentation Task 2 (labels non nuls attendus)
SEG_LABELS = {
    1: "hippocampe_G", 2: "hippocampe_D",
    3: "caude_G",      4: "caude_D",
    5: "putamen_G",    6: "putamen_D",
    7: "globus_G",     8: "globus_D",
    9: "thalamus_G",  10: "thalamus_D",
    11: "corps_calleux",
    12: "ventricule_G", 13: "ventricule_D",
}


# ─────────────────────────────────────────────────────────────────────────────
# Collecte des fichiers
# ─────────────────────────────────────────────────────────────────────────────

def collect_files(data_root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(data_root.iterdir()):
        m = FILE_RE.match(f.name)
        if not m:
            continue
        subject = m.group(1)          # ex. "LISA_0001" ou "LISA_VALIDATION_0001"
        sid     = m.group(2)          # numéro brut
        orient  = m.group(3)
        is_val  = "VALIDATION" in subject

        # détermination du split à partir de l'ID numérique
        sid_int = int(sid)
        if is_val:
            split = "validation"
        elif 1 <= sid_int <= 999:
            split = "train_seg"        # 0001–0096 : avec segmentation
        elif 1000 <= sid_int <= 1999:
            split = "train_seg_hf"     # 1001–1048 : avec seg HF
        elif 2000 <= sid_int <= 2999:
            split = "task1a_qa"        # 2001–2100 : QA seulement, pas de seg
        else:
            split = "unknown"

        has_lf_seg  = (data_root / f"{subject}_LF_seg.nii.gz").exists()
        has_hf_seg  = (data_root / f"{subject}_seg.nii.gz").exists()
        has_ciso    = (data_root / f"{subject}_ciso.nii.gz").exists()

        rows.append({
            "subject": subject,
            "sid_int": sid_int,
            "split": split,
            "orientation": orient,
            "filepath": str(f),
            "has_lf_seg": has_lf_seg,
            "has_hf_seg": has_hf_seg,
            "has_ciso": has_ciso,
            "has_ciso_with_seg": has_ciso and has_lf_seg,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sep(char="─", n=72):  print(char * n)
def section(t):   print(); sep("═"); print(f"  {t}"); sep("═")
def subsection(t): print(); sep(); print(f"  {t}"); sep()


def nifti_info(path: str) -> dict:
    img = nib.load(path)
    hdr = img.header
    shape = tuple(int(x) for x in img.shape[:3])
    zooms = tuple(float(round(z, 4)) for z in hdr.get_zooms()[:3])
    return {"shape": shape, "zooms": zooms}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--no-nifti", action="store_true",
                        help="Sauter la lecture des headers NIfTI (plus rapide)")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scan de : {data_root}")
    df = collect_files(data_root)
    print(f"  → {len(df)} fichiers LF trouvés\n")

    # ── 1. Volumes par split × orientation ───────────────────────────────────
    section("1. Volumes par split × orientation")
    vol_table = (
        df.groupby(["split", "orientation"])
        .size().unstack("orientation", fill_value=0)
        .reindex(columns=ORIENTATIONS, fill_value=0)
    )
    vol_table["TOTAL"] = vol_table.sum(axis=1)
    print(vol_table.to_string())
    vol_table.to_csv(RESULTS_DIR / "volumes_per_split_orientation.csv")

    # ── 2. Sujets uniques par split ───────────────────────────────────────────
    section("2. Sujets uniques par split")
    rows_split = []
    for split, grp in df.groupby("split"):
        n_sub            = grp["subject"].nunique()
        n_lf_seg         = grp[grp["has_lf_seg"]]["subject"].nunique()
        n_hf_seg         = grp[grp["has_hf_seg"]]["subject"].nunique()
        n_ciso           = grp[grp["has_ciso"]]["subject"].nunique()
        n_ciso_with_seg  = grp[grp["has_ciso_with_seg"]]["subject"].nunique()
        print(f"  [{split:15s}]  sujets: {n_sub:4d}  "
              f"LF_seg: {n_lf_seg:4d}  HF_seg: {n_hf_seg:4d}  "
              f"ciso: {n_ciso:4d}  ciso+seg: {n_ciso_with_seg:4d}")
        rows_split.append({"split": split, "n_subjects": n_sub,
                           "n_lf_seg": n_lf_seg, "n_hf_seg": n_hf_seg,
                           "n_ciso": n_ciso, "n_ciso_with_seg": n_ciso_with_seg})
    pd.DataFrame(rows_split).to_csv(RESULTS_DIR / "subjects_per_split.csv", index=False)

    # ── 3. Headers NIfTI (shapes + voxel sizes) ───────────────────────────────
    if not args.no_nifti:
        section("3. Dimensions et tailles de voxels par orientation (Training, 5 premiers sujets)")
        train_df = df[df["split"].isin(["train_seg", "train_seg_hf"])]
        rows_hdr = []
        for orient in ORIENTATIONS:
            sub_orient = train_df[train_df["orientation"] == orient].head(5)
            subsection(f"Orientation LF : {orient}")
            shapes, zooms_list = [], []
            for _, row in sub_orient.iterrows():
                info = nifti_info(row["filepath"])
                shapes.append(info["shape"])
                zooms_list.append(info["zooms"])
                print(f"    {row['subject']:25s}  shape={info['shape']}  zooms={info['zooms']}")
                rows_hdr.append({"type": f"LF_{orient}", "subject": row["subject"],
                                 "shape": str(info["shape"]), "zooms": str(info["zooms"])})
            shapes_arr = np.array(shapes)
            zooms_arr  = np.array(zooms_list)
            if len(shapes_arr):
                print(f"  Shape min/max : {tuple(shapes_arr.min(0))} / {tuple(shapes_arr.max(0))}")
                print(f"  Zooms min/max : {tuple(zooms_arr.min(0).round(3))} / "
                      f"{tuple(zooms_arr.max(0).round(3))}")

        # ── Images ciso (isotropiques HF) ────────────────────────────────────
        subsection("Images ciso (isotropiques HF) — tous les sujets")
        ciso_subjects = df[df["has_ciso"] & (df["orientation"] == "axi")]["subject"].unique()
        shapes_c, zooms_c = [], []
        for subj in sorted(ciso_subjects)[:5]:
            ciso_path = str(data_root / f"{subj}_ciso.nii.gz")
            info = nifti_info(ciso_path)
            shapes_c.append(info["shape"])
            zooms_c.append(info["zooms"])
            has_seg = (data_root / f"{subj}_seg.nii.gz").exists()
            print(f"    {subj:25s}  shape={info['shape']}  zooms={info['zooms']}  seg={'✓' if has_seg else '✗'}")
            rows_hdr.append({"type": "ciso", "subject": subj,
                             "shape": str(info["shape"]), "zooms": str(info["zooms"])})
        if len(shapes_c):
            sa = np.array(shapes_c); za = np.array(zooms_c)
            print(f"  Shape min/max : {tuple(sa.min(0))} / {tuple(sa.max(0))}")
            print(f"  Zooms min/max : {tuple(za.min(0).round(3))} / {tuple(za.max(0).round(3))}")
        n_ciso_total = len(ciso_subjects)
        n_ciso_seg   = sum(1 for s in ciso_subjects
                          if (data_root / f"{s}_seg.nii.gz").exists())
        print(f"  Total ciso : {n_ciso_total}  dont avec seg : {n_ciso_seg}")
        pd.DataFrame(rows_hdr).to_csv(RESULTS_DIR / "nifti_headers_sample.csv", index=False)

    # ── 4. Analyse du CSV Task 1a ─────────────────────────────────────────────
    section("4. Task 1a — distribution des artefacts")
    csv_1a = data_root / "LISA_Task1a_2026.csv"
    if csv_1a.exists():
        df_1a = pd.read_csv(csv_1a)
        artifact_cols = [c for c in df_1a.columns if c != "filename"]
        print(f"  Fichiers labellisés : {len(df_1a)}")
        print(f"  Colonnes artefacts  : {artifact_cols}\n")
        counts = df_1a[artifact_cols].sum().sort_values(ascending=False)
        for art, cnt in counts.items():
            pct = 100 * cnt / len(df_1a)
            print(f"    {art:15s} : {cnt:4d}  ({pct:.1f}%)")
        # images sans aucun artefact
        n_clean = (df_1a[artifact_cols].sum(axis=1) == 0).sum()
        print(f"\n  Images sans artefact (toutes colonnes = 0) : {n_clean} "
              f"({100*n_clean/len(df_1a):.1f}%)")
        # co-occurrences
        subsection("Co-occurrences (nombre d'artefacts par image)")
        co = df_1a[artifact_cols].sum(axis=1).value_counts().sort_index()
        for n_art, cnt in co.items():
            print(f"    {n_art} artefact(s) : {cnt:4d} images")
        df_1a.to_csv(RESULTS_DIR / "task1a_summary.csv", index=False)
    else:
        print(f"  ⚠  Fichier non trouvé : {csv_1a}")

    # ── 5. Analyse des CSV Task 1b ────────────────────────────────────────────
    section("5. Task 1b — fichiers de ratings bruit/mouvement")
    csv_1b_files = {
        "NoNoise_NoMotion":   "Task_1b_NoNoise_NoMotion.csv",
        "WithNoise_NoMotion": "Task_1b_WithNoise_NoMotion.csv",
        "NoNoise_WithMotion": "Task_1b_NoNoise_WithMotion.csv",
        "WithNoise_WithMotion":"Task_1b_WithNoise_WithMotion.csv",
    }
    for label, fname in csv_1b_files.items():
        p = data_root / fname
        if p.exists():
            d = pd.read_csv(p)
            print(f"\n  [{label}]  → {len(d)} lignes")
            art_cols = [c for c in d.columns if c != "filename"]
            pos = d[art_cols].sum()
            for col, cnt in pos.items():
                print(f"    {col:15s} : {cnt} positifs / {len(d)}")
        else:
            print(f"  ⚠  {fname} non trouvé")

    # ── 6. Statistiques de segmentation (Task 2) ─────────────────────────────
    if not args.no_nifti:
        section("6. Segmentation (Task 2) — volumes par structure (5 premiers sujets)")
        rows_seg = []

        def analyse_seg(seg_path: Path, subj: str, seg_type: str):
            img = nib.load(str(seg_path))
            seg_data = np.asarray(img.dataobj, dtype=np.int16)
            zooms = img.header.get_zooms()[:3]
            vox_vol_mm3 = float(np.prod(zooms))
            unique_labels = np.unique(seg_data[seg_data > 0])
            print(f"\n  [{seg_type}] {subj}  labels: {list(unique_labels)}")
            for lbl, name in SEG_LABELS.items():
                vol_vox = int((seg_data == lbl).sum())
                vol_mm3 = round(vol_vox * vox_vol_mm3, 1)
                if vol_vox > 0:
                    print(f"    [{lbl:2d}] {name:20s} : {vol_vox:6d} vox  ({vol_mm3:.1f} mm³)")
                rows_seg.append({"subject": subj, "seg_type": seg_type,
                                 "label": lbl, "name": name,
                                 "vol_vox": vol_vox, "vol_mm3": vol_mm3})

        subsection("Segmentations LF (_LF_seg)")
        lf_seg_subjects = df[df["has_lf_seg"] & (df["orientation"] == "axi")].head(5)
        for _, row in lf_seg_subjects.iterrows():
            analyse_seg(data_root / f"{row['subject']}_LF_seg.nii.gz",
                        row["subject"], "LF_seg")

        subsection("Segmentations LF (_LF_seg) des sujets ciso")
        ciso_seg_subjects = (
            df[df["has_ciso_with_seg"] & (df["orientation"] == "axi")]
            .head(5)
        )
        for _, row in ciso_seg_subjects.iterrows():
            analyse_seg(data_root / f"{row['subject']}_LF_seg.nii.gz",
                        row["subject"], "LF_seg_ciso")

        pd.DataFrame(rows_seg).to_csv(RESULTS_DIR / "seg_volumes_sample.csv", index=False)

    # ── 7. Statistiques d'intensité (5 premiers sujets training) ─────────────
    if not args.no_nifti:
        section("7. Statistiques d'intensité par orientation (5 premiers sujets)")
        train_df = df[df["split"] == "train_seg"]
        rows_int = []

        def intensity_stats(path: str, subj: str, img_type: str) -> dict:
            img = nib.load(path)
            data = np.asarray(img.dataobj, dtype=np.float32)
            return {
                "subject": subj, "type": img_type,
                "mean": float(np.mean(data)), "std": float(np.std(data)),
                "min": float(np.min(data)), "max": float(np.max(data)),
                "p5":  float(np.percentile(data, 5)),
                "p95": float(np.percentile(data, 95)),
            }

        for orient in ORIENTATIONS:
            sub = train_df[train_df["orientation"] == orient].head(5)
            print(f"\n  LF {orient.upper()}")
            for _, row in sub.iterrows():
                s = intensity_stats(row["filepath"], row["subject"], f"LF_{orient}")
                print(f"    {row['subject']:25s}  "
                      f"mean={s['mean']:.1f}  std={s['std']:.1f}  "
                      f"min={s['min']:.0f}  max={s['max']:.0f}  "
                      f"[p5={s['p5']:.1f}, p95={s['p95']:.1f}]")
                rows_int.append(s)

        subsection("Images ciso (isotropiques HF) — 5 premiers sujets")
        ciso_subjects = sorted(
            df[df["has_ciso"] & (df["orientation"] == "axi")]["subject"].unique()
        )[:5]
        for subj in ciso_subjects:
            ciso_path = str(data_root / f"{subj}_ciso.nii.gz")
            s = intensity_stats(ciso_path, subj, "ciso")
            print(f"    {subj:25s}  "
                  f"mean={s['mean']:.1f}  std={s['std']:.1f}  "
                  f"min={s['min']:.0f}  max={s['max']:.0f}  "
                  f"[p5={s['p5']:.1f}, p95={s['p95']:.1f}]")
            rows_int.append(s)

        pd.DataFrame(rows_int).to_csv(RESULTS_DIR / "intensity_stats_sample.csv", index=False)

    print()
    sep("═")
    print(f"  CSVs exportés dans : {RESULTS_DIR}")
    sep("═")


if __name__ == "__main__":
    main()
