#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic single-GPU Federated Averaging (FedAvg) for nnU-Net v2.

This public/shareable script trains multiple nnU-Net v2 client datasets
sequentially on one GPU, aggregates their checkpoints with case-count-weighted
FedAvg, and exports each global checkpoint into an nnU-Net-compatible results
tree for inference with ``nnUNetv2_predict``.

Before running, prepare:
  1. nnU-Net v2 environment variables:
       nnUNet_raw, nnUNet_preprocessed, nnUNet_results
  2. Client datasets in nnU-Net v2 format:
       <NNUNET_RAW>/DatasetXXX_<NAME>/imagesTr
       <NNUNET_RAW>/DatasetXXX_<NAME>/labelsTr
  3. One initial global checkpoint:
       <PATH_TO_INITIAL_CHECKPOINT.pth>
  4. A common model setup across all clients:
       same configuration, trainer, plans, fold, modalities, and label schema.

Example command with placeholders:
  python nnunetv2_fedavg_case_weighted.py \
    --datasets <CLIENT_DATASET_ID_1> <CLIENT_DATASET_ID_2> \
    --anchor_dataset <DATASET_ID_USED_FOR_EXPORT_AND_PREDICTION> \
    --rounds <NUMBER_OF_FL_ROUNDS> \
    --init_global <PATH_TO_INITIAL_CHECKPOINT.pth> \
    --fl_root <PATH_TO_FL_OUTPUT_DIRECTORY> \
    --raw_data_root <PATH_TO_NNUNET_RAW> \
    --config <NNUNET_CONFIG> \
    --trainer <TRAINER_NAME> \
    --plans <PLANS_NAME> \
    --fold <FOLD_ID>

Notes:
  - This script does not include any study-specific paths, dataset names, or
    institution-specific parameters.
  - Use ``--dry_run`` first to check commands without launching training.
  - All client checkpoints must have compatible network weights.
"""

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import torch


# --------------------------
# Helpers: filesystem & nnU-Net conventions
# --------------------------

def run_cmd(cmd: List[str], env: Dict[str, str], dry_run: bool = False) -> None:
    print("\n[CMD]", " ".join(cmd))
    if dry_run:
        print("[DRY RUN] skipped")
        return
    subprocess.run(cmd, env=env, check=True)


def find_dataset_folder(raw_data_root: Path, dataset_id: int) -> str:
    """Find folder name like Dataset002_Something under raw_data_root."""
    pat = f"Dataset{dataset_id:03d}_*"
    matches = sorted([p.name for p in raw_data_root.glob(pat) if p.is_dir()])
    if not matches:
        raise FileNotFoundError(f"Cannot find {pat} under {raw_data_root}")
    if len(matches) > 1:
        print(f"[WARN] Multiple matches for dataset {dataset_id}: {matches}. Using the first one.")
    return matches[0]


def count_cases_in_imagesTr(raw_data_root: Path, dataset_folder: str) -> int:
    """
    Count unique case IDs in imagesTr.
    Supports nnU-Net naming: <case>_0000.nii.gz, <case>_0001.nii.gz, ...
    """
    imagesTr = raw_data_root / dataset_folder / "imagesTr"
    if not imagesTr.exists():
        raise FileNotFoundError(f"imagesTr not found: {imagesTr}")

    # case id = prefix before _0000 / _0001 ...
    # Examples: Case_001_0000.nii.gz -> Case_001
    rx = re.compile(r"^(?P<case>.+)_(?P<mod>\d{4})\.nii(\.gz)?$")
    cases = set()
    for f in imagesTr.iterdir():
        if not f.is_file():
            continue
        m = rx.match(f.name)
        if m:
            cases.add(m.group("case"))
    if not cases:
        # fallback: count all nii.gz / nii
        all_nii = [x for x in imagesTr.iterdir() if x.is_file() and (x.name.endswith(".nii.gz") or x.name.endswith(".nii"))]
        if not all_nii:
            raise RuntimeError(f"No nii(.gz) files found in {imagesTr}")
        print("[WARN] Could not parse case ids by nnU-Net pattern; fallback to counting files. This may be wrong for multi-modality.")
        return len(all_nii)
    return len(cases)


def newest_checkpoint_final(search_root: Path) -> Path:
    """Pick the newest checkpoint_final.pth under search_root."""
    cands = list(search_root.rglob("checkpoint_final.pth"))
    if not cands:
        raise FileNotFoundError(f"checkpoint_final.pth not found under {search_root}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def nnunet_model_dir(nnUNet_results_root: Path,
                    dataset_folder: str,
                    trainer: str,
                    plans: str,
                    config: str,
                    fold: str) -> Path:
    """
    Build nnU-Net v2-like model dir:
      nnUNet_results/DatasetXXX_NAME/{trainer}__{plans}__{config}/{all or fold_0 ...}
    """
    base = nnUNet_results_root / dataset_folder / f"{trainer}__{plans}__{config}"
    if fold == "all":
        return base / "all"
    # numeric fold
    return base / f"fold_{fold}"


# --------------------------
# FedAvg
# --------------------------

def extract_weights(ckpt: dict) -> Dict[str, torch.Tensor]:
    if "network_weights" in ckpt:
        return ckpt["network_weights"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    raise KeyError(f"Cannot find weight key in checkpoint. Keys={list(ckpt.keys())}")


def fedavg_state_dict(state_dicts: List[Dict[str, torch.Tensor]],
                      weights: List[float]) -> Dict[str, torch.Tensor]:
    if len(state_dicts) == 0:
        raise ValueError("No state_dicts provided")
    if len(state_dicts) != len(weights):
        raise ValueError("state_dicts and weights length mismatch")

    # normalize
    w = torch.tensor(weights, dtype=torch.float64)
    w = w / w.sum()

    keys0 = list(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if list(sd.keys()) != keys0:
            raise RuntimeError(f"State dict keys mismatch between client0 and client{i}")

    out: Dict[str, torch.Tensor] = {}
    for k in keys0:
        vals = [sd[k] for sd in state_dicts]
        v0 = vals[0]
        if torch.is_floating_point(v0):
            stacked = torch.stack([v.to(torch.float64) for v in vals], dim=0)
            flat = stacked.view(len(vals), -1)
            avg_flat = (w[:, None] * flat).sum(dim=0)
            out[k] = avg_flat.view_as(v0).to(v0.dtype)
        else:
            out[k] = v0
    return out


def make_global_ckpt(base_ckpt: dict,
                     avg_weights: Dict[str, torch.Tensor]) -> dict:
    out = dict(base_ckpt)
    # overwrite weights
    if "network_weights" in out:
        out["network_weights"] = avg_weights
    elif "state_dict" in out:
        out["state_dict"] = avg_weights
    else:
        out["network_weights"] = avg_weights

    # drop optimizer states to avoid confusion
    for drop_key in ["optimizer_state", "grad_scaler_state"]:
        if drop_key in out:
            out[drop_key] = None
    return out


# --------------------------
# Core routine: rounds
# --------------------------

def train_one_client(dataset_id: int,
                     config: str,
                     fold: str,
                     trainer: str,
                     plans: str,
                     global_init_ckpt: Path,
                     client_results_root: Path,
                     train_extra: List[str],
                     env_base: Dict[str, str],
                     dry_run: bool) -> Path:
    """
    Train nnUNetv2_train with -pretrained_weights global_init_ckpt, storing outputs under client_results_root via nnUNet_results.
    Return path to newest checkpoint_final.pth.
    """
    client_results_root.mkdir(parents=True, exist_ok=True)

    env = dict(env_base)
    env["nnUNet_results"] = str(client_results_root)

    cmd = ["nnUNetv2_train", str(dataset_id), config, fold,
           "-tr", trainer,
           "-p", plans,
           "-pretrained_weights", str(global_init_ckpt)]
    cmd.extend(train_extra)

    run_cmd(cmd, env=env, dry_run=dry_run)

    # find checkpoint
    if dry_run:
        # fake path (won't be used)
        return client_results_root / "DUMMY_CHECKPOINT_FINAL.pth"

    ckpt = newest_checkpoint_final(client_results_root)
    return ckpt


def export_global_to_nnunet_results(global_ckpt: Path,
                                   export_root: Path,
                                   anchor_dataset_folder: str,
                                   trainer: str,
                                   plans: str,
                                   config: str,
                                   fold: str,
                                   checkpoint_name: str = "checkpoint_final.pth") -> Path:
    """
    Create a nnU-Net-like results tree and place global_ckpt as checkpoint_final.pth.
    Returns the model dir.
    """
    model_dir = nnunet_model_dir(export_root, anchor_dataset_folder, trainer, plans, config, fold)
    model_dir.mkdir(parents=True, exist_ok=True)

    dst = model_dir / checkpoint_name
    shutil.copy2(global_ckpt, dst)
    return model_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", type=int, required=True, help="Client dataset IDs, e.g. <ID1> <ID2> <ID3>")
    ap.add_argument("--anchor_dataset", type=int, required=True, help="Which dataset ID to use as nnUNetv2_predict -d (also for exported results tree)")
    ap.add_argument("--rounds", type=int, required=True, help="Number of federated learning rounds to run, starting from round 1")
    ap.add_argument("--init_global", type=str, required=True, help="Initial global checkpoint for round 0 (.pth)")
    ap.add_argument("--fl_root", type=str, required=True, help="Output directory for FL artifacts")
    ap.add_argument("--raw_data_root", type=str, required=True, help="Path to nnU-Net raw data root containing DatasetXXX_* folders")

    ap.add_argument("--config", type=str, required=True, help="nnU-Net configuration, e.g. 2d, 3d_fullres")
    ap.add_argument("--trainer", type=str, required=True, help="Trainer class name used for all clients")
    ap.add_argument("--plans", type=str, default="nnUNetPlans", help="Plans name, default nnUNetPlans")
    ap.add_argument("--fold", type=str, default="all", help="Fold: all or 0/1/2/3/4")

    ap.add_argument("--train_extra", nargs="*", default=[], help="Additional arguments passed directly to nnUNetv2_train")
    ap.add_argument("--export_checkpoint_name", type=str, default="checkpoint_final.pth",
                    help="Name used under exported nnU-Net results directory (default checkpoint_final.pth)")

    ap.add_argument("--dry_run", action="store_true", help="Print commands but do not run training")
    args = ap.parse_args()

    fl_root = Path(args.fl_root).resolve()
    raw_root = Path(args.raw_data_root).resolve()
    fl_root.mkdir(parents=True, exist_ok=True)

    globals_dir = fl_root / "globals"
    locals_dir = fl_root / "locals"
    rounds_dir = fl_root / "rounds"
    globals_dir.mkdir(exist_ok=True)
    locals_dir.mkdir(exist_ok=True)
    rounds_dir.mkdir(exist_ok=True)

    # detect dataset folder name for anchor (Dataset002_xxx)
    anchor_folder = find_dataset_folder(raw_root, args.anchor_dataset)

    # base env (inherit current env, but we'll override nnUNet_results per client)
    env_base = dict(os.environ)

    # init global -> round0
    global_prev = Path(args.init_global).resolve()
    if not global_prev.exists():
        raise FileNotFoundError(f"init_global not found: {global_prev}")
    # also copy a stable snapshot into globals/
    global0 = globals_dir / "global_round0.pth"
    if not global0.exists():
        shutil.copy2(global_prev, global0)
    global_prev = global0

    # auto weights by counting cases in each client's imagesTr
    client_case_counts: Dict[int, int] = {}
    for did in args.datasets:
        folder = find_dataset_folder(raw_root, did)
        n_cases = count_cases_in_imagesTr(raw_root, folder)
        client_case_counts[did] = n_cases
    print("\n[INFO] Client weights (num cases):", client_case_counts)

    for r in range(1, args.rounds + 1):
        print("\n" + "=" * 72)
        print(f"[ROUND {r}] global init = {global_prev}")
        print("=" * 72)

        local_ckpts: List[Tuple[int, Path]] = []

        # ---- train each client sequentially ----
        for did in args.datasets:
            client_out_root = rounds_dir / f"round{r}" / f"client{did}" / "nnUNet_results"
            ckpt_final = train_one_client(
                dataset_id=did,
                config=args.config,
                fold=args.fold,
                trainer=args.trainer,
                plans=args.plans,
                global_init_ckpt=global_prev,
                client_results_root=client_out_root,
                train_extra=args.train_extra,
                env_base=env_base,
                dry_run=args.dry_run
            )

            # snapshot local checkpoint
            snap = locals_dir / f"round{r}_client{did}.pth"
            if not args.dry_run:
                shutil.copy2(ckpt_final, snap)
            local_ckpts.append((did, snap))
            print(f"[ROUND {r}] client{did} checkpoint -> {snap}")

        if args.dry_run:
            print("[DRY RUN] skip aggregation/export")
            break

        # ---- aggregate (FedAvg) ----
        ckpt_objs = []
        sds = []
        ws = []
        for did, p in local_ckpts:
            ckpt = torch.load(p, map_location="cpu")
            ckpt_objs.append(ckpt)
            sds.append(extract_weights(ckpt))
            ws.append(float(client_case_counts[did]))

        avg_sd = fedavg_state_dict(sds, ws)
        global_ckpt_obj = make_global_ckpt(ckpt_objs[0], avg_sd)

        global_path = globals_dir / f"global_round{r}.pth"
        torch.save(global_ckpt_obj, global_path)
        print(f"[ROUND {r}] global saved -> {global_path}")

        # ---- export to nnU-Net-like results tree ----
        export_root = fl_root / f"nnUNet_results_global_round{r}"
        model_dir = export_global_to_nnunet_results(
            global_ckpt=global_path,
            export_root=export_root,
            anchor_dataset_folder=anchor_folder,
            trainer=args.trainer,
            plans=args.plans,
            config=args.config,
            fold=args.fold,
            checkpoint_name=args.export_checkpoint_name
        )
        print(f"[ROUND {r}] exported nnU-Net model dir -> {model_dir}")
        print(f"[ROUND {r}] predict usage example:")
        print(f"  nnUNet_results={export_root} nnUNetv2_predict -d {args.anchor_dataset} -c {args.config} -tr {args.trainer} -p {args.plans} -f {args.fold} -chk {args.export_checkpoint_name} -i <imagesTs> -o <out_pred> --disable_tta")

        # next round
        global_prev = global_path

    print("\n[DONE]")

if __name__ == "__main__":
    main()
