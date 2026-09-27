"""
Evaluate CardiacPULSE on CAMEO (per-view checkpoints). Reports per-view and view-averaged ED / ES MAE (ms).

Usage:
    # Evaluate all views (auto-discovers per-view checkpoints):
    CUDA_VISIBLE_DEVICES=0 python evaluate_cameo.py \
        --results_dir /path/to/cameo_checkpoints --data_root /path/to/CAMEO

    # Evaluate a single view with an explicit checkpoint:
    CUDA_VISIBLE_DEVICES=0 python evaluate_cameo.py \
        --checkpoint /path/to/A4C/best_model.ckpt --view A4C \
        --data_root /path/to/CAMEO
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import CardiacPULSE
from models.decoder import PhaseDecoder
import pipeline as pl
from datasets.cameo import build_cameo_dataframe, split_cameo_df, CAMEOEntireVideo, CAMEO_VIEWS

import lightning as L
L.seed_everything(666)

# Path to CAMEO root; override with --data_root or the CAMEO_ROOT env var.
CAMEO_ROOT = os.environ.get('CAMEO_ROOT', '/path/to/CAMEO')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory with per-view subdirs containing checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Single checkpoint (requires --view)')
    parser.add_argument('--view', type=str, default=None,
                        help='Single view to evaluate')
    parser.add_argument('--data_root', type=str, default=os.environ.get('CAMEO_ROOT'),
                        help='CAMEO dataset root')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_videos', type=int, default=None)
    parser.add_argument('--decoder', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'decoder.pt'))
    parser.add_argument('--topk', type=int, default=1, choices=[1, 2])
    parser.add_argument('--reference_split', type=str, default='train', choices=['train', 'eval'],
                        help='unlabeled videos used to calibrate each view (train split, or the evaluated videos themselves)')
    return parser.parse_args()


# =========================================================================
# Evaluation
# =========================================================================

def run_videos(model, decoder, df_view, device, desc):
    """Per-video decoder scores for both signal orientations (no labels used)."""
    loader = DataLoader(CAMEOEntireVideo(CAMEO_ROOT, df_view, size=128, period=1),
                        batch_size=1, shuffle=False, num_workers=0)
    items = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            try:
                fps = batch['fps'].item()
                sig = pl.extract_signals(model, batch['video'].to(device), fps)
                ra, rb = decoder(pl.features(sig)), decoder(pl.features(sig, flip=True))
                za = np.stack([pl.zscore(r) for r in ra]); zb = np.stack([pl.zscore(r) for r in rb])
                items.append(dict(patient_id=batch['patient_id'][0], fps=fps,
                                  gt_ed=batch['ed_index'].item(), gt_es=batch['es_index'].item(),
                                  za=za, zb=zb, P=pl.refine_period(np.concatenate([za, zb]), fps, sig['P']),
                                  prefer_b=pl.confidence(rb, sig['P']) > pl.confidence(ra, sig['P'])))
            except Exception as e:
                print(f"\n  Error {desc}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return items


def evaluate_view(checkpoint_path, view, df_eval, df_ref, device, topk=1, max_videos=None):
    """Evaluate a single view. Returns per-video results as list of dicts."""
    model = CardiacPULSE.load_from_checkpoint(checkpoint_path, map_location=device).to(device).eval()
    decoder = PhaseDecoder(ARGS.decoder, device)
    df_view = df_eval[df_eval['view'] == view].reset_index(drop=True)
    if max_videos:
        df_view = df_view.head(max_videos)
    if len(df_view) == 0:
        print(f"  No videos for view {view}")
        return []
    items = run_videos(model, decoder, df_view, device, f"  {view}")
    ref = items if df_ref is None else run_videos(
        model, decoder, df_ref[df_ref['view'] == view].reset_index(drop=True), device, f"  {view} (ref)")
    swap = np.mean([it['prefer_b'] for it in ref]) > 0.5 if ref else False
    results = []
    for it in items:
        z, other = (it['zb'], it['za']) if swap else (it['za'], it['zb'])
        row = dict(patient_id=it['patient_id'], view=view, fps=it['fps'], gt_ed=it['gt_ed'], gt_es=it['gt_es'])
        for ch, name, gt in [(0, 'ed', it['gt_ed']), (1, 'es', it['gt_es'])]:
            pred = pl.decode(z[ch], it['P'], topk, alt=other[ch] if (swap and topk == 2) else None)
            row[f'{name}_mae_ms'] = pl.mae_ms(pred, gt, it['fps'])
            row[f'{name}_ae_frames'] = float(np.min(np.abs(np.asarray(pred) - gt))) if pred else float('inf')
        results.append(row)
    return results


def print_view_results(view, results):
    """Print summary for one view."""
    if not results:
        print(f"  {view}: no results")
        return

    ed_ms = [r['ed_mae_ms'] for r in results]
    es_ms = [r['es_mae_ms'] for r in results]
    ed_fr = [r['ed_ae_frames'] for r in results]
    es_fr = [r['es_ae_frames'] for r in results]

    print(f"  {view:18s}  n={len(results):3d}  "
          f"ED {np.mean(ed_ms):6.1f}ms ({np.mean(ed_fr):.2f}fr)  "
          f"ES {np.mean(es_ms):6.1f}ms ({np.mean(es_fr):.2f}fr)  "
          f"Avg {np.mean(ed_ms + es_ms) / 2:.1f}ms")


def main():
    global ARGS
    args = parse_args()
    ARGS = args
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    global CAMEO_ROOT
    if args.data_root:
        CAMEO_ROOT = args.data_root
    if not os.path.isdir(CAMEO_ROOT):
        raise SystemExit(
            f"CAMEO not found at '{CAMEO_ROOT}'. "
            f"Pass --data_root /path/to/CAMEO or set the CAMEO_ROOT env var.")

    # Build test set
    print("Building CAMEO dataframe...")
    df = build_cameo_dataframe(CAMEO_ROOT)
    df_train, df_val, df_test = split_cameo_df(df, seed=666)
    df_eval = {'train': df_train, 'val': df_val, 'test': df_test}[args.split]
    df_ref = df_train if args.reference_split == 'train' else None
    print(f"{args.split} set: {df_eval['patient_id'].nunique()} patients, {len(df_eval)} videos")

    # Determine which views and checkpoints to evaluate
    view_checkpoints = {}

    if args.checkpoint and args.view:
        # Single view mode
        view_checkpoints[args.view] = args.checkpoint
    elif args.results_dir:
        # Auto-discover per-view checkpoints: <results_dir>/<view>/best_model.ckpt
        for view in CAMEO_VIEWS:
            ckpt_path = os.path.join(args.results_dir, view, 'best_model.ckpt')
            if os.path.exists(ckpt_path):
                view_checkpoints[view] = ckpt_path
            else:
                print(f"  WARNING: No checkpoint for {view} at {ckpt_path}")
    else:
        print("ERROR: Provide either --results_dir or --checkpoint + --view")
        sys.exit(1)

    print(f"\nEvaluating {len(view_checkpoints)} views: {list(view_checkpoints.keys())}")
    print("=" * 90)

    all_results = []
    for view, ckpt in view_checkpoints.items():
        print(f"\n--- {view} ---")
        print(f"  Checkpoint: {ckpt}")
        view_results = evaluate_view(ckpt, view, df_eval, df_ref, device, args.topk, args.max_videos)
        all_results.extend(view_results)
        print_view_results(view, view_results)

    if not all_results:
        print("No results.")
        return

    # Overall summary
    print("\n" + "=" * 90)
    print(f"OVERALL RESULTS (topk={args.topk})")
    print("=" * 90)

    df_results = pd.DataFrame(all_results)

    # Per-view table
    print(f"\n{'View':18s} {'N':>4s}  {'ED MAE(ms)':>10s} {'ES MAE(ms)':>10s} "
          f"{'Avg MAE(ms)':>11s} {'ED AE(fr)':>9s} {'ES AE(fr)':>9s}")
    print("-" * 90)

    for view in CAMEO_VIEWS:
        vdf = df_results[df_results['view'] == view]
        if len(vdf) == 0:
            continue
        ed_ms = vdf['ed_mae_ms'].mean()
        es_ms = vdf['es_mae_ms'].mean()
        avg_ms = (ed_ms + es_ms) / 2
        ed_fr = vdf['ed_ae_frames'].mean()
        es_fr = vdf['es_ae_frames'].mean()
        print(f"{view:18s} {len(vdf):4d}  {ed_ms:10.1f} {es_ms:10.1f} "
              f"{avg_ms:11.1f} {ed_fr:9.2f} {es_fr:9.2f}")

    # Aggregate
    ed_ms_all = df_results['ed_mae_ms'].mean()
    es_ms_all = df_results['es_mae_ms'].mean()
    avg_ms_all = (ed_ms_all + es_ms_all) / 2
    ed_fr_all = df_results['ed_ae_frames'].mean()
    es_fr_all = df_results['es_ae_frames'].mean()
    print("-" * 90)
    print(f"{'OVERALL':18s} {len(df_results):4d}  {ed_ms_all:10.1f} {es_ms_all:10.1f} "
          f"{avg_ms_all:11.1f} {ed_fr_all:9.2f} {es_fr_all:9.2f}")
    vm = df_results.groupby('view')[['ed_mae_ms', 'es_mae_ms']].mean()
    print(f"{'VIEW-AVERAGED':18s}       {vm['ed_mae_ms'].mean():10.1f} {vm['es_mae_ms'].mean():10.1f} "
          f"{(vm['ed_mae_ms'].mean() + vm['es_mae_ms'].mean()) / 2:11.1f}")

    # Accuracy at thresholds
    print(f"\nAccuracy at thresholds (all views):")
    for thr in [1, 2, 3, 5]:
        ed_acc = (df_results['ed_ae_frames'] <= thr).mean() * 100
        es_acc = (df_results['es_ae_frames'] <= thr).mean() * 100
        print(f"  <= {thr} frames: ED {ed_acc:.1f}%  ES {es_acc:.1f}%")

    # Save results CSV
    if args.results_dir:
        csv_path = os.path.join(args.results_dir, f'cameo_{args.split}_results.csv')
        df_results.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")


if __name__ == '__main__':
    main()
