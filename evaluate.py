"""
Evaluate CardiacPULSE on EchoNet-Dynamic. Reports ED / ES / average MAE (ms).

Usage:
    CUDA_VISIBLE_DEVICES=0 python evaluate.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --data_root /path/to/EchoNet-Dynamic --split test
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
from datasets.echonet import EntireEcho as EchoEntireVideo

import lightning as L
L.seed_everything(666)

# Path to EchoNet-Dynamic root; override with --data_root or the ECHONET_ROOT env var.
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '/path/to/EchoNet-Dynamic')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default=os.environ.get('ECHONET_ROOT'),
                        help='EchoNet-Dynamic root (contains Videos/, FileList.csv, VolumeTracings.csv)')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_videos', type=int, default=None)
    parser.add_argument('--decoder', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'decoder.pt'))
    parser.add_argument('--topk', type=int, default=1, choices=[1, 2])
    return parser.parse_args()


def load_echonet_gt(split='TEST'):
    df_files = pd.read_csv(os.path.join(ECHONET_ROOT, 'FileList.csv'))
    df_files = df_files[df_files['Split'] == split].reset_index(drop=True)
    df_tracings = pd.read_csv(os.path.join(ECHONET_ROOT, 'VolumeTracings.csv'))
    ed_es = {}
    for fn, grp in df_tracings.groupby('FileName'):
        frames = sorted(grp['Frame'].unique())
        if len(frames) >= 2:
            ed_es[fn] = {'ED': int(frames[0]), 'ES': int(frames[1])}
    df_files['FileName_avi'] = df_files['FileName'].apply(
        lambda x: x if x.endswith('.avi') else x + '.avi')
    df_files['ED'] = df_files['FileName_avi'].map(lambda x: ed_es.get(x, {}).get('ED'))
    df_files['ES'] = df_files['FileName_avi'].map(lambda x: ed_es.get(x, {}).get('ES'))
    df_files = df_files.dropna(subset=['ED', 'ES']).reset_index(drop=True)
    df_files['ED'] = df_files['ED'].astype(int)
    df_files['ES'] = df_files['ES'].astype(int)
    return df_files


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    global ECHONET_ROOT
    if args.data_root:
        ECHONET_ROOT = args.data_root
    if not os.path.isdir(ECHONET_ROOT):
        raise SystemExit(
            f"EchoNet-Dynamic not found at '{ECHONET_ROOT}'. "
            f"Pass --data_root /path/to/EchoNet-Dynamic or set the ECHONET_ROOT env var.")

    print("Loading CardiacPULSE model...")
    model = CardiacPULSE.load_from_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device).eval()
    decoder = PhaseDecoder(args.decoder, device)
    print(f"Loaded: {args.checkpoint}")

    split_map = {'train': 'TRAIN', 'val': 'VAL', 'test': 'TEST'}
    df = load_echonet_gt(split_map[args.split])
    if args.max_videos:
        df = df.head(args.max_videos)

    dataset = EchoEntireVideo(
        os.path.join(ECHONET_ROOT, 'Videos'), df, size=128, period=1)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Evaluating {len(df)} videos")
    print("=" * 70)

    ed_list, es_list = [], []

    with torch.no_grad():
        for idx, video_p1 in enumerate(tqdm(loader, desc="Evaluating")):
            try:
                row = df.iloc[idx]
                fps_native = row['FPS']
                gt_ed = float(int(row['ED']))
                gt_es = float(int(row['ES']))

                sig = pl.extract_signals(model, video_p1.to(device), fps_native)
                z = decoder(pl.features(sig))
                ed_list.append(pl.mae_ms(pl.decode(pl.zscore(z[0]), sig['P'], args.topk), gt_ed, fps_native))
                es_list.append(pl.mae_ms(pl.decode(pl.zscore(z[1]), sig['P'], args.topk), gt_es, fps_native))

            except Exception as e:
                print(f"\nError {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # ===== Results =====
    ed = np.array(ed_list); es = np.array(es_list)
    valid = np.isfinite(ed) & np.isfinite(es)
    ed, es = ed[valid], es[valid]
    print("\n" + "=" * 60)
    print(f"CardiacPULSE on EchoNet-Dynamic ({args.split}, topk={args.topk}), {len(ed)} videos")
    print("=" * 60)
    print(f"  ED MAE:  {ed.mean():.1f} ms")
    print(f"  ES MAE:  {es.mean():.1f} ms")
    print(f"  Average: {(ed.mean() + es.mean()) / 2:.1f} ms")
    for t in [20, 40, 50, 100]:
        print(f"  <= {t:>3d} ms: ED {(ed <= t).mean() * 100:5.1f}%  ES {(es <= t).mean() * 100:5.1f}%")


if __name__ == '__main__':
    main()
