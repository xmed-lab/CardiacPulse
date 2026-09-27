"""
CardiacPULSE Training: Fourier Cardiac Phase Network

Per-pixel temporal FFT + learned spatial attention mask.
Self-supervised via spectral + phase coherence priors.

Usage:
    CUDA_VISIBLE_DEVICES=0 python train.py --config configs/echonet.yaml

    # Dry run:
    python train.py --config configs/echonet.yaml --dry_run

    # Smoke test (2 epochs):
    CUDA_VISIBLE_DEVICES=0 python train.py --config configs/echonet.yaml --max_epochs 2
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import pandas as pd

from models.model import CardiacPULSE
from datasets.echonet import EchoDynamicDatasetLazy


def get_dataloader(config, split='train'):
    data_cfg = config['data']
    data_dir = data_cfg['echonet_root']
    batch_size = data_cfg['batch_size']
    num_workers = data_cfg['num_workers']
    num_frames = data_cfg['num_frames']
    period = data_cfg['period']

    split_file = os.path.join(data_dir, 'FileList.csv')
    df = pd.read_csv(split_file)
    df = df[df['Split'].str.lower() == split.lower()]

    video_dir = os.path.join(data_dir, 'Videos')
    cache_dir = os.path.join(data_dir, f'cache_frames{num_frames}_period{period}')

    dataset = EchoDynamicDatasetLazy(
        data_dir=video_dir,
        data_df=df,
        split=split,
        frames=num_frames,
        resize=128,
        t_step=period,
        cache_dir=cache_dir,
        device=None,
        augment=None,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )

    return dataloader


def main():
    parser = argparse.ArgumentParser(description='Train CardiacPULSE')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--max_epochs', type=int, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train_cfg = config['training']
    model_cfg = config['model']
    loss_cfg = config['losses']
    data_cfg = config['data']

    if args.max_epochs is not None:
        train_cfg['max_epochs'] = args.max_epochs

    seed = train_cfg.get('seed', 42)
    L.seed_everything(seed)

    exp_name = config.get('experiment_name', 'cardiacpulse')

    print("=" * 70)
    print("CardiacPULSE: Fourier Cardiac Phase Network")
    print("=" * 70)
    print(f"Experiment:       {exp_name}")
    print()
    print("--- Key Innovation ---")
    print("  Per-pixel temporal FFT on raw pixel intensities.")
    print("  Fourier amplitude at cardiac freq = natural spatial mask.")
    print("  Weighted avg intensity = 1x HR signal (2 extrema/cycle).")
      print()
    print("--- Architecture ---")
    print(f"  Spatial attention: {model_cfg['spatial_hidden']} hidden channels")
    print(f"  FPS:               {model_cfg['fps']}")
    print(f"  Cardiac band:      [{model_cfg['cardiac_low_hz']}, {model_cfg['cardiac_high_hz']}] Hz")
    print(f"  Mask combine:      {model_cfg['mask_combine_mode']}")
    print(f"  FFT avg bins:      {model_cfg['n_avg_bins']}")
    print(f"  Num frames:        {data_cfg['num_frames']}")
    print()
    print("--- Loss Weights ---")
    for k, v in loss_cfg.items():
        print(f"  {k}: {v}")
    print()
    print("--- Training ---")
    print(f"  LR:               {train_cfg['lr']}")
    print(f"  Weight decay:     {train_cfg['weight_decay']}")
    print(f"  Max epochs:       {train_cfg['max_epochs']}")
    print(f"  Batch size:       {data_cfg['batch_size']}")
    print("=" * 70)
    print()

    model = CardiacPULSE(
        spatial_hidden=model_cfg['spatial_hidden'],
        fps=model_cfg['fps'],
        cardiac_low_hz=model_cfg['cardiac_low_hz'],
        cardiac_high_hz=model_cfg['cardiac_high_hz'],
        mask_combine_mode=model_cfg['mask_combine_mode'],
        n_avg_bins=model_cfg['n_avg_bins'],
        lambda_spectral=loss_cfg['lambda_spectral'],
        lambda_band=loss_cfg['lambda_band'],
        lambda_phase=loss_cfg['lambda_phase'],
        lambda_align=loss_cfg['lambda_align'],
        lambda_coverage=loss_cfg['lambda_coverage'],
        lambda_mask_tv=loss_cfg['lambda_mask_tv'],
        lambda_amp=loss_cfg['lambda_amp'],
        amp_threshold=loss_cfg['amp_threshold'],
        target_coverage=loss_cfg['target_coverage'],
        coverage_barrier_coeff=loss_cfg.get('coverage_barrier_coeff', 0.01),
        mask_min_floor=model_cfg.get('mask_min_floor', 0.0),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
        max_epochs=train_cfg['max_epochs'],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Dry run
    if args.dry_run:
        print("--- Dry Run ---")
        x = torch.randn(2, data_cfg['num_frames'], 1, 128, 128).clamp(0, 1)
        out = model(x)
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                dtype_str = f" ({v.dtype})" if v.is_complex() else ""
                print(f"  {k}: {v.shape}{dtype_str}")
            else:
                print(f"  {k}: {v}")
        total_loss, loss_dict = model.compute_losses(out)
        print(f"  total_loss: {total_loss.item():.4f}")
        for k, v in loss_dict.items():
            print(f"  loss/{k}: {v.item():.4f}")

        # Test backward
        total_loss.backward()
        print("\nBackward pass OK!")
        grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters()
                      if p.grad is not None}
        print(f"Non-zero grads: {sum(1 for v in grad_norms.values() if v > 0)}/{len(grad_norms)}")

        # Check key outputs
        s = out['s']
        print(f"\nSignal s(t) stats:")
        print(f"  Shape: {s.shape}")
        print(f"  Range: [{s.min().item():.4f}, {s.max().item():.4f}]")
        print(f"  Amplitude: {(s.max(dim=1).values - s.min(dim=1).values).mean().item():.4f}")

        cnn_mask = out['cnn_mask']
        print(f"CNN mask: mean={cnn_mask.mean().item():.4f}, "
              f"max={cnn_mask.max().item():.4f}")

        combined_mask = out['combined_mask']
        print(f"Combined mask: mean={combined_mask.mean().item():.4f}, "
              f"max={combined_mask.max().item():.4f}")

        cardiac_idx = out['cardiac_freq_idx']
        freqs = torch.fft.rfftfreq(data_cfg['num_frames'], d=1.0/model_cfg['fps'])
        cardiac_hz = freqs[cardiac_idx]
        print(f"Cardiac freq: idx={cardiac_idx.tolist()}, Hz={cardiac_hz.tolist()}")

        print("\nDry run successful!")
        return

    # Data
    print("Loading dataset...")
    train_loader = get_dataloader(config, 'train')
    val_loader = get_dataloader(config, 'val')
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print()

    # Output
    output_dir = Path(config.get('output_dir', './results/cardiacpulse_echonet'))
    results_dir = output_dir / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    wandb_logger = WandbLogger(
        project=config['logging']['project'],
        name=exp_name,
        save_dir=str(results_dir),
        offline=True,
    )

    # Callbacks
    checkpoint_dir = str(results_dir / 'checkpoints')
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best_model',
            monitor='val/loss',
            mode='min',
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor='val/loss',
            patience=train_cfg.get('patience', 40),
            mode='min',
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    # Trainer
    trainer = L.Trainer(
        max_epochs=train_cfg['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic=False,
    )

    print("Starting training...")
    print(f"Checkpoint dir: {checkpoint_dir}")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume,
    )

    print()
    print("=" * 70)
    print("Training complete!")
    print(f"Best model: {callbacks[0].best_model_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
