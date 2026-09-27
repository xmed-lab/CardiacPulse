"""
CAMEO Dataset Loader for CardiacPhase Framework

CAMEO dataset contains:
- 100 patients (001-100), 9 views each
  (A2C, A3C, A4C, PLAX, PSAX-Aortic, PSAX-Apical, PSAX-Mitral, PSAX-Papillary, S4C)
- DICOM format: (N_frames, 708, 1016, 3), YBR_FULL_422, uint8
- 856 valid videos (44 views missing DCM files)
- Each patient has .info file with ED/ES annotations, clinical info, image quality
- FPS varies per patient (~50-57 fps)
- No predefined split -> 80/10/10 patient-level random split
"""

import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import os
import glob
import pandas as pd
from typing import Optional, Dict, List, Tuple

CAMEO_VIEWS = [
    'A2C', 'A3C', 'A4C', 'PLAX',
    'PSAX-Aortic', 'PSAX-Apical', 'PSAX-Mitral', 'PSAX-Papillary', 'S4C'
]


def parse_cameo_info(info_path: str) -> Dict:
    """
    Parse a CAMEO .info file.

    Returns dict with:
        patient_id, sex, age, bsa, ef, diagnosis, fps,
        views: {view_name: {ed_frame, es_frame, quality_dynamic, quality_ed, quality_es}}
    """
    info = {
        'patient_id': None,
        'sex': None,
        'age': None,
        'bsa': None,
        'ef': None,
        'diagnosis': None,
        'fps': None,
        'views': {}
    }

    with open(info_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' not in line:
                continue

            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            if key == 'patient number':
                info['patient_id'] = value
            elif key == 'sex':
                info['sex'] = value
            elif key == 'age':
                info['age'] = int(value)
            elif key.startswith('body surface area'):
                info['bsa'] = float(value)
            elif key.startswith('ejection fration') or key.startswith('ejection fraction'):
                info['ef'] = int(value)
            elif key == 'diagnosis':
                info['diagnosis'] = value
            elif key.startswith('frame rate'):
                info['fps'] = float(value)
            elif key.startswith('image quality of'):
                # "image quality of A2C (Dynamic ED ES): Good Good Good"
                view_name = key.split('of ')[1].split(' (')[0]
                qualities = value.split()
                if view_name not in info['views']:
                    info['views'][view_name] = {}
                info['views'][view_name]['quality_dynamic'] = qualities[0] if len(qualities) > 0 else None
                info['views'][view_name]['quality_ed'] = qualities[1] if len(qualities) > 1 else None
                info['views'][view_name]['quality_es'] = qualities[2] if len(qualities) > 2 else None
            elif key.startswith('selected ED frame number of'):
                view_name = key.split('of ')[1].strip()
                if view_name not in info['views']:
                    info['views'][view_name] = {}
                info['views'][view_name]['ed_frame'] = int(value)
            elif key.startswith('selected ES frame number of'):
                view_name = key.split('of ')[1].strip()
                if view_name not in info['views']:
                    info['views'][view_name] = {}
                info['views'][view_name]['es_frame'] = int(value)

    return info


def build_cameo_dataframe(cameo_root: str, views: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Scan all patients, parse .info files, produce a DataFrame with one row per valid video.

    Columns: patient_id, view, dcm_path, fps, ed_frame, es_frame,
             quality_dynamic, quality_ed, quality_es, age, sex, ef, diagnosis, num_frames
    """
    import pydicom

    if views is None:
        views = CAMEO_VIEWS

    records = []
    patient_dirs = sorted([
        d for d in os.listdir(cameo_root)
        if os.path.isdir(os.path.join(cameo_root, d)) and d.isdigit()
    ])

    for patient_dir in tqdm(patient_dirs, desc="Scanning CAMEO patients"):
        patient_path = os.path.join(cameo_root, patient_dir)
        info_files = glob.glob(os.path.join(patient_path, '*_information.info'))
        if not info_files:
            continue

        info = parse_cameo_info(info_files[0])

        for view in views:
            view_dir = os.path.join(patient_path, view)
            if not os.path.isdir(view_dir):
                continue

            dcm_files = glob.glob(os.path.join(view_dir, '*.dcm'))
            if not dcm_files:
                continue

            dcm_path = dcm_files[0]

            # Get view annotations
            view_info = info['views'].get(view, {})
            ed_frame = view_info.get('ed_frame')
            es_frame = view_info.get('es_frame')
            if ed_frame is None or es_frame is None:
                continue

            # Read number of frames from DICOM header (fast, doesn't load pixels)
            try:
                ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                num_frames = int(ds.get('NumberOfFrames', 0))
            except Exception:
                continue

            if num_frames == 0:
                continue

            records.append({
                'patient_id': patient_dir,
                'view': view,
                'dcm_path': dcm_path,
                'fps': info['fps'],
                'ed_frame': ed_frame,
                'es_frame': es_frame,
                'quality_dynamic': view_info.get('quality_dynamic', 'Unknown'),
                'quality_ed': view_info.get('quality_ed', 'Unknown'),
                'quality_es': view_info.get('quality_es', 'Unknown'),
                'age': info['age'],
                'sex': info['sex'],
                'ef': info['ef'],
                'diagnosis': info['diagnosis'],
                'num_frames': num_frames,
            })

    return pd.DataFrame(records)


def split_cameo_df(
    df: pd.DataFrame,
    seed: int = 666,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split at patient level to avoid data leakage.
    Returns df_train, df_val, df_test with a 'Split' column added.
    """
    rng = np.random.RandomState(seed)
    patients = sorted(df['patient_id'].unique())
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:])

    df = df.copy()
    df['Split'] = df['patient_id'].apply(
        lambda pid: 'TRAIN' if pid in train_patients
        else ('VAL' if pid in val_patients else 'TEST')
    )

    df_train = df[df['Split'] == 'TRAIN'].reset_index(drop=True)
    df_val = df[df['Split'] == 'VAL'].reset_index(drop=True)
    df_test = df[df['Split'] == 'TEST'].reset_index(drop=True)

    return df_train, df_val, df_test


def read_dicom_video(dcm_path: str) -> np.ndarray:
    """
    Read DICOM video, convert YBR to grayscale.
    Returns numpy array (N_frames, H, W) as float32, range 0-255.
    """
    import pydicom
    ds = pydicom.dcmread(dcm_path)
    pixel_array = ds.pixel_array  # (N, H, W, 3) uint8, YBR_FULL_422

    # For YBR format, channel 0 is Y (luminance) = grayscale
    if pixel_array.ndim == 4 and pixel_array.shape[-1] == 3:
        grayscale = pixel_array[..., 0].astype(np.float32)
    elif pixel_array.ndim == 3:
        grayscale = pixel_array.astype(np.float32)
    else:
        raise ValueError(f"Unexpected pixel_array shape: {pixel_array.shape}")

    return grayscale


class CAMEOEntireVideo(data.Dataset):
    """
    Dataset for loading entire CAMEO videos for evaluation.
    Returns full video with temporal downsampling.
    """

    def __init__(self, cameo_root: str, df: pd.DataFrame, size: int = 128, period: int = 2):
        super().__init__()
        self.cameo_root = cameo_root
        self.df = df.reset_index(drop=True)
        self.size = size
        self.period = period

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> Dict:
        entry = self.df.iloc[index]

        # Read DICOM video
        video = read_dicom_video(entry['dcm_path'])  # (T, H, W), float32

        # Temporal downsampling
        video = video[::self.period]

        # Convert to tensor -> (T, 1, H, W)
        sample = torch.from_numpy(video).unsqueeze(1)

        # Resize
        if sample.shape[2] != self.size or sample.shape[3] != self.size:
            sample = transforms.functional.resize(sample, (self.size, self.size))

        # Normalize to [0, 1]
        sample = sample / 255.0

        # Adjust ED/ES indices for period (CAMEO uses 1-indexed frames)
        ed_idx = (entry['ed_frame'] - 1) // self.period
        es_idx = (entry['es_frame'] - 1) // self.period

        return {
            'video': sample,  # (T, 1, H, W)
            'patient_id': entry['patient_id'],
            'view': entry['view'],
            'ed_index': ed_idx,
            'es_index': es_idx,
            'fps': entry['fps'] / self.period,
            'ef': entry['ef'],
            'num_frames': sample.shape[0],
        }


class CAMEOClip(data.Dataset):
    """
    Dataset for loading fixed-length clips from CAMEO for training.
    Supports caching for fast repeated loading.
    """

    def __init__(
        self,
        cameo_root: str,
        df: pd.DataFrame,
        frames: int = 64,
        size: int = 128,
        period: int = 2,
        cache_dir: Optional[str] = None,
        split: str = 'train',
        augment=None,
    ):
        super().__init__()
        self.cameo_root = cameo_root
        self.size = size
        self.num_frames = frames
        self.period = period
        self.augment = augment

        # Filter out videos that are too short
        min_frames_needed = frames * period
        self.df = df[df['num_frames'] >= min_frames_needed].reset_index(drop=True)

        if len(self.df) < len(df):
            skipped = len(df) - len(self.df)
            print(f"CAMEOClip: skipped {skipped} videos shorter than {min_frames_needed} frames")

        # Cache support
        if cache_dir is not None:
            cache_fn = f"cameo_split{split}_len{frames}_tstep{period}.pt"
            cache_path = os.path.join(cache_dir, cache_fn)

            if not os.path.exists(cache_path):
                self._build_cache(cache_path)

            self.data = torch.load(cache_path)
        else:
            self.data = None

    def _build_cache(self, cache_path: str):
        """Build and save cache file with preprocessed clips."""
        Path(os.path.dirname(cache_path)).mkdir(parents=True, exist_ok=True)

        data = []
        for i in tqdm(range(len(self.df)), desc="Building CAMEO cache"):
            clip = self._load_clip(i)
            processed = self._preprocess(clip)
            data.append(processed)

        torch.save(data, cache_path)
        print(f"Saved CAMEO cache ({len(data)} clips) to {cache_path}")

    def _load_clip(self, index: int) -> np.ndarray:
        """Load a random clip from a video."""
        entry = self.df.iloc[index]
        video = read_dicom_video(entry['dcm_path'])  # (T, H, W)

        T = video.shape[0]
        required_length = self.num_frames * self.period

        # Random temporal cropping
        max_start = T - required_length
        if max_start > 0:
            start = np.random.randint(0, max_start + 1)
        else:
            start = 0

        clip = video[start:start + required_length:self.period][:self.num_frames]
        return clip

    def _preprocess(self, clip: np.ndarray) -> torch.Tensor:
        """Preprocess clip: resize, equalize, normalize."""
        sample = torch.from_numpy(clip.astype(np.ubyte))
        sample = sample.unsqueeze(1)  # (T, 1, H, W)

        # Resize
        if sample.shape[2] != self.size or sample.shape[3] != self.size:
            sample = transforms.functional.resize(sample, (self.size, self.size))

        # Histogram equalization
        background_idx = sample == 0
        if sample[sample != 0].numel() > 0:
            sample[background_idx] = torch.randint_like(
                sample, int(sample[sample != 0].min()), 255
            )[background_idx]
        sample = transforms.functional.equalize(sample)
        sample[background_idx] = 0

        # Normalize to [0, 1]
        sample = sample.float() / 255.0
        return sample

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.data is not None:
            sample = self.data[index]
        else:
            clip = self._load_clip(index)
            sample = self._preprocess(clip)

        if self.augment is not None:
            sample = self.augment(sample)

        # Return tensor directly for compatibility with MotionAnatomy2DAE
        return sample


if __name__ == '__main__':
    cameo_root = '/path/to/CAMEO'

    print("Building CAMEO DataFrame...")
    df = build_cameo_dataframe(cameo_root)
    print(f"Total valid videos: {len(df)}")
    print(f"Total patients: {df['patient_id'].nunique()}")
    print()

    print("Videos per view:")
    print(df['view'].value_counts().sort_index().to_string())
    print()

    print("Diagnosis distribution:")
    print(df.groupby('diagnosis')['patient_id'].nunique().to_string())
    print()

    print("FPS range: {:.1f} - {:.1f}".format(df['fps'].min(), df['fps'].max()))
    print("Frame count range: {} - {}".format(df['num_frames'].min(), df['num_frames'].max()))
    print()

    # Split
    df_train, df_val, df_test = split_cameo_df(df)
    print("Split (patient-level, seed=666):")
    print(f"  Train: {df_train['patient_id'].nunique()} patients, {len(df_train)} videos")
    print(f"  Val:   {df_val['patient_id'].nunique()} patients, {len(df_val)} videos")
    print(f"  Test:  {df_test['patient_id'].nunique()} patients, {len(df_test)} videos")
    print()

    # Quality distribution
    print("Image quality (dynamic) distribution:")
    print(df['quality_dynamic'].value_counts().to_string())
    print()

    # Sample video
    print("Loading sample video for sanity check...")
    sample_entry = df.iloc[0]
    video = read_dicom_video(sample_entry['dcm_path'])
    print(f"  Patient: {sample_entry['patient_id']}, View: {sample_entry['view']}")
    print(f"  Video shape: {video.shape}")
    print(f"  Value range: {video.min():.0f} - {video.max():.0f}")
    print(f"  ED frame: {sample_entry['ed_frame']}, ES frame: {sample_entry['es_frame']}")
    print(f"  FPS: {sample_entry['fps']:.2f}")

    # Test entire video dataset
    print("\nTesting CAMEOEntireVideo dataset...")
    test_dataset = CAMEOEntireVideo(cameo_root, df_test.head(2), size=128, period=2)
    sample = test_dataset[0]
    print(f"  Video tensor shape: {sample['video'].shape}")
    print(f"  Value range: {sample['video'].min():.3f} - {sample['video'].max():.3f}")
    print(f"  Patient: {sample['patient_id']}, View: {sample['view']}")
    print(f"  ED index: {sample['ed_index']}, ES index: {sample['es_index']}")
    print(f"  FPS: {sample['fps']:.2f}")
