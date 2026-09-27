import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import os
import imageio
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd


def read_avi(filepath):
    """
    Read avi video from EchoNetDynamic dataset. 
    """
    reader = imageio.get_reader(filepath, 'ffmpeg')
    # Initialize a list to store frames
    frames = []

    # Iterate over each frame and turn to grayscale 
    for frame in reader:
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])
        frames.append(frame)

    # Convert list of frames to a NumPy array
    frames_np = np.array(frames)
    reader.close()
    return frames_np


def save_tensor_as_mp4_imageio(tensor, file_name="output.mp4", fps=30):
    """
    Save a tensor (B, T, C, H, W) into an MP4 video using imageio.
    Parameters:
        tensor (numpy.ndarray): The tensor of shape (B, T, C, H, W).
        file_name (str): The name of the output video file.
        fps (int): Frames per second for the output video.
    """
    B, T, C, H, W = tensor.shape
    for b in range(B):
        video_frames = tensor[b]

        # Normalize to 0-255 and convert to uint8
        if tensor.dtype != np.uint8:
            video_frames = (video_frames * 255).clip(0, 255).astype(np.uint8)

        # If grayscale (C=1), convert to RGB
        if C == 1:
            video_frames = np.repeat(video_frames, 3, axis=1)

        # Transpose to (T, H, W, C)
        video_frames = np.transpose(video_frames, (0, 2, 3, 1))

        # Write video using imageio
        writer = imageio.get_writer(file_name, fps=fps)
        for frame in video_frames:
            writer.append_data(frame)
        writer.close()
        #print(f"Saved video: {file_name[:-4]}_batch{b}.mp4")


class EntireEcho(data.Dataset):
    """
    Dataloader for EchoNetDynamic entire video loading (with resizing and temporal downsampling). 
    """
    def __init__(self, root, df, size=128, period=4):
        super().__init__()
        self.root = root
        self.period = period # downsample 
        self.df = df
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        entry = self.df.iloc[index]
        filename = entry['FileName']
        images = read_avi(os.path.join(self.root, filename + '.avi'))
        images = np.round(images).astype(float) 
        # sample if needed
        images = images[::self.period]
        sample = torch.from_numpy(images.astype(np.ubyte))
        sample = sample.unsqueeze(1)
        # Resize correctly
        sample = transforms.functional.resize(sample, (self.size,self.size))
        # Normalize to [0,1]
        sample = sample.float()/255.
        return sample
    

class Echo(data.Dataset):
    """
    Dataloader for short video clip loading. 
    """
    def __init__(self, root, df, length, period=4):
        super().__init__()
        self.root = root
        self.length = length 
        self.period = period # downsample 
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        entry = self.df.iloc[index]
        filename = entry['FileName']
        images = read_avi(os.path.join(self.root, filename + '.avi'))

        required_length = self.length * self.period

        # If video is too short, repeat it cyclically to meet requirements
        if len(images) < required_length:
            # Calculate how many times to repeat
            num_repeats = int(np.ceil(required_length / len(images)))
            # Tile the video
            images = np.tile(images, (num_repeats, 1, 1))[:required_length]

        # Random temporal cropping for short video clip
        if len(images) > required_length:
            start = np.random.randint(0, len(images) - required_length + 1)
        else:
            start = 0

        images = images[start:][::self.period][:self.length]

        return np.round(images).astype(int) 

# part of this class code adapted from https://github.com/alain-ryser/tvae 
class EchoDynamicDatasetFromCache(data.Dataset):
    """
    Dataset for echo-dynamic preprocessing
    """
    def __init__(self, data_dir, data_df, split='train', frames=25, resize=128, t_step=4,  cache_dir='.',
                 device = None, augment=None, num_worker=4):
        # Define data transforms that should be applied to dataset
        self.data_dir = data_dir
        self.size = resize
        self.num_frames = frames
        # Load data if in cache, else generate cache
        cache_fn = f"echo_dynamic_vids_split{split}_len{frames}_tstep{t_step}.pt"
        if not Path(os.path.join(cache_dir,cache_fn)).exists():
            dynamic_dataset = Echo(root=data_dir, df=data_df, length=frames, period=t_step)
            self.rebuild_video_cache(cache_fn, cache_dir, dynamic_dataset)
        self.data = torch.load(os.path.join(cache_dir,cache_fn), weights_only=False)
        self.device = device
        self.df = data_df
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sample = sample.to(self.device)
        if self.augment is not None:
            sample = self.augment(sample)
        # Return as dict for compatibility with Lightning models
        return {'video': sample} 
    
    def preprocess(self, sample):
        """
        Preprocess video frames
        """
        # Assemble tensor
        sample = torch.from_numpy(sample.astype(np.ubyte))
        sample = sample.unsqueeze(1)
        # Resize correctly
        sample = transforms.functional.resize(sample, (self.size,self.size))
        # Histogram equalization
        background_idx = sample==0
        sample[background_idx]= torch.randint_like(sample,sample[sample!=0].min(),255)[background_idx]
        sample = transforms.functional.equalize(sample)
        sample[background_idx] = 0
        # Normalization
        sample = sample.float()/255.
        return sample
    
    def rebuild_video_cache(self,fn, cache_dir, dataset):
        Path(cache_dir).mkdir(exist_ok=True)
        data = []
        skipped = 0
        for i in tqdm(range(len(dataset))):
            try:
                sample = dataset[i]
                preprocessed_sample = self.preprocess(sample)
                if self.num_frames is not None:
                    data.append(preprocessed_sample)
                else:
                    data += [x for x in preprocessed_sample]
            except Exception as e:
                skipped += 1
                print(f"\nWarning: Skipping sample {i} due to error: {e}")
                continue
        if skipped > 0:
            print(f"Skipped {skipped}/{len(dataset)} samples due to errors.")
        # Save data
        torch.save(data,os.path.join(cache_dir,fn))


class EchoDynamicDatasetLazy(data.Dataset):
    """
    Lazy-loading dataset for echo-dynamic preprocessing.
    Each sample is stored as a separate file to avoid loading entire cache into memory.
    """
    def __init__(self, data_dir, data_df, split='train', frames=25, resize=128, t_step=4, cache_dir='.',
                 device=None, augment=None, num_worker=4):
        self.data_dir = data_dir
        self.size = resize
        self.num_frames = frames
        self.device = device
        self.df = data_df.reset_index(drop=True)
        self.augment = augment
        self.t_step = t_step
        self.split = split

        # Use individual file cache directory
        self.cache_dir = os.path.join(cache_dir, f"lazy_split{split}_len{frames}_tstep{t_step}")

        # Build cache if not exists (use marker file to avoid infinite rebuild when some samples are skipped)
        cache_done_marker = os.path.join(self.cache_dir, "_cache_done.marker")
        if not Path(cache_done_marker).exists():
            self._build_lazy_cache()
            Path(cache_done_marker).touch()

        # Get list of cached files
        self.cached_files = sorted(list(Path(self.cache_dir).glob("*.pt")), key=lambda x: int(x.stem))
        print(f"Lazy dataset loaded: {len(self.cached_files)} samples from {self.cache_dir}")

    def __len__(self):
        return len(self.cached_files)

    def __getitem__(self, index):
        # Load single sample on demand
        sample = torch.load(self.cached_files[index], weights_only=True)
        if self.device is not None:
            sample = sample.to(self.device)
        if self.augment is not None:
            sample = self.augment(sample)
        return {'video': sample}

    def preprocess(self, sample):
        """Preprocess video frames"""
        sample = torch.from_numpy(sample.astype(np.ubyte))
        sample = sample.unsqueeze(1)
        sample = transforms.functional.resize(sample, (self.size, self.size))
        # Histogram equalization
        background_idx = sample == 0
        sample[background_idx] = torch.randint_like(sample, sample[sample != 0].min(), 255)[background_idx]
        sample = transforms.functional.equalize(sample)
        sample[background_idx] = 0
        sample = sample.float() / 255.
        return sample

    def _build_lazy_cache(self):
        """Build cache with individual files for each sample"""
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        dynamic_dataset = Echo(root=self.data_dir, df=self.df, length=self.num_frames, period=self.t_step)

        skipped = 0
        for i in tqdm(range(len(dynamic_dataset)), desc=f"Building lazy cache ({self.split})"):
            cache_file = os.path.join(self.cache_dir, f"{i}.pt")
            if Path(cache_file).exists():
                continue
            try:
                sample = dynamic_dataset[i]
                preprocessed_sample = self.preprocess(sample)
                torch.save(preprocessed_sample, cache_file)
            except Exception as e:
                skipped += 1
                print(f"\nWarning: Skipping sample {i} due to error: {e}")
                continue

        if skipped > 0:
            print(f"Skipped {skipped}/{len(dynamic_dataset)} samples due to errors.")


if __name__ =='__main__':
    # build data cache
    data_dir = '' # path to Videos 
    csv_path = '' # path to FileList.csv
    cache_dir = '' # path to cache 
    os.makedirs(cache_dir, exist_ok=True)
    t_step = 2 # temporal sampling step, 2 means we take one image every two frames 
    frames = 25 # number of total frames as input 

    # filter df
    df = pd.read_csv(csv_path)
    idx = np.where(df['NumberOfFrames'].values - t_step * frames > 0)[0]
    df = df.iloc[idx]
    df_train = df[df['Split']=='TRAIN'][:20]
    df_val = df[df['Split']=='VAL'][:5]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_train = EchoDynamicDatasetFromCache(data_dir, df_train, split='train', resize=128, t_step=t_step, 
                                 frames=frames, cache_dir=cache_dir,
                                 device = device)
    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)

    dataset_valid = EchoDynamicDatasetFromCache(data_dir, df_val, split='val', resize = 128, t_step=t_step, 
                                 frames=frames, cache_dir=cache_dir,
                                 device = device)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=True)

    # check the length of train loader and validation loader 
    print(len(dataloader_train), len(dataloader_valid))

    # verify the quality of cached data by generating a video example 
    for batch in dataloader_train:
        print(batch.shape)
        batch = batch.to(device, non_blocking=True)
        img = batch[0,0].repeat(3,1,1).permute((1,2,0)).detach().cpu().numpy()
        print(img.min(), img.max())
        save_tensor_as_mp4_imageio(batch.detach().cpu().numpy(), os.path.join(cache_dir, 'tmp.mp4'), fps=12)
        break