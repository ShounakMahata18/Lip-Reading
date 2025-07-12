import os
import pytest
import numpy as np
import tensorflow as tf
from src.extract_mouth_region import VideoDataset

MANUAL_VIDEO_PATHS = [
    "data/raw/20250630_093632.mp4",

]

@pytest.mark.skipif(len(MANUAL_VIDEO_PATHS) == 0, reason="No video paths provided manually")
@pytest.mark.parametrize("video_path", MANUAL_VIDEO_PATHS)
class TestVideoDataset:
    def setup_method(self, method):
        self.max_frames = 75
        self.resize = (140, 46)

    def test_video_array_shape(self, video_path):
        assert os.path.exists(video_path), f"File not found: {video_path}"

        dataset_loader = VideoDataset(video_path, max_frames=self.max_frames, resize=self.resize)
        video = dataset_loader._load_video()

        assert isinstance(video, np.ndarray), "Video should be a NumPy array"
        assert video.shape == (75, 46, 140, 1), f"{video_path}: Unexpected shape {video.shape}"
        assert video.dtype == np.float32, f"{video_path}: Expected dtype float32"
        assert np.all((video >= 0) & (video <= 1)), f"{video_path}: Pixel values out of range [0, 1]"

    def test_tf_dataset_output(self, video_path):
        assert os.path.exists(video_path), f"File not found: {video_path}"

        dataset_loader = VideoDataset(video_path, max_frames=self.max_frames, resize=self.resize)
        tf_dataset = dataset_loader.load(batch_size=1)
        batch = next(iter(tf_dataset))

        if isinstance(batch, tuple):
            x = batch[0]
        else:
            x = batch

        assert isinstance(x, tf.Tensor), "Output must be a tf.Tensor"
        assert x.shape[1:] == (75, 46, 140, 1), f"{video_path}: Tensor shape mismatch: {x.shape}"
        assert tf.reduce_max(x) <= 1.0 and tf.reduce_min(x) >= 0.0, f"{video_path}: Tensor values out of expected range"
