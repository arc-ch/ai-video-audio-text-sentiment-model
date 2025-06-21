from torch.utils.data import Dataset, DataLoader  # Dataset class and DataLoader for batching/loading data
import pandas as pd  # For reading CSV files and data manipulation
import torch.utils.data.dataloader  # Contains utilities for data loading (used internally)
from transformers import AutoTokenizer  # Tokenizer from Hugging Face Transformers for text preprocessing
import os  # OS utilities like file path handling and environment variables
import cv2  # OpenCV for video processing (reading frames, resizing)
import numpy as np  # Numerical operations and array handling
import torch  # PyTorch core library for tensor operations
import subprocess  # To run shell commands (extract audio with ffmpeg)
import torchaudio  # Audio processing utilities compatible with PyTorch

# Disable parallelism in tokenizer to avoid warnings/errors in some environments
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MELDDataset(Dataset):
    """
    Custom Dataset for loading MELD dataset samples, which include
    video, audio, and text data along with their sentiment and emotion labels.
    """

    def __init__(self, csv_path, video_dir):
        # Load CSV file containing metadata (utterances, labels, etc.)
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir

        # Initialize BERT tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Label maps for emotions and sentiments
        self.emotion_map = {
            'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3,
            'neutral': 4, 'sadness': 5, 'surprise': 6
        }
        self.sentiment_map = {
            'negative': 0, 'neutral': 1, 'positive': 2
        }

    def _load_video_frames(self, video_path):
        """
        Loads up to 30 frames from a video file, resizes them to 224x224,
        normalizes pixel values, and returns a tensor shaped
        [frames, channels, height, width].
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            # Check if video can be opened
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")

            # Read first frame to verify video integrity
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found: {video_path}")

            # Reset video frame pointer to start (important!)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Extract up to 30 frames
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame to 224x224
                frame = cv2.resize(frame, (224, 224))
                # Normalize pixel values to range [0, 1]
                frame = frame / 255.0
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video error: {str(e)}")
        finally:
            # Release video capture resource
            cap.release()

        if len(frames) == 0:
            raise ValueError("No frames could be extracted")

        # Pad with zeros if less than 30 frames, or truncate if more
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        # Convert frames to tensor and reorder dims: [frames, height, width, channels] -> [frames, channels, height, width]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

    def _extract_audio_features(self, video_path):
        """
        Extracts audio from video as a wav file, computes mel spectrogram,
        normalizes it, pads or truncates to length 300, and returns a tensor.
        """
        audio_path = video_path.replace('.mp4', '.wav')

        try:
            # Extract audio with ffmpeg (16kHz mono wav)
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',  # no video
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Load audio waveform
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample if sample rate != 16000 Hz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Compute Mel Spectrogram
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )
            mel_spec = mel_spectrogram(waveform)

            # Normalize mel spectrogram (zero mean, unit variance)
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            # Pad or truncate to fixed length 300 frames along time dimension
            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Audio error: {str(e)}")
        finally:
            # Clean up temporary wav file
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def __len__(self):
        # Return number of samples in dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Allows indexing via integer or tensor
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        row = self.data.iloc[idx]  # Get metadata row from CSV

        try:
            # Construct expected video filename from CSV columns
            video_filename = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""
            path = os.path.join(self.video_dir, video_filename)

            # Check if video file exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"No video found for filename: {path}")

            # Tokenize the utterance text (pad/truncate to length 128)
            text_inputs = self.tokenizer(
                row['Utterance'],
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )

            # Load video frames as tensor
            video_frames = self._load_video_frames(path)
            # Extract audio mel spectrogram features
            audio_features = self._extract_audio_features(path)

            # Map emotion and sentiment labels to integers
            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]

            return {
                'text_inputs': {
                    'input_ids': text_inputs['input_ids'].squeeze(),       # [seq_len]
                    'attention_mask': text_inputs['attention_mask'].squeeze()  # [seq_len]
                },
                'video_frames': video_frames,         # [30, 3, 224, 224]
                'audio_features': audio_features,     # [1, 64, 300]
                'emotion_label': torch.tensor(emotion_label),       # scalar tensor
                'sentiment_label': torch.tensor(sentiment_label)    # scalar tensor
            }

        except Exception as e:
            # Print error and skip sample if any problem occurs
            print(f"Error processing {path}: {str(e)}")
            return None


def collate_fn(batch):
    """
    Custom collate function for DataLoader to filter out None samples
    and use default collation for the rest.
    """
    batch = list(filter(None, batch))  # Remove failed samples
    return torch.utils.data.dataloader.default_collate(batch)


def prepare_dataloaders(train_csv, train_video_dir,
                        dev_csv, dev_video_dir,
                        test_csv, test_video_dir, batch_size=32):
    """
    Prepare DataLoader objects for train, dev, and test sets.
    """
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    # Prepare dataloaders using dataset CSVs and video directories
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        '../dataset/train/train_sent_emo.csv', '../dataset/train/train_splits',
        '../dataset/dev/dev_sent_emo.csv', '../dataset/dev/dev_splits_complete',
        '../dataset/test/test_sent_emo.csv', '../dataset/test/output_repeated_splits_test'
    )

    # Iterate through first batch of training data and print shapes
    for batch in train_loader:
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)      # (batch_size, 30, 3, 224, 224)
        print(batch['audio_features'].shape)    # (batch_size, 1, 64, 300)
        print(batch['emotion_label'])            # (batch_size,)
        print(batch['sentiment_label'])          # (batch_size,)
        break
