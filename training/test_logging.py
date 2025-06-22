from collections import namedtuple
import torch
from torch.utils.data import DataLoader
from models import MultimodalSentimentModel, MultimodalTrainer

def test_logging():
    # Create a simple namedtuple Batch to mimic a batch of data with required fields
    Batch = namedtuple('Batch', ['text_inputs', 'video_frames', 'audio_features'])
    
    # Create a mock batch with tensors filled with ones for text, video, and audio inputs
    # text_inputs is a dict with 'input_ids' and 'attention_mask', both are tensors of shape [1]
    mock_batch = Batch(
        text_inputs={'input_ids': torch.ones(1), 'attention_mask': torch.ones(1)},
        video_frames=torch.ones(1),          # video input tensor, shape [1]
        audio_features=torch.ones(1)         # audio input tensor, shape [1]
    )
    
    # Create a DataLoader with just one batch (mock_batch) to simulate training and validation loaders
    mock_loader = DataLoader([mock_batch])

    # Instantiate the model
    model = MultimodalSentimentModel()
    
    # Instantiate the trainer with model, train loader, and validation loader (both mock_loader here)
    trainer = MultimodalTrainer(model, mock_loader, mock_loader)

    # Define dummy training losses to simulate logged losses during training phase
    train_losses = {
        'total': 2.5,
        'emotion': 1.0,
        'sentiment': 1.5
    }

    # Log training losses using the trainer's log_metrics method
    trainer.log_metrics(train_losses, phase="train")

    # Define dummy validation losses and metrics to simulate validation phase logging
    val_losses = {
        'total': 1.5,
        'emotion': 0.5,
        'sentiment': 1.0
    }
    val_metrics = {
        'emotion_precision': 0.65,
        'emotion_accuracy': 0.75,
        'sentiment_precision': 0.85,
        'sentiment_accuracy': 0.95
    }

    # Log validation losses and metrics
    trainer.log_metrics(val_losses, val_metrics, phase="val")


if __name__ == "__main__":
    test_logging()

### Explanation:

# * **Purpose:**
#   This test function is designed to verify the logging mechanism in the `MultimodalTrainer` class, which typically logs losses and metrics during training and validation.

# * **How it works:**

#   * We create a minimal synthetic batch of data (`mock_batch`) with dummy tensors, wrapped inside a namedtuple to mimic the real batch structure the trainer expects.
#   * We use a `DataLoader` with this single batch to simulate the training and validation data loaders.
#   * We instantiate the model and trainer.
#   * We create example loss values for training (`train_losses`) and validation (`val_losses`) and metrics for validation (`val_metrics`).
#   * The `log_metrics()` method of the trainer is called for both training and validation phases. This method internally updates or writes logs (usually to TensorBoard or similar).

# * **Why:**
#   Testing logging independently like this helps confirm that the logging functionality does not raise errors and works as expected when given mock data and metrics — without needing to run a full training loop.

# * **Additional notes:**

#   * `torch.ones(1)` creates a tensor with shape `[1]` filled with ones, just placeholders.
#   * The `phase` parameter in `log_metrics()` specifies if we are logging for the training or validation stage.
#   * In a real scenario, the trainer would log these values to visualization tools like TensorBoard for monitoring training progress.
