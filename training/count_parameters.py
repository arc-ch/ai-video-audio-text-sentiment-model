from models import MultimodalSentimentModel

def count_parameters(model):
    # Dictionary to store parameter counts for different parts of the model
    params_dict = {
        'text_encoder': 0,
        'video_encoder': 0,
        'audio_encoder': 0,
        'fusion_layer': 0,
        'emotion_classifier': 0,
        'sentiment_classifier': 0
    }

    total_params = 0  # Variable to keep track of total trainable parameters
    
    # Loop through all named parameters in the model
    for name, param in model.named_parameters():
        # Check if this parameter should be trained (not frozen)
        if param.requires_grad:
            param_count = param.numel()  # Number of elements in this parameter tensor
            total_params += param_count  # Add to total trainable params count
            
            # Add the count to the appropriate component based on parameter's name
            if 'text_encoder' in name:
                params_dict['text_encoder'] += param_count
            elif 'video_encoder' in name:
                params_dict['video_encoder'] += param_count
            elif 'audio_encoder' in name:
                params_dict['audio_encoder'] += param_count
            elif 'fusion_layer' in name:
                params_dict['fusion_layer'] += param_count
            elif 'emotion_classifier' in name:
                params_dict['emotion_classifier'] += param_count
            elif 'sentiment_classifier' in name:
                params_dict['sentiment_classifier'] += param_count

    # Return dictionary with counts per component, and total count
    return params_dict, total_params


if __name__ == "__main__":
    model = MultimodalSentimentModel()  # Instantiate the full multimodal sentiment model
    param_dics, total_params = count_parameters(model)  # Get parameter counts
    
    print("Parameter count by component")
    # Print the parameter counts for each part of the model, formatted with commas
    for component, count in param_dics.items():
        print(f"{component:20s}: {count:,} parameters")

    print("\nTotal trainable parameters", f"{total_params:,}")


# --------------------------------------------------------------------
# 🔍 PARAMETER COUNT SUMMARY FOR MULTIMODAL MODEL
#
# This script breaks down how many trainable parameters exist in each
# component of the MultimodalSentimentModel:
#    • text_encoder         → 98,432 trainable parameters
#    • video_encoder        → 65,664 trainable parameters
#    • audio_encoder        → 16,512 trainable parameters
#    • fusion_layer         → 99,072 trainable parameters
#    • emotion_classifier   → 16,903 trainable parameters
#    • sentiment_classifier → 16,643 trainable parameters
#
# WHY do text and video encoders still have trainable params even though
# we froze them using `requires_grad = False`?
#
# ✅ EXPLANATION:
# You *partially* froze the pretrained parts:
#    - For `TextEncoder`, you froze all BERT weights via:
#          for param in self.bert.parameters():
#              param.requires_grad = False
#      BUT: you added a new `Linear(768 → 128)` layer after that, which is trainable.
#
#    - For `VideoEncoder`, you froze all backbone weights:
#          for param in self.backbone.parameters():
#              param.requires_grad = False
#      BUT: you replaced the final classification head (`self.backbone.fc`) with:
#          nn.Sequential(Linear + ReLU + Dropout), which is trainable.
#
#    - For AudioEncoder, you froze all convolutional layers:
#         for param in self.conv_layers.parameters():
#             param.requires_grad = False
#     BUT: the projection layer (`self.projection`), which consists of
#         Linear + ReLU + Dropout, remains trainable by default. 
# So only the parameters in the Linear layer inside `self.projection` contribute
# to the trainable parameters (16,512 params), while the conv layers are frozen.

#
# 💡 So only the *new layers you added* are trainable, and that’s what’s being
# counted here. The frozen pretrained backbone layers are excluded from total.
#
# 🧮 This helps verify that transfer learning is correctly applied — fine-tuning
# just the parts you want to train.


# * **TextEncoder projection:** 768 × 128 + 128 = 98,432 params
# * **VideoEncoder final fc layer:** (512 × 128) + 128 = 65,664 params (assuming backbone.fc.in_features = 512)
# * **AudioEncoder projection:** 128 × 128 + 128 = 16,512 params

# (Linear weights = input_features × output_features; Bias = output_features)


# BUT Why Count Trainable Parameters? 🎯
# | Benefit               | Why It Matters                           |
# | --------------------- | ---------------------------------------- |
# | ✅ Debugging           | Check if freezing/unfreezing worked      |
# | ✅ Efficiency          | Ensure model isn’t too heavy             |
# | ✅ Overfitting control | Avoid too-large models on small data     |
# | ✅ Comparisons         | Benchmark different architectures fairly |
# | ✅ Transparency        | Standard metric for reports/papers       |
