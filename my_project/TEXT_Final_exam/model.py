import torch
import torch.nn as nn
from transformers import BlipModel, BertModel


class CrossAttentionFusion(nn.Module):
    def __init__(self, blip_hidden=512, fused_hidden=768, num_classes=3):
        super().__init__()
        # Load pre-trained BLIP model for vision-to-text feature extraction
        self.blip = BlipModel.from_pretrained('Salesforce/blip-image-captioning-base')
        # Load pre-trained BERT model for text processing
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')

        # Projection layer to match BLIP output dimension to the fused hidden size
        self.vision_proj = nn.Linear(blip_hidden, fused_hidden)

        # Cross-attention mechanism to fuse vision and text features
        self.cross_attention = nn.MultiheadAttention(embed_dim=fused_hidden, num_heads=8, batch_first=True)

        # Classifier to predict the final output class from the fused features
        self.classifier = nn.Sequential(
            nn.Linear(fused_hidden, 256),  # Fully connected layer to reduce feature dimension
            nn.ReLU(),  # Non-linear activation function
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(256, num_classes)  # Output layer to match the number of classes
        )

    def forward(self, input_ids, attention_mask, image):
        # Extract features from the image using the BLIP model (freeze weights)
        with torch.no_grad():
            vision_out = self.blip.get_image_features(pixel_values=image)  # [batch, 512]

        # Project the vision features to match the fused hidden size (768)
        vision_out = self.vision_proj(vision_out).unsqueeze(1)  # → [batch, 1, 768]

        # Extract features from the text using BERT model
        text_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask).last_hidden_state  # [batch, seq_len, 768]

        # Apply cross-attention to combine text and vision features
        fused, _ = self.cross_attention(query=text_out, key=vision_out, value=vision_out)

        # Use the [CLS] token representation for classification
        pooled = fused[:, 0, :]  # [CLS] token representation: [batch, 768]

        # Pass the pooled features through the classifier to get the logits
        logits = self.classifier(pooled)

        return logits
