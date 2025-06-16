import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer


class MultiModalDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_map, tokenizer_name='bert-base-multilingual-cased', max_length=128,
                 transform=None):
        # Load data from the CSV file
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_map = label_map
        # Initialize the tokenizer for the specified BERT model
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        # Default transformations for image preprocessing
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to 224x224
            transforms.ToTensor()  # Convert image to tensor
        ])

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get the row of data for the specified index
        row = self.data.iloc[idx]
        text = row['tweet_text']  # Extract text (tweet)
        image_rel_path = row['image'].replace("\\", "/")  # Normalize file path slashes

        # Check if the image path is absolute or relative to the dataset directory
        if os.path.isabs(image_rel_path) or image_rel_path.startswith(self.img_dir):
            image_path = os.path.normpath(image_rel_path)
        else:
            # If relative, join with the image directory path
            image_path = os.path.normpath(os.path.join(self.img_dir, image_rel_path))

        # Check if the image file exists
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Open and preprocess the image
        image = Image.open(image_path).convert('RGB')  # Open image in RGB mode
        image = self.transform(image)  # Apply image transformations

        # Tokenize the text (tweet) using the BERT tokenizer
        encoded_text = self.tokenizer(text, return_tensors='pt', padding='max_length',
                                      truncation=True, max_length=self.max_length)

        # Get the label for the text
        label_str = row['label']
        label = self.label_map.get(label_str, 2)  # Use default label 2 if the label is not found in the map

        # Return the encoded text, image, and label as a dictionary
        return {
            'input_ids': encoded_text['input_ids'].squeeze(0),  # Remove the batch dimension
            'attention_mask': encoded_text['attention_mask'].squeeze(0),  # Remove the batch dimension
            'image': image,
            'label': label
        }

