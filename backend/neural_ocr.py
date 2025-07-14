"""
Advanced Neural OCR for Russian Diacritics in Sanskrit Texts
Specialized PyTorch model for recognizing Russian diacritical marks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import numpy as np
import cv2
import logging
import random
import string
from typing import List, Dict, Tuple, Optional
import os

logger = logging.getLogger(__name__)

# Russian diacritics mapping for Sanskrit
RUSSIAN_DIACRITICS = {
    # Base vowels with length markers
    'а': ['а', 'а̄'],  # a, ā
    'и': ['и', 'ӣ'],   # i, ī  
    'у': ['у', 'ӯ'],   # u, ū
    'е': ['е', 'е̄'],   # e, ē
    'о': ['о', 'о̄'],   # o, ō
    
    # Consonants with diacritics
    'р': ['р', 'р̣'],   # r, ṛ
    'л': ['л', 'л̣'],   # l, ḷ
    'н': ['н', 'н̣', 'н̇'], # n, ṇ, ṅ
    'т': ['т', 'т̣'],   # t, ṭ
    'д': ['д', 'д̣'],   # d, ḍ
    'с': ['с', 'ш̣'],   # s, ṣ
    'ш': ['ш', 'ш́'],   # ś, ś
    'м': ['м', 'м̣', 'м̇'], # m, ṃ, ṁ
    'х': ['х', 'х̣'],   # h, ḥ
    'к': ['к', 'кх'],  # k, kh
    'г': ['г', 'гх'],  # g, gh
    'ч': ['ч', 'чх'],  # c, ch
    'дж': ['дж', 'джх'], # j, jh
    'п': ['п', 'пх'],  # p, ph
    'б': ['б', 'бх'],  # b, bh
}

# All possible characters in Russian diacritical system
ALL_CHARS = []
for base_char, variants in RUSSIAN_DIACRITICS.items():
    ALL_CHARS.extend(variants)

# Add common punctuation and spaces
ALL_CHARS.extend([' ', '.', ',', ';', ':', '!', '?', '-', '\'', '"', '(', ')', '\n'])
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ALL_CHARS)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
NUM_CLASSES = len(ALL_CHARS)

logger.info(f"Neural OCR initialized with {NUM_CLASSES} character classes")

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow in deep networks"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AttentionModule(nn.Module):
    """Attention mechanism for focusing on important features"""
    
    def __init__(self, in_channels: int):
        super(AttentionModule, self).__init__()
        
        self.conv_query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        query = self.conv_query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.conv_key(x).view(batch_size, -1, height * width)
        value = self.conv_value(x).view(batch_size, -1, height * width)
        
        # Attention weights
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return x + out

class AdvancedOCRNet(nn.Module):
    """Advanced OCR Network with ResNet backbone and attention mechanism"""
    
    def __init__(self, num_classes: int = NUM_CLASSES, input_height: int = 64, input_width: int = 256):
        super(AdvancedOCRNet, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Attention module
        self.attention = AttentionModule(512)
        
        # Calculate feature map dimensions after convolutions
        # Input: 1x64x256 -> 512x4x16 (approximately)
        self.feature_height = input_height // 16
        self.feature_width = input_width // 16
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=512 * self.feature_height,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # 512 = 256 * 2 (bidirectional)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Prepare for LSTM: (batch, channels, height, width) -> (batch, width, channels*height)
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch, width, channels, height)
        x = x.view(batch_size, width, channels * height)  # (batch, width, features)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch, width, 512)
        
        # Classification for each time step
        output = self.classifier(lstm_out)  # (batch, width, num_classes)
        
        return output

class SyntheticDataGenerator:
    """Generate synthetic training data for Russian diacritics"""
    
    def __init__(self, image_height: int = 64, image_width: int = 256):
        self.image_height = image_height
        self.image_width = image_width
        self.fonts = self._load_system_fonts()
        
    def _load_system_fonts(self) -> List[str]:
        """Load available system fonts"""
        font_paths = []
        
        # Common font directories on different systems
        font_dirs = [
            '/usr/share/fonts/',
            '/System/Library/Fonts/',
            'C:/Windows/Fonts/',
            '/usr/local/share/fonts/',
        ]
        
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for root, dirs, files in os.walk(font_dir):
                    for file in files:
                        if file.endswith(('.ttf', '.otf')):
                            font_paths.append(os.path.join(root, file))
        
        # Fallback to default font if no fonts found
        if not font_paths:
            logger.warning("No system fonts found, using default font")
            return ['default']
        
        logger.info(f"Found {len(font_paths)} system fonts")
        return font_paths[:20]  # Limit to 20 fonts for variety
    
    def generate_random_text(self, min_length: int = 5, max_length: int = 20) -> str:
        """Generate random Sanskrit text with Russian diacritics"""
        length = random.randint(min_length, max_length)
        text = ""
        
        for _ in range(length):
            if random.random() < 0.8:  # 80% chance for diacritical character
                base_char = random.choice(list(RUSSIAN_DIACRITICS.keys()))
                variants = RUSSIAN_DIACRITICS[base_char]
                char = random.choice(variants)
            else:  # 20% chance for space or punctuation
                char = random.choice([' ', '.', ',', '-'])
            
            text += char
        
        return text.strip()
    
    def create_text_image(self, text: str) -> Tuple[Image.Image, str]:
        """Create image from text with various augmentations"""
        
        # Create image with white background
        img = Image.new('RGB', (self.image_width, self.image_height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Choose random font
        try:
            if self.fonts[0] != 'default':
                font_path = random.choice(self.fonts)
                font_size = random.randint(16, 28)
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        
        # Calculate text position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text
        x = max(0, (self.image_width - text_width) // 2)
        y = max(0, (self.image_height - text_height) // 2)
        
        # Random text color (dark colors)
        text_color = (
            random.randint(0, 80),
            random.randint(0, 80), 
            random.randint(0, 80)
        )
        
        # Draw text
        draw.text((x, y), text, fill=text_color, font=font)
        
        # Apply augmentations
        img = self._apply_augmentations(img)
        
        # Convert to grayscale
        img = img.convert('L')
        
        return img, text
    
    def _apply_augmentations(self, img: Image.Image) -> Image.Image:
        """Apply various augmentations to make training more robust"""
        
        # Random rotation
        if random.random() < 0.3:
            angle = random.uniform(-3, 3)
            img = img.rotate(angle, fillcolor='white')
        
        # Random brightness
        if random.random() < 0.4:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Random contrast
        if random.random() < 0.4:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.9, 1.1))
        
        # Random blur
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
        
        # Random noise
        if random.random() < 0.3:
            img_array = np.array(img)
            noise = np.random.normal(0, 5, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        
        return img
    
    def generate_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, List[str]]:
        """Generate a batch of training data"""
        images = []
        texts = []
        
        for _ in range(batch_size):
            text = self.generate_random_text()
            img, _ = self.create_text_image(text)
            
            # Convert to tensor
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
            ])
            
            img_tensor = transform(img)
            images.append(img_tensor)
            texts.append(text)
        
        return torch.stack(images), texts

class NeuralOCRService:
    """Neural OCR service for Russian diacritics recognition"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Neural OCR using device: {self.device}")
        
        # Initialize model
        self.model = AdvancedOCRNet()
        self.model.to(self.device)
        
        # Load pre-trained weights if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.info("No pre-trained model found, using randomly initialized weights")
            # Quick training on synthetic data
            self._quick_train()
        
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((64, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def load_model(self, model_path: str):
        """Load pre-trained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded pre-trained model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
    
    def save_model(self, model_path: str):
        """Save current model weights"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'char_to_idx': CHAR_TO_IDX,
                'idx_to_char': IDX_TO_CHAR
            }, model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model to {model_path}: {e}")
    
    def _quick_train(self, num_epochs: int = 5, batch_size: int = 16):
        """Quick training on synthetic data for basic functionality"""
        logger.info("Starting quick training on synthetic data...")
        
        # Create data generator
        data_gen = SyntheticDataGenerator()
        
        # Setup training
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 10  # Limited training for quick setup
            
            for batch_idx in range(num_batches):
                # Generate batch
                images, texts = data_gen.generate_batch(batch_size)
                images = images.to(self.device)
                
                # Create targets
                targets = self._texts_to_targets(texts, max_length=16)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)  # (batch, width, num_classes)
                
                # Reshape for loss calculation
                outputs = outputs.view(-1, NUM_CLASSES)
                targets = targets.view(-1)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            logger.info(f"Quick training epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        self.model.eval()
        logger.info("Quick training completed")
    
    def _texts_to_targets(self, texts: List[str], max_length: int) -> torch.Tensor:
        """Convert list of texts to target tensor"""
        batch_size = len(texts)
        targets = torch.full((batch_size, max_length), -1, dtype=torch.long)  # -1 for padding
        
        for i, text in enumerate(texts):
            for j, char in enumerate(text[:max_length]):
                if char in CHAR_TO_IDX:
                    targets[i, j] = CHAR_TO_IDX[char]
        
        return targets
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for neural network"""
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply preprocessing transform
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(self.device)
    
    def recognize_text(self, image: Image.Image) -> Tuple[str, float]:
        """Recognize text from image using neural network"""
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)  # (1, width, num_classes)
                probabilities = F.softmax(outputs, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)  # (1, width)
            
            # Decode predictions
            predicted_indices = predictions.squeeze(0).cpu().numpy()
            text = self._decode_predictions(predicted_indices, probabilities.squeeze(0))
            
            # Calculate confidence
            confidence = self._calculate_confidence(probabilities.squeeze(0), predictions.squeeze(0))
            
            logger.info(f"Neural OCR recognized: '{text}' (confidence: {confidence:.3f})")
            return text, confidence
            
        except Exception as e:
            logger.error(f"Neural OCR recognition failed: {e}")
            return "", 0.0
    
    def _decode_predictions(self, predictions: np.ndarray, probabilities: torch.Tensor) -> str:
        """Decode predictions to text with CTC-like processing"""
        text = ""
        prev_char = None
        
        for i, pred_idx in enumerate(predictions):
            if pred_idx < len(IDX_TO_CHAR):
                char = IDX_TO_CHAR[pred_idx]
                
                # Skip repeated characters and padding
                if char != prev_char and char != ' ' or (char == ' ' and prev_char != ' '):
                    # Add confidence threshold
                    char_confidence = probabilities[i, pred_idx].item()
                    if char_confidence > 0.3:  # Confidence threshold
                        text += char
                
                prev_char = char
        
        return text.strip()
    
    def _calculate_confidence(self, probabilities: torch.Tensor, predictions: torch.Tensor) -> float:
        """Calculate overall confidence score"""
        confidences = []
        
        for i, pred_idx in enumerate(predictions):
            if pred_idx < probabilities.shape[1]:
                conf = probabilities[i, pred_idx].item()
                confidences.append(conf)
        
        return np.mean(confidences) if confidences else 0.0

# Initialize global neural OCR service
neural_ocr_service = NeuralOCRService() 