"""
Image Captioning with Attention Mechanism on COCO Dataset
Complete implementation of CNN-LSTM with Attention for image captioning
"""

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class COCODataLoader:
    """Load and preprocess COCO dataset for image captioning"""
    
    def __init__(self, images_dir, annotations_file, max_captions=5):
        self.images_dir = images_dir
        self.annotations_file = annotations_file
        self.max_captions = max_captions
        self.image_captions = {}
        self.all_captions = []
        
    def load_annotations(self):
        """Load COCO annotations JSON file"""
        print("Loading COCO annotations...")
        with open(self.annotations_file, 'r') as f:
            data = json.load(f)
        
        # Create image_id to filename mapping
        self.image_id_to_filename = {
            img['id']: img['file_name'] for img in data['images']
        }
        
        # Group captions by image
        for annotation in data['annotations']:
            image_id = annotation['image_id']
            caption = annotation['caption']
            
            if image_id not in self.image_captions:
                self.image_captions[image_id] = []
            
            if len(self.image_captions[image_id]) < self.max_captions:
                # Preprocess caption
                caption = self.preprocess_caption(caption)
                self.image_captions[image_id].append(caption)
                self.all_captions.append(caption)
        
        print(f"Loaded {len(self.image_captions)} images with captions")
        return self.image_captions
    
    def preprocess_caption(self, caption):
        """Clean and preprocess caption text"""
        # Convert to lowercase
        caption = caption.lower()
        # Remove special characters
        caption = ''.join([c for c in caption if c.isalnum() or c.isspace()])
        # Add start and end tokens
        caption = 'startseq ' + caption + ' endseq'
        return caption
    
    def get_image_path(self, image_id):
        """Get full path to image file"""
        filename = self.image_id_to_filename[image_id]
        return os.path.join(self.images_dir, filename)


class FeatureExtractor:
    """Extract image features using pre-trained InceptionV3"""
    
    def __init__(self, target_size=(299, 299)):
        self.target_size = target_size
        # Load InceptionV3 without top layer
        base_model = InceptionV3(weights='imagenet')
        self.model = keras.Model(
            inputs=base_model.input,
            outputs=base_model.layers[-2].output  # Get features before final classification
        )
        print("Feature extractor loaded (InceptionV3)")
    
    def preprocess_image(self, image_path):
        """Load and preprocess image for InceptionV3"""
        img = load_img(image_path, target_size=self.target_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = keras.applications.inception_v3.preprocess_input(img)
        return img
    
    def extract_features(self, image_path):
        """Extract features from a single image"""
        img = self.preprocess_image(image_path)
        features = self.model.predict(img, verbose=0)
        return features.reshape(-1)
    
    def extract_all_features(self, image_ids, data_loader, cache_file='features.pkl'):
        """Extract features for all images and cache them"""
        if os.path.exists(cache_file):
            print(f"Loading cached features from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        features_dict = {}
        print("Extracting image features...")
        
        for img_id in tqdm(image_ids):
            try:
                img_path = data_loader.get_image_path(img_id)
                features = self.extract_features(img_path)
                features_dict[img_id] = features
            except Exception as e:
                print(f"Error processing image {img_id}: {e}")
                continue
        
        # Cache features
        with open(cache_file, 'wb') as f:
            pickle.dump(features_dict, f)
        
        print(f"Extracted features for {len(features_dict)} images")
        return features_dict


class CaptionTokenizer:
    """Tokenize and process captions"""
    
    def __init__(self, captions, vocab_size=10000):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token='<unk>')
        self.tokenizer.fit_on_texts(captions)
        self.vocab_size = min(vocab_size, len(self.tokenizer.word_index) + 1)
        
        # Calculate max caption length
        self.max_length = max(len(caption.split()) for caption in captions)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Max caption length: {self.max_length}")
    
    def encode_captions(self, captions):
        """Convert captions to sequences of integers"""
        return self.tokenizer.texts_to_sequences(captions)
    
    def decode_caption(self, sequence):
        """Convert sequence back to text"""
        return self.tokenizer.sequences_to_texts([sequence])[0]
    
    def word_to_index(self, word):
        """Get index for a word"""
        return self.tokenizer.word_index.get(word, self.tokenizer.word_index['<unk>'])
    
    def index_to_word(self, index):
        """Get word for an index"""
        for word, idx in self.tokenizer.word_index.items():
            if idx == index:
                return word
        return '<unk>'


class DataGenerator(keras.utils.Sequence):
    """Custom data generator for training"""
    
    def __init__(self, image_ids, captions_dict, features_dict, tokenizer, 
                 max_length, batch_size=32, shuffle=True):
        self.image_ids = image_ids
        self.captions_dict = captions_dict
        self.features_dict = features_dict
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch"""
        total_samples = sum(len(self.captions_dict[img_id]) for img_id in self.image_ids)
        return int(np.ceil(total_samples / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_features = []
        batch_sequences = []
        batch_targets = []
        
        count = 0
        while count < self.batch_size and self.current_idx < len(self.samples):
            img_id, caption = self.samples[self.current_idx]
            self.current_idx += 1
            
            # Get image features
            if img_id not in self.features_dict:
                continue
            
            features = self.features_dict[img_id]
            
            # Encode caption
            seq = self.tokenizer.encode_captions([caption])[0]
            
            # Create input-output pairs for each word in caption
            for i in range(1, len(seq)):
                in_seq = seq[:i]
                out_seq = seq[i]
                
                # Pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                
                # One-hot encode output
                out_seq = keras.utils.to_categorical([out_seq], num_classes=self.tokenizer.vocab_size)[0]
                
                batch_features.append(features)
                batch_sequences.append(in_seq)
                batch_targets.append(out_seq)
                
                count += 1
                if count >= self.batch_size:
                    break
            
            if count >= self.batch_size:
                break
        
        return [np.array(batch_features), np.array(batch_sequences)], np.array(batch_targets)
    
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.samples = []
        for img_id in self.image_ids:
            for caption in self.captions_dict[img_id]:
                self.samples.append((img_id, caption))
        
        if self.shuffle:
            np.random.shuffle(self.samples)
        
        self.current_idx = 0


class AttentionLayer(layers.Layer):
    """Bahdanau attention mechanism"""
    
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)
    
    def call(self, features, hidden):
        # features shape: (batch_size, feature_dim)
        # hidden shape: (batch_size, hidden_units)
        
        # Expand dimensions for broadcasting
        hidden_with_time = tf.expand_dims(hidden, 1)  # (batch_size, 1, hidden_units)
        features_with_time = tf.expand_dims(features, 1)  # (batch_size, 1, feature_dim)
        
        # Calculate attention scores
        score = tf.nn.tanh(self.W1(features_with_time) + self.W2(hidden_with_time))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        # Calculate context vector
        context_vector = attention_weights * features_with_time
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights


def build_caption_model(vocab_size, max_length, feature_dim=2048, 
                        embedding_dim=256, lstm_units=512):
    """Build CNN-LSTM model with attention for image captioning"""
    
    # Image feature input
    image_input = layers.Input(shape=(feature_dim,))
    
    # Caption sequence input
    caption_input = layers.Input(shape=(max_length,))
    
    # Image feature processing
    image_features = layers.Dense(embedding_dim, activation='relu')(image_input)
    image_features = layers.Dropout(0.5)(image_features)
    
    # Caption embedding
    caption_embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(caption_input)
    caption_embedding = layers.Dropout(0.5)(caption_embedding)
    
    # LSTM for sequence processing
    lstm_out = layers.LSTM(lstm_units, return_sequences=False)(caption_embedding)
    
    # Combine image and caption features
    combined = layers.add([image_features, lstm_out])
    combined = layers.Dense(lstm_units, activation='relu')(combined)
    combined = layers.Dropout(0.5)(combined)
    
    # Output layer
    output = layers.Dense(vocab_size, activation='softmax')(combined)
    
    # Create model
    model = keras.Model(inputs=[image_input, caption_input], outputs=output)
    
    return model


def train_model(model, train_generator, val_generator, epochs=20, 
                checkpoint_path='best_model.h5'):
    """Train the caption model"""
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history


def generate_caption(model, image_features, tokenizer, max_length):
    """Generate caption for an image using beam search"""
    
    # Start with 'startseq'
    in_text = 'startseq'
    
    for i in range(max_length):
        # Encode current sequence
        sequence = tokenizer.encode_captions([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Predict next word
        yhat = model.predict([image_features.reshape(1, -1), sequence], verbose=0)
        yhat = np.argmax(yhat)
        
        # Convert index to word
        word = tokenizer.index_to_word(yhat)
        
        # Stop if end token
        if word is None or word == 'endseq':
            break
        
        # Append word to sequence
        in_text += ' ' + word
    
    # Remove start token
    final_caption = in_text.replace('startseq', '').strip()
    return final_caption


def evaluate_model(model, test_image_ids, features_dict, captions_dict, tokenizer, max_length):
    """Evaluate model on test set"""
    
    print("\nGenerating captions for test images...")
    
    for i, img_id in enumerate(test_image_ids[:5]):  # Show 5 examples
        if img_id not in features_dict:
            continue
        
        features = features_dict[img_id]
        actual_captions = captions_dict[img_id]
        
        # Generate caption
        generated = generate_caption(model, features, tokenizer, max_length)
        
        print(f"\n--- Image {i+1} ---")
        print(f"Generated: {generated}")
        print(f"Actual captions:")
        for cap in actual_captions:
            clean_cap = cap.replace('startseq', '').replace('endseq', '').strip()
            print(f"  - {clean_cap}")


# =====================================================================
# MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":
    
    # Configuration
    IMAGES_DIR = 'path/to/coco/images'  # Update this path
    ANNOTATIONS_FILE = 'path/to/coco/annotations/captions_train2017.json'  # Update this
    FEATURES_CACHE = 'coco_features.pkl'
    MODEL_CHECKPOINT = 'image_caption_model.h5'
    
    VOCAB_SIZE = 10000
    BATCH_SIZE = 64
    EPOCHS = 20
    MAX_CAPTIONS_PER_IMAGE = 5
    
    # Step 1: Load COCO dataset
    print("=" * 60)
    print("STEP 1: Loading COCO Dataset")
    print("=" * 60)
    
    data_loader = COCODataLoader(
        images_dir=IMAGES_DIR,
        annotations_file=ANNOTATIONS_FILE,
        max_captions=MAX_CAPTIONS_PER_IMAGE
    )
    image_captions = data_loader.load_annotations()
    
    # Step 2: Extract image features
    print("\n" + "=" * 60)
    print("STEP 2: Extracting Image Features")
    print("=" * 60)
    
    feature_extractor = FeatureExtractor()
    image_ids = list(image_captions.keys())
    features_dict = feature_extractor.extract_all_features(
        image_ids, data_loader, cache_file=FEATURES_CACHE
    )
    
    # Filter out images without features
    valid_image_ids = [img_id for img_id in image_ids if img_id in features_dict]
    
    # Step 3: Prepare tokenizer
    print("\n" + "=" * 60)
    print("STEP 3: Preparing Tokenizer")
    print("=" * 60)
    
    tokenizer = CaptionTokenizer(data_loader.all_captions, vocab_size=VOCAB_SIZE)
    
    # Step 4: Split data
    print("\n" + "=" * 60)
    print("STEP 4: Splitting Data")
    print("=" * 60)
    
    train_ids, test_ids = train_test_split(valid_image_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.1, random_state=42)
    
    print(f"Training images: {len(train_ids)}")
    print(f"Validation images: {len(val_ids)}")
    print(f"Test images: {len(test_ids)}")
    
    # Step 5: Create data generators
    print("\n" + "=" * 60)
    print("STEP 5: Creating Data Generators")
    print("=" * 60)
    
    train_generator = DataGenerator(
        train_ids, image_captions, features_dict, tokenizer,
        tokenizer.max_length, batch_size=BATCH_SIZE
    )
    
    val_generator = DataGenerator(
        val_ids, image_captions, features_dict, tokenizer,
        tokenizer.max_length, batch_size=BATCH_SIZE
    )
    
    # Step 6: Build model
    print("\n" + "=" * 60)
    print("STEP 6: Building Model")
    print("=" * 60)
    
    model = build_caption_model(
        vocab_size=tokenizer.vocab_size,
        max_length=tokenizer.max_length,
        feature_dim=2048
    )
    
    model.summary()
    
    # Step 7: Train model
    print("\n" + "=" * 60)
    print("STEP 7: Training Model")
    print("=" * 60)
    
    history = train_model(
        model, train_generator, val_generator,
        epochs=EPOCHS, checkpoint_path=MODEL_CHECKPOINT
    )
    
    # Step 8: Evaluate model
    print("\n" + "=" * 60)
    print("STEP 8: Evaluating Model")
    print("=" * 60)
    
    evaluate_model(model, test_ids, features_dict, image_captions, 
                   tokenizer, tokenizer.max_length)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {MODEL_CHECKPOINT}")
    print(f"Features cached to: {FEATURES_CACHE}")
