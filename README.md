# Vision-Caption
Deep learning–based image captioning system using an encoder–decoder architecture with Xception for visual feature extraction and LSTM for caption generation. BLEU score above 0.5.


## Overview
VisionCaption is a deep learning–based image captioning system that automatically generates natural language descriptions for input images. The project is built using an encoder–decoder architecture, where a convolutional neural network (CNN) extracts visual features and a recurrent neural network (RNN) generates corresponding textual captions.

The encoder uses a pretrained Xception model to capture high-level image representations, while the decoder employs Long Short-Term Memory (LSTM) networks to produce coherent and context-aware sentences.

## Objectives
- Automatically generate descriptive captions for images.
- Learn joint representations of visual and textual data.
- Evaluate model performance using standard NLP metrics.

## Architecture
The system follows an Encoder–Decoder framework:

### Encoder
- Pretrained Xception CNN.
- Removes final classification layer.
- Outputs a fixed-length feature vector for each image.

### Decoder
- LSTM-based sequence model.
- Takes image features and previous words as input.
- Predicts the next word in the caption.

## Dataset
The model is trained on an image-caption dataset consisting of:
- A set of images.
- Multiple human-annotated captions per image.

Captions are preprocessed using:
- Lowercasing
- Tokenization
- Removal of punctuation
- Padding and vocabulary indexing

## Preprocessing
Text preprocessing steps include:
- Cleaning and normalization
- Vocabulary creation
- Integer encoding
- Sequence padding

Images are preprocessed by:
- Resizing to Xception input size
- Normalization
- Feature extraction using the encoder

## Training
- Loss function: Categorical Cross-Entropy
- Optimizer: Adam
- Training uses teacher forcing.
- Model is trained for multiple epochs until convergence.

## Evaluation
Model performance is evaluated using the BLEU (Bilingual Evaluation Understudy) score.  
The system achieves a BLEU score greater than 0.5, indicating satisfactory caption quality.

## Results
The trained model can generate semantically meaningful captions for unseen images, capturing key objects and contextual relationships.

## Tech Stack
- Python  
- TensorFlow / Keras  
- Xception (CNN)  
- LSTM (RNN)  
- NLTK  
- NumPy, Pandas  

## Project Structure
