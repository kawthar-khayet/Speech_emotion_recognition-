# Speech Emotion Recognition using Deep Learning
## A complete speech emotion recognition pipeline classifying 7 emotions (Neutral, Happy, Sad, Angry, Fear, Disgust, Surprise) from audio, using CNN-LSTM as a baseline and fine-tuned Wav2Vec2 as the primary model.

## Trained Model
The fine-tuned Wav2Vec2 model is available on Hugging Face: speech-emotion-recognition-wav2vec2

## Results
Model	Accuracy	F1 Macro
CNN-LSTM (baseline)	54.00%	0.53
Wav2Vec2 (fine-tuned)	84.88%	0.86
## Datasets
RAVDESS — Ryerson Audio-Visual Database (1,440 files, 24 actors)
CREMA-D — Crowd-sourced Emotional Multimodal Actors Dataset (7,442 files, 91 actors)
TESS — Toronto Emotional Speech Set (2,800 files, 2 actresses)
SAVEE — Surrey Audio-Visual Expressed Emotion (480 files, 4 actors)
Total: 11,970 files across 7 emotions.

## Project Structure
├── Projet_Audio_1.ipynb          # Main notebook (EDA, CNN-LSTM, results visualization)
├── train_wav2vec2.py             # Wav2Vec2 training script
├── gradio_app.py                 # Gradio web app for real-time prediction
├── rapport_ser.tex               # Project report (LaTeX)
├── Datasets/                     # Audio datasets (not included, see below)
│   ├── RAVDESS/
│   ├── CREMA-D/
│   ├── SAVEE/
│   └── TESS/
├── wav2vec2-final/               # Saved Wav2Vec2 model
├── wav2vec2-results/             # Training results (results.pkl)
├── demo_files/                   # Sample audio files for demo
└── requirements.txt
## Setup
Prerequisites
Python 3.12+
NVIDIA GPU with CUDA support
WSL2 (if on Windows)
## Installation
git clone https://github.com/YOUR_USERNAME/speech-emotion-recognition.git
cd speech-emotion-recognition
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
## Datasets
Download the datasets and place them in the Datasets/ directory:

RAVDESS
CREMA-D
TESS
SAVEE
## Usage
### Training
CNN-LSTM (baseline):

Run the notebook Projet_Audio_1.ipynb cells 1-25.

### Wav2Vec2:

python3 train_wav2vec2.py
Inference (Gradio App)
python3 gradio_app.py
Open the provided URL in your browser to record audio or upload .wav files for emotion prediction.

MLflow Dashboard
mlflow ui
Open http://127.0.0.1:5000 to view experiment metrics and training curves.

Technologies
Librosa — Audio feature extraction (MFCC, Mel spectrograms)
TensorFlow/Keras — CNN-LSTM model
PyTorch + HuggingFace Transformers — Wav2Vec2 fine-tuning
Gradio — Web interface for real-time prediction
MLflow — Experiment tracking
