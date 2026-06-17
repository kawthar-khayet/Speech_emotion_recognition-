#  Speech Emotion Recognition using Deep Learning

A complete speech emotion recognition pipeline classifying **7 emotions** (Neutral, Happy, Sad, Angry, Fear, Disgust, Surprise) from audio, using CNN-LSTM as a baseline and fine-tuned Wav2Vec2 as the primary model.

---

## Trained Model

The fine-tuned Wav2Vec2 model is available on Hugging Face:  
👉 [speech-emotion-recognition-wav2vec2](https://huggingface.co/YOUR_USERNAME/speech-emotion-recognition-wav2vec2)

---

##  Results

| Model | Accuracy | F1 Macro |
|---|---|---|
| CNN-LSTM (baseline) | 54.00% | 0.53 |
| Wav2Vec2 (fine-tuned) | **84.88%** | **0.86** |

---

##  Datasets

| Dataset | Description | Files |
|---|---|---|
| [RAVDESS](https://zenodo.org/record/1188976) | Ryerson Audio-Visual Database, 24 actors | 1,440 |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | Crowd-sourced Emotional Multimodal Actors, 91 actors | 7,442 |
| [TESS](https://tspace.library.utoronto.ca/handle/1807/24487) | Toronto Emotional Speech Set, 2 actresses | 2,800 |
| [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/) | Surrey Audio-Visual Expressed Emotion, 4 actors | 480 |

**Total: 11,970 files across 7 emotions.**

---

## 📁 Project Structure

```
├── Projet_Audio_1.ipynb          # Main notebook (EDA, CNN-LSTM, results visualization)
├── train_wav2vec2.py             # Wav2Vec2 training script
├── gradio_app.py                 # Gradio web app for real-time prediction
├── rapport_ser.tex               # Project report (LaTeX)
├── requirements.txt
├── Datasets/                     # Audio datasets (not included, see below)
│   ├── RAVDESS/
│   ├── CREMA-D/
│   ├── SAVEE/
│   └── TESS/
├── wav2vec2-final/               # Saved Wav2Vec2 model weights
├── wav2vec2-results/             # Training results (results.pkl)
└── demo_files/                   # Sample audio files for demo
```

---

##  Setup

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA support
- WSL2 (if on Windows)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/speech-emotion-recognition.git
cd speech-emotion-recognition
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Download Datasets

Download the datasets and place them in the `Datasets/` directory:

- [RAVDESS](https://zenodo.org/record/1188976)
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- [TESS](https://tspace.library.utoronto.ca/handle/1807/24487)
- [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/)

---

##  Usage

### Training

**CNN-LSTM (baseline):**

Run cells 1–25 of the notebook:
```bash
jupyter notebook Projet_Audio_1.ipynb
```

**Wav2Vec2 (fine-tuning):**

```bash
python3 train_wav2vec2.py
```

### Inference — Gradio App

```bash
python3 gradio_app.py
```

Open the provided URL in your browser to record audio or upload `.wav` files for real-time emotion prediction.

### MLflow Dashboard

```bash
mlflow ui
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to view experiment metrics and training curves.

---

##  Technologies

| Library | Role |
|---|---|
| [Librosa](https://librosa.org/) | Audio feature extraction (MFCC, Mel spectrograms) |
| [TensorFlow / Keras](https://www.tensorflow.org/) | CNN-LSTM model |
| [PyTorch](https://pytorch.org/) + [HuggingFace Transformers](https://huggingface.co/docs/transformers) | Wav2Vec2 fine-tuning |
| [Gradio](https://www.gradio.app/) | Web interface for real-time prediction |
| [MLflow](https://mlflow.org/) | Experiment tracking |

---

## 📄 License

This project is for academic purposes. Dataset licenses apply individually — refer to each dataset's original source.
