# Speech Emotion Recognition using Deep Learning

A complete speech emotion recognition pipeline classifying 7 emotions (Neutral, Happy, Sad, Angry, Fear, Disgust, Surprise) from audio, using CNN-LSTM as a baseline and fine-tuned Wav2Vec2 as the primary model.

## 🎯 Trained Model

The fine-tuned Wav2Vec2 model is available on [Hugging Face](https://huggingface.co/your-username/speech-emotion-recognition-wav2vec2)

## 📊 Results

| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| CNN-LSTM (baseline) | 54.00% | 0.53 |
| Wav2Vec2 (fine-tuned) | 84.88% | 0.86 |

## 📁 Datasets

- **RAVDESS** — Ryerson Audio-Visual Database (1,440 files, 24 actors)
- **CREMA-D** — Crowd-sourced Emotional Multimodal Actors Dataset (7,442 files, 91 actors)
- **TESS** — Toronto Emotional Speech Set (2,800 files, 2 actresses)
- **SAVEE** — Surrey Audio-Visual Expressed Emotion (480 files, 4 actors)

**Total:** 11,970 files across 7 emotions.

## 📂 Project Structure
