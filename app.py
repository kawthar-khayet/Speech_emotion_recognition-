import os
import numpy as np
import librosa
import torch
import gradio as gr
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = os.path.expanduser("~/audio-project/wav2vec2-final")
SR = 16000
MAX_LENGTH = SR * 3  # 3 seconds

# ============================================================
# LOAD MODEL
# ============================================================
print("Loading model...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    print("Using GPU")
else:
    print("Using CPU")

id2label = model.config.id2label
print(f"Classes: {list(id2label.values())}")
print("Model loaded!")

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_emotion(audio):
    if audio is None:
        return {label: 0.0 for label in id2label.values()}

    sr_in, waveform = audio

    # Convert to float32 and mono
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    elif waveform.dtype == np.int32:
        waveform = waveform.astype(np.float32) / 2147483648.0

    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)

    # Resample to 16kHz if needed
    if sr_in != SR:
        waveform = librosa.resample(waveform, orig_sr=sr_in, target_sr=SR)

    # Pad or truncate to 3 seconds
    if len(waveform) > MAX_LENGTH:
        waveform = waveform[:MAX_LENGTH]
    else:
        waveform = np.pad(waveform, (0, MAX_LENGTH - len(waveform)))

    # Feature extraction
    inputs = feature_extractor(waveform, sampling_rate=SR, return_tensors="pt", padding=False)

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

    # Return as label: confidence dict
    return {id2label[i]: float(probs[i]) for i in range(len(probs))}

# ============================================================
# GRADIO INTERFACE
# ============================================================
demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy", label="Audio Input"),
    outputs=gr.Label(num_top_classes=7, label="Predicted Emotion"),
    title="Speech Emotion Recognition",
    description="Record your voice or upload an audio file to detect the emotion. Model: Fine-tuned Wav2Vec2 on RAVDESS, CREMA-D, SAVEE, TESS datasets.",
    examples=[],

)

if __name__ == "__main__":
    demo.launch(share=True)
