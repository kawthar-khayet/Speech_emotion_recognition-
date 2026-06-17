import os
import numpy as np
import pandas as pd
import librosa
import glob
import pickle
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_from_disk
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import torch

# ============================================================
# CONFIG
# ============================================================
DATASET_PATH = os.path.expanduser("~/audio-project/Datasets")
OUTPUT_DIR = os.path.expanduser("~/audio-project/wav2vec2-emotion")
RESULTS_DIR = os.path.expanduser("~/audio-project/wav2vec2-results")
HF_CACHE = os.path.expanduser("~/audio-project/hf_cache")
MODEL_NAME = "facebook/wav2vec2-base"
SR = 16000
MAX_LENGTH = SR * 3  # 3 seconds
NUM_EPOCHS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(HF_CACHE, exist_ok=True)

# ============================================================
# 1. PARSE DATASETS (same logic as notebook)
# ============================================================
print("=" * 50)
print("1. Parsing datasets...")
print("=" * 50)

RAVDESS_PATH = os.path.join(DATASET_PATH, "RAVDESS")
CREMAD_PATH = os.path.join(DATASET_PATH, "CREMA-D")
SAVEE_PATH = os.path.join(DATASET_PATH, "SAVEE")
TESS_PATH = os.path.join(DATASET_PATH, "TESS")

emotion_map_str = {
    'Neutral': 0, 'Happy': 1, 'Sad': 2,
    'Angry': 3, 'Fear': 4, 'Disgust': 5, 'Surprise': 6
}

ravdess_map = {'01':'Neutral', '02':'Calm', '03':'Happy', '04':'Sad',
               '05':'Angry', '06':'Fear', '07':'Disgust', '08':'Surprise'}
crema_map = {'NEU':'Neutral', 'HAP':'Happy', 'SAD':'Sad',
             'ANG':'Angry', 'FEA':'Fear', 'DIS':'Disgust'}
savee_map = {'n':'Neutral', 'h':'Happy', 'sa':'Sad',
             'a':'Angry', 'f':'Fear', 'd':'Disgust', 'su':'Surprise'}
tess_map = {
    'neutral': 'Neutral', 'happy': 'Happy', 'sad': 'Sad',
    'angry': 'Angry', 'fear': 'Fear', 'disgust': 'Disgust',
    'ps': 'Surprise', 'pleasant_surprise': 'Surprise', 'pleasant_surprised': 'Surprise'
}

records = []

# RAVDESS
for fp in glob.glob(os.path.join(RAVDESS_PATH, "**", "*.wav"), recursive=True):
    parts = os.path.basename(fp).split('-')
    if len(parts) >= 3 and parts[2] in ravdess_map:
        emo = ravdess_map[parts[2]]
        if emo != 'Calm':
            records.append({'path': fp, 'emotion': emo})

# CREMA-D
for fp in glob.glob(os.path.join(CREMAD_PATH, "*.wav")):
    parts = os.path.basename(fp).split('_')
    if len(parts) >= 3 and parts[2] in crema_map:
        emo = crema_map[parts[2]]
        records.append({'path': fp, 'emotion': emo})

# SAVEE
for fp in glob.glob(os.path.join(SAVEE_PATH, "**", "*.wav"), recursive=True):
    parts = os.path.basename(fp).split('_')
    if len(parts) == 2:
        emo_code = ''.join([c for c in parts[1] if not c.isdigit()]).replace('.wav', '')
        if emo_code in savee_map:
            emo = savee_map[emo_code]
            records.append({'path': fp, 'emotion': emo})

# TESS
for fp in glob.glob(os.path.join(TESS_PATH, "**", "*.wav"), recursive=True):
    fn_lower = fp.lower()
    for key, emo in tess_map.items():
        if key in fn_lower:
            records.append({'path': fp, 'emotion': emo})
            break

df = pd.DataFrame(records)
df['label'] = df['emotion'].map(emotion_map_str)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

print(f"Total files: {len(df)}")
print(f"Classes: {list(emotion_map_str.keys())}")
print(df['emotion'].value_counts())

# ============================================================
# 2. TRAIN/VAL/TEST SPLIT
# ============================================================
print("\n" + "=" * 50)
print("2. Splitting data...")
print("=" * 50)

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ============================================================
# 3. LOAD AUDIO + PREPROCESS
# ============================================================
print("\n" + "=" * 50)
print("3. Loading audio and preprocessing...")
print("=" * 50)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

def load_and_preprocess(example):
    y, _ = librosa.load(example['path'], sr=SR)
    if len(y) > MAX_LENGTH:
        y = y[:MAX_LENGTH]
    else:
        y = np.pad(y, (0, MAX_LENGTH - len(y)))
    inputs = feature_extractor(y, sampling_rate=SR, return_tensors="np", padding=False)
    example['input_values'] = inputs.input_values[0]
    return example

# Process and save to disk (memory-mapped)
for name, split_df, cache_path in [
    ("Train", train_df, os.path.join(HF_CACHE, "train")),
    ("Val", val_df, os.path.join(HF_CACHE, "val")),
    ("Test", test_df, os.path.join(HF_CACHE, "test")),
]:
    if os.path.exists(cache_path):
        print(f"{name}: loading from cache...")
    else:
        print(f"{name}: processing {len(split_df)} files...")
        ds = Dataset.from_dict({
            'path': split_df['path'].tolist(),
            'label': split_df['label'].tolist()
        })
        ds = ds.map(load_and_preprocess, remove_columns=['path'])
        ds.save_to_disk(cache_path)
        del ds

# Reload as memory-mapped
train_dataset = load_from_disk(os.path.join(HF_CACHE, "train"))
val_dataset = load_from_disk(os.path.join(HF_CACHE, "val"))
test_dataset = load_from_disk(os.path.join(HF_CACHE, "test"))

train_dataset.set_format('torch')
val_dataset.set_format('torch')
test_dataset.set_format('torch')

print("Datasets ready!")

# ============================================================
# 4. MODEL
# ============================================================
print("\n" + "=" * 50)
print("4. Loading Wav2Vec2 model...")
print("=" * 50)

label2id = emotion_map_str
id2label = {v: k for k, v in emotion_map_str.items()}

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(emotion_map_str),
    label2id=label2id,
    id2label=id2label,
)
model.freeze_feature_encoder()
model.gradient_checkpointing_enable()

print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================================
# 5. TRAINING
# ============================================================
print("\n" + "=" * 50)
print("5. Training...")
print("=" * 50)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)['accuracy']
    f1 = f1_metric.compute(predictions=preds, references=labels, average='macro')['f1']
    return {'accuracy': acc, 'f1_macro': f1}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=1e-4,
    warmup_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=True,
    report_to="none",
    save_total_limit=2,
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# ============================================================
# 6. EVALUATION ON TEST SET
# ============================================================
print("\n" + "=" * 50)
print("6. Evaluating on test set...")
print("=" * 50)

test_results = trainer.evaluate(test_dataset)
print(f"Test Accuracy: {test_results['eval_accuracy']*100:.2f}%")
print(f"Test F1 Macro: {test_results['eval_f1_macro']*100:.2f}%")

# Detailed predictions
preds_output = trainer.predict(test_dataset)
y_pred = np.argmax(preds_output.predictions, axis=-1)
y_true = preds_output.label_ids

from sklearn.metrics import classification_report, confusion_matrix

labels_list = [id2label[i] for i in range(len(id2label))]
report = classification_report(y_true, y_pred, target_names=labels_list)
cm = confusion_matrix(y_true, y_pred)

print("\n--- Classification Report ---")
print(report)

# ============================================================
# 7. SAVE EVERYTHING
# ============================================================
print("\n" + "=" * 50)
print("7. Saving model and results...")
print("=" * 50)

# Save model
model_save_path = os.path.expanduser("~/audio-project/wav2vec2-final")
trainer.save_model(model_save_path)
feature_extractor.save_pretrained(model_save_path)

# Save results for notebook visualization
results = {
    'test_results': test_results,
    'y_true': y_true,
    'y_pred': y_pred,
    'labels': labels_list,
    'classification_report': report,
    'confusion_matrix': cm,
    'train_history': trainer.state.log_history,
}

with open(os.path.join(RESULTS_DIR, "results.pkl"), 'wb') as f:
    pickle.dump(results, f)

print(f"Model saved to: {model_save_path}")
print(f"Results saved to: {RESULTS_DIR}/results.pkl")
print("\nDone!")
