import os
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Force CPU usage
torch.cuda.is_available = lambda: False

# Configuration
MODEL_NAME = "MLRS/wav2vec2-xls-r-300m-mt-50"
CLIPS_DIR = "/app/malteseGEC/data/monolingual/common_voice/cv-18-09-2024/clips"
VALIDATED_TSV = "/app/malteseGEC/data/monolingual/common_voice/cv-18-09-2024/validated.tsv"
OUTPUT_TSV = "/app/malteseGEC/data/monolingual/common_voice/cv-18-09-2024/validated_with_predictions.tsv"

# Get token from user
token = input("Please enter your Hugging Face token: ")

# Load model on CPU
print("Loading model on CPU...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME, use_auth_token=token)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME, use_auth_token=token)
device = torch.device("cpu")
model = model.to(device).eval()
print("Model loaded successfully on CPU")

# Read validated TSV
print(f"Reading {VALIDATED_TSV}...")
df = pd.read_csv(VALIDATED_TSV, sep="\t")

# Process in batches to save progress
BATCH_SIZE = 100
START_IDX = 0  # Change this if you need to resume

predictions = []
print(f"Processing {len(df)} audio files...")

for batch_start in range(START_IDX, len(df), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(df))
    batch_predictions = []
    
    print(f"\nProcessing batch {batch_start//BATCH_SIZE + 1} ({batch_start}-{batch_end})")
    
    for idx in tqdm(range(batch_start, batch_end)):
        row = df.iloc[idx]
        audio_path = os.path.join(CLIPS_DIR, row["path"])
        
        if os.path.exists(audio_path):
            try:
                # Load and process audio
                speech_array, sr = torchaudio.load(audio_path)
                if sr != 16000:
                    speech_array = torchaudio.transforms.Resample(sr, 16000)(speech_array)
                if speech_array.shape[0] > 1:
                    speech_array = torch.mean(speech_array, dim=0, keepdim=True)
                
                # Transcribe
                inputs = processor(speech_array.squeeze().numpy(), 
                                 sampling_rate=16000, 
                                 return_tensors="pt", 
                                 padding=True)
                
                with torch.no_grad():
                    logits = model(inputs.input_values).logits
                    
                pred_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(pred_ids)[0]
                batch_predictions.append(transcription)
                
            except Exception as e:
                print(f"Error with {audio_path}: {e}")
                batch_predictions.append("")
        else:
            batch_predictions.append("")
    
    predictions.extend(batch_predictions)
    
    # Save intermediate results every 10 batches
    if (batch_start // BATCH_SIZE + 1) % 10 == 0:
        temp_df = df.iloc[:len(predictions)].copy()
        temp_df["model_prediction_sentence"] = predictions
        temp_output = OUTPUT_TSV.replace(".tsv", f"_temp_{len(predictions)}.tsv")
        temp_df[["client_id", "sentence", "model_prediction_sentence"]].to_csv(temp_output, sep="\t", index=False)
        print(f"Saved intermediate results to {temp_output}")

# Final save
df["model_prediction_sentence"] = predictions
output_df = df[["client_id", "sentence", "model_prediction_sentence"]]
output_df.columns = ["id", "sentence", "model_prediction_sentence"]
output_df.to_csv(OUTPUT_TSV, sep="\t", index=False)
print(f"\nFinal results saved to {OUTPUT_TSV}")
print("\nFirst 5 results:")
print(output_df.head())
