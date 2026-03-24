import torch
import librosa
from datasets import load_dataset
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# ✅ Enable better GPU performance
torch.backends.cuda.matmul.allow_tf32 = True

# 1. Load Data
dataset = load_dataset("csv", data_files="train.csv")["train"]

model_id = "openai/whisper-small"

processor = WhisperProcessor.from_pretrained(
    model_id,
    language="malayalam",
    task="transcribe"
)

def prepare_dataset(batch):
    try:
        audio_path = batch["file_name"]
        audio_array, _ = librosa.load(audio_path, sr=16000)

        batch["input_features"] = processor.feature_extractor(
            audio_array,
            sampling_rate=16000
        ).input_features[0]

        batch["labels"] = processor.tokenizer(
            batch["transcription"]
        ).input_ids

        return batch

    except Exception as e:
        print(f"Skipping {batch.get('file_name')} due to error: {e}")
        return None

print("Preprocessing audio...")
dataset = dataset.map(prepare_dataset)
dataset = dataset.filter(lambda x: x is not None)
dataset = dataset.remove_columns(["file_name", "transcription"])

# 2. Load Model (NO 4-bit)
print("Loading model...")

model = WhisperForConditionalGeneration.from_pretrained(
    model_id
).to("cuda")

# ✅ LoRA Configuration (FIXED)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, lora_config)

# 3. Training Args (optimized for 6GB VRAM)
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-ml-checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,   # ✅ FIXED
    max_steps=400,
    fp16=True,
    optim="adamw_torch",
    logging_steps=10,
    remove_unused_columns=False,
    label_names=["labels"],
    save_strategy="no",
)

# 4. Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch

# 5. Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset,
    data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
    tokenizer=processor.feature_extractor,
)

model.config.use_cache = False

print("Starting training...")
trainer.train()

# 6. Save model
model.save_pretrained("./whisper-malayalam-finetuned")
processor.save_pretrained("./whisper-malayalam-finetuned")

print("Done! Model saved.")