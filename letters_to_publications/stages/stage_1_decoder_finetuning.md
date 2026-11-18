# Stage 1: Fine-Tuning Mistral-7B on Darwin's Correspondence

## Objective
Fine-tune a decoder-only language model (Mistral-7B-v0.1) on Darwin's letter corpus to learn his writing style, vocabulary, and conceptual patterns. This adapted model will serve as the foundation for Stage 2's encoder conversion.

---

## 1. Setup & Environment

### 1.1 Install Dependencies

```python
!pip install -q transformers>=4.36.0 datasets accelerate bitsandbytes peft>=0.7.0 trl sentencepiece torch

# For monitoring and logging
!pip install -q wandb tensorboard

# Import required libraries
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pandas as pd
import numpy as np
from datetime import datetime
import json
```

### 1.2 Authenticate with Hugging Face

```python
# Authenticate with Hugging Face
# IMPORTANT: Before running, visit https://huggingface.co/mistralai/Mistral-7B-v0.1
# and click "Agree and access repository" to request access to the gated model
import os
HF_TOKEN = os.environ["HF_TOKEN"]  # set HF_TOKEN in your environment

# Login to Hugging Face
from huggingface_hub import login
login(token=HF_TOKEN)
print("✓ Successfully authenticated with Hugging Face")
```

### 1.3 Mount Google Drive

```python
# Mount Google Drive to access data and store artifacts
from google.colab import drive
drive.mount('/content/drive')

# Set base path for Darwin project
DARWIN_BASE_PATH = "/content/drive/My Drive/Darwin"
print(f"Darwin project path: {DARWIN_BASE_PATH}")

# Verify data file exists
import os
data_path = f"{DARWIN_BASE_PATH}/letters_date_fix_new.tsv"
if os.path.exists(data_path):
    print(f"✓ Data file found: {data_path}")
else:
    print(f"✗ Data file NOT found: {data_path}")
    print("Please ensure letters_date_fix_new.tsv is in My Drive/Darwin/")
```

### 1.4 GPU Configuration

```python
# Verify A100 availability
!nvidia-smi

# IMPORTANT: Ensure you're using A100 (not T4 or other GPU)
# Go to Runtime → Change runtime type → Hardware accelerator → A100 GPU

# Set memory optimization flags
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Check GPU memory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Verify it's A100
    gpu_name = torch.cuda.get_device_name(0)
    if "A100" not in gpu_name:
        print(f"⚠️  WARNING: Expected A100 but got {gpu_name}")
        print("Please change runtime type to A100 for optimal performance")
else:
    print("❌ ERROR: No GPU detected! Please enable GPU in Runtime settings")
```

---

## 2. Data Preparation

### 2.1 Load Darwin Letters Dataset

```python
# Load the letters dataset from Google Drive
data_path = f"{DARWIN_BASE_PATH}/letters_date_fix_new.tsv"
df = pd.read_csv(data_path, sep='\t', on_bad_lines='skip')

print(f"Total letters: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Display sample
df.head()
```

### 2.2 Data Cleaning & Preprocessing

```python
# Filter valid records (must have 6 columns)
df = df.dropna(subset=['File', 'Year', 'Sender', 'Receiver', 'Letter'])

# Focus on letters written BY Darwin (his writing style)
darwin_letters = df[df['Sender'].str.contains('Darwin, C. R.', na=False)].copy()
print(f"Letters written by Darwin: {len(darwin_letters)}")

# Also include letters TO Darwin for context (optional: experiment with this)
received_letters = df[df['Receiver'].str.contains('Darwin, C. R.', na=False)].copy()
print(f"Letters received by Darwin: {len(received_letters)}")

# Combine: Use Darwin's letters as primary + sample of received for dialogue context
# Sample 30% of received letters to balance the dataset
received_sample = received_letters.sample(frac=0.3, random_state=42)
combined_df = pd.concat([darwin_letters, received_sample]).reset_index(drop=True)
print(f"Combined dataset size: {len(combined_df)}")
```

### 2.3 Format Training Data

```python
def format_letter_for_training(row):
    """
    Format letter into structured prompt for language modeling.
    Include metadata to help model learn contextual patterns.
    """
    # Create structured format
    text = f"""Date: {row['Year']}
From: {row['Sender']}
To: {row['Receiver']}

Letter:
{row['Letter']}"""

    # Optionally include footnote context (contains scholarly annotations)
    if pd.notna(row.get('Footnote')) and len(str(row['Footnote'])) > 10:
        text += f"\n\nContext: {row['Footnote']}"

    return text

# Apply formatting
combined_df['formatted_text'] = combined_df.apply(format_letter_for_training, axis=1)

# Basic statistics
combined_df['text_length'] = combined_df['formatted_text'].str.len()
print(f"Average letter length: {combined_df['text_length'].mean():.0f} characters")
print(f"Median letter length: {combined_df['text_length'].median():.0f} characters")
print(f"Total corpus size: {combined_df['text_length'].sum() / 1e6:.2f}M characters")
```

### 2.4 Train/Validation Split

```python
from sklearn.model_selection import train_test_split

# Temporal split: Use earlier letters for training, later for validation
# This tests generalization to Darwin's later writing style
df_sorted = combined_df.sort_values('Year')

# 95% train, 5% validation (chronological split)
split_idx = int(len(df_sorted) * 0.95)
train_df = df_sorted.iloc[:split_idx]
val_df = df_sorted.iloc[split_idx:]

print(f"Training set: {len(train_df)} letters ({train_df['Year'].min()}-{train_df['Year'].max()})")
print(f"Validation set: {len(val_df)} letters ({val_df['Year'].min()}-{val_df['Year'].max()})")

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df[['formatted_text']])
val_dataset = Dataset.from_pandas(val_df[['formatted_text']])
```

---

## 3. Model Configuration

### 3.1 Load Mistral-7B with Quantization

```python
# Model selection: Mistral-7B-v0.1 (base model, not instruction-tuned)
model_name = "mistralai/Mistral-7B-v0.1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("✓ Tokenizer loaded successfully")

# Load model with 4-bit quantization (fit in A100 with larger batch sizes)
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,  # Reduce CPU memory during loading
    token=HF_TOKEN            # Explicitly pass token to avoid warnings
)

print("✓ Model loaded successfully")

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # Required for gradient checkpointing
```

### 3.2 Configure LoRA for Efficient Fine-Tuning

```python
# LoRA configuration: Low-Rank Adaptation for parameter-efficient fine-tuning
lora_config = LoraConfig(
    r=64,                        # Rank: higher = more expressive but more memory
    lora_alpha=16,               # Scaling factor
    target_modules=[             # Apply LoRA to attention and MLP layers
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Should show ~1-2% of params are trainable
```

---

## 4. Tokenization & Data Collation

### 4.1 Tokenize Dataset

```python
# Set max sequence length (balance between context and memory)
max_seq_length = 1024  # Most letters fit within 1K tokens

def tokenize_function(examples):
    """Tokenize text with truncation and padding"""
    tokenized = tokenizer(
        examples['formatted_text'],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_tensors=None
    )
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Apply tokenization
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing training set"
)

tokenized_val = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=val_dataset.column_names,
    desc="Tokenizing validation set"
)

print(f"Tokenized training samples: {len(tokenized_train)}")
print(f"Tokenized validation samples: {len(tokenized_val)}")
```

### 4.2 Data Collator

```python
# Use DataCollatorForLanguageModeling for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM, not masked LM
)
```

---

## 5. Training Configuration

### 5.1 Training Arguments

```python
# Output directory for checkpoints (save to Google Drive)
output_dir = f"{DARWIN_BASE_PATH}/darwin_mistral_7b_finetuned"
logs_dir = f"{DARWIN_BASE_PATH}/logs"

training_args = TrainingArguments(
    # Directory and logging
    output_dir=output_dir,
    logging_dir=logs_dir,
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,          # Keep only 3 best checkpoints

    # Training hyperparameters
    num_train_epochs=3,           # 3 epochs should be sufficient for this corpus size
    per_device_train_batch_size=4,  # Adjust based on A100 memory (4-8 typical)
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 4*4 = 16
    learning_rate=2e-4,           # Standard for LoRA fine-tuning
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,

    # Optimization
    gradient_checkpointing=True,  # Reduce memory at cost of speed
    optim="paged_adamw_8bit",    # Memory-efficient optimizer
    fp16=False,
    bf16=True,                    # Use bfloat16 for A100

    # Evaluation (use eval_strategy instead of evaluation_strategy for newer transformers)
    eval_strategy="steps",       # Changed from evaluation_strategy
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    # Performance
    dataloader_num_workers=2,
    group_by_length=True,        # Group similar lengths for efficiency

    # Reporting
    report_to="tensorboard",     # Change to "wandb" if using Weights & Biases
)
```

### 5.2 Initialize Trainer

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    tokenizer=tokenizer
)
```

---

## 6. Training Execution

### 6.1 Start Fine-Tuning

```python
# Start training
print("Starting fine-tuning...")
print(f"Training samples: {len(tokenized_train)}")
print(f"Validation samples: {len(tokenized_val)}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Total training steps: {len(tokenized_train) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")

# Train
train_result = trainer.train()

# Save training metrics
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
```

### 6.2 Model Evaluation

```python
# Evaluate on validation set
print("\nEvaluating model...")
eval_metrics = trainer.evaluate()
print(f"Validation Loss: {eval_metrics['eval_loss']:.4f}")
print(f"Validation Perplexity: {np.exp(eval_metrics['eval_loss']):.2f}")

trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)
```

---

## 7. Model Saving & Export

### 7.1 Save Fine-Tuned Model

```python
# Save the fine-tuned LoRA adapters
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\nModel saved to {output_dir}")

# Save training configuration for reproducibility
config_info = {
    "base_model": model_name,
    "dataset_size": len(combined_df),
    "training_samples": len(train_df),
    "validation_samples": len(val_df),
    "lora_config": {
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "target_modules": list(lora_config.target_modules)  # Convert set to list for JSON
    },
    "training_args": {
        "epochs": training_args.num_train_epochs,
        "batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "max_seq_length": max_seq_length
    },
    "final_metrics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                      for k, v in eval_metrics.items()},  # Convert numpy types to native Python
    "trained_on": datetime.now().isoformat()
}

with open(f"{output_dir}/training_config.json", "w") as f:
    json.dump(config_info, f, indent=2)

print(f"✓ Training config saved to {output_dir}/training_config.json")
```

### 7.2 Test Generation (Sanity Check)

```python
# Quick test: Generate text in Darwin's style
def generate_darwin_text(prompt, max_length=200):
    """Generate text in Darwin's style"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Test prompts
test_prompts = [
    "Date: 1859\nFrom: Darwin, C. R.\nTo: Hooker, J. D.\n\nLetter:\nMy dear Hooker,\nI have been thinking about",
    "Date: 1871\nFrom: Darwin, C. R.\nTo: Wallace, A. R.\n\nLetter:\nDear Wallace,\nYour observations on"
]

print("\n" + "="*50)
print("GENERATION TEST (Darwin's Style)")
print("="*50)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n--- Test {i} ---")
    print(f"Prompt:\n{prompt[:100]}...")
    generated = generate_darwin_text(prompt, max_length=150)
    print(f"\nGenerated:\n{generated}")
    print("-" * 50)
```

---

## 8. Resource Management & Cleanup

### 8.1 Monitor GPU Memory

```python
# Check GPU memory usage
if torch.cuda.is_available():
    print(f"\nGPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    print(f"GPU Memory Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB")
```

### 8.2 Clear Memory & Unassign Runtime

```python
# IMPORTANT: Clear GPU memory before disconnecting runtime
import gc

# Delete large objects
del model
del trainer
del tokenized_train
del tokenized_val

# Clear PyTorch cache
torch.cuda.empty_cache()
gc.collect()

print("✓ Memory cleared.")
print(f"✓ Model saved to Google Drive: {output_dir}")
print(f"✓ Logs saved to Google Drive: {logs_dir}")
```

### 8.3 **CRITICAL: Unassign Runtime to Avoid Charges**

```python
# Verify all artifacts are saved to Google Drive
import os

print("\n" + "="*60)
print("RUNTIME UNASSIGNMENT CHECKLIST")
print("="*60)

# Check model saved
model_exists = os.path.exists(output_dir)
print(f"{'✓' if model_exists else '✗'} Model saved: {output_dir}")

# Check logs saved
logs_exist = os.path.exists(logs_dir)
print(f"{'✓' if logs_exist else '✗'} Logs saved: {logs_dir}")

# Check training config saved
config_exists = os.path.exists(f"{output_dir}/training_config.json")
print(f"{'✓' if config_exists else '✗'} Training config saved: {output_dir}/training_config.json")

print("\n" + "="*60)
if model_exists and logs_exist and config_exists:
    print("✓ ALL ARTIFACTS SAVED TO GOOGLE DRIVE")
    print("\nNEXT STEPS:")
    print("1. Go to Runtime → Manage sessions")
    print("2. Terminate this session")
    print("3. Disconnect from runtime")
    print("="*60)
else:
    print("✗ WARNING: Some artifacts may not be saved!")
    print("Please verify before disconnecting.")
    print("="*60)
```

**Manual Steps (REQUIRED):**
1. After running the checklist above, go to **Runtime → Manage sessions** in Colab menu
2. Click **Terminate** on this session
3. Close the notebook or click **Disconnect** to ensure no charges accrue

**Important:** Failing to unassign the runtime will continue to charge your account even if the notebook is idle!

---

## 9. Expected Outcomes & Next Steps

### 9.1 Success Metrics

**Training Metrics:**
- **Final Training Loss**: Expected ~1.5-2.0 (causal LM on specialized corpus)
- **Validation Perplexity**: Expected ~5-8 (lower = better language modeling)
- **Training Time**: ~2-4 hours on A100 (depends on batch size & dataset size)

**Qualitative Assessment:**
- Generated text should reflect Darwin's formal 19th-century writing style
- Should use period-appropriate vocabulary and sentence structure
- Should demonstrate understanding of Darwin's scientific concepts

### 9.2 Troubleshooting

**If OOM (Out of Memory) Errors:**
- Reduce `per_device_train_batch_size` to 2
- Reduce `max_seq_length` to 768
- Reduce LoRA `r` to 32

**If Training is Too Slow:**
- Increase `per_device_train_batch_size` if memory allows
- Reduce `gradient_accumulation_steps`
- Reduce `num_train_epochs` to 2

**If Validation Loss Plateaus:**
- Increase learning rate to 3e-4
- Increase LoRA `r` to 128
- Add more training epochs

### 9.3 Next Steps

✅ **Stage 1 Complete** → Fine-tuned Mistral-7B on Darwin's letters

⏭️ **Stage 2**: Convert fine-tuned decoder to bidirectional encoder using LLM2Vec
- Load this fine-tuned model as base
- Apply LLM2Vec training procedure
- Generate embeddings for letter-publication matching

---

## 10. Cost & Time Estimates

### A100 Runtime Costs (Colab Pro+)

**Estimated Training Time:**
- Dataset loading & preprocessing: ~10 minutes
- Model training (3 epochs): ~2-4 hours
- Evaluation & saving: ~10 minutes
- **Total**: ~3-5 hours

**Cost Estimate:**
- Colab Pro+ A100: ~$1.99/hour
- **Estimated cost**: $6-10 per run

**IMPORTANT**: Remember to disconnect runtime after downloading model to avoid additional charges!

---

## Notes

- This notebook is designed to be converted to Colab using `notedown`
- All file paths assume standard Colab environment (`/content/`)
- Upload `letters_date_fix_new.tsv` to Colab before running
- Monitor training via TensorBoard: `%load_ext tensorboard` → `%tensorboard --logdir logs`
