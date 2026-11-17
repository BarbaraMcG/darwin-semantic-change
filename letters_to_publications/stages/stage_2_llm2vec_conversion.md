# Stage 2: Converting Fine-Tuned Decoder to Encoder with LLM2Vec

## Objective
Convert the fine-tuned Mistral-7B decoder model from Stage 1 into a bidirectional encoder using LLM2Vec methodology. This encoder will produce high-quality embeddings for Darwin's letters and publications, enabling fine-grained semantic matching in Stage 3.

---

## Background: What is LLM2Vec?

LLM2Vec transforms decoder-only language models (like Mistral, Llama) into effective text encoders through:

1. **Bidirectional Attention**: Modify attention masks to allow tokens to attend to future tokens
2. **Masked Next Token Prediction (MNTP)**: Train with masked prediction objective (optional)
3. **Contrastive Learning**: Use SimCSE-style contrastive loss on paired examples (optional)

**Why This Matters for Darwin Project:**
- Decoder models generate text sequentially (left-to-right)
- Encoder models understand text bidirectionally (better for semantic similarity)
- We need bidirectional understanding to match letter passages to publication passages

**Two Approaches for This Stage:**

**Option A: Simplified (Recommended for this notebook)**
- ✓ Use LLM2Vec wrapper for bidirectional attention (automatic)
- ✓ Use Stage 1 fine-tuning (already done - domain adaptation)
- ✓ Skip additional MNTP/contrastive training
- ⏱️ **Time**: ~30 minutes
- 📊 **Quality**: Good (benefits from Stage 1 + bidirectional attention)

**Option B: Full LLM2Vec Training (Advanced)**
- Requires separate training scripts (`run_mntp.py`, `run_simcse.py`)
- Requires preparing JSON config files
- Requires additional contrastive training data
- ⏱️ **Time**: ~4-6 hours
- 📊 **Quality**: Better (full LLM2Vec methodology)

**This notebook uses Option A** for simplicity and speed. If you need Option B, refer to the LLM2Vec repository: https://github.com/McGill-NLP/llm2vec

---

## 1. Setup & Environment

### 1.1 Install Dependencies

```python
# Install LLM2Vec and dependencies
!pip install -q llm2vec transformers>=4.36.0 datasets accelerate bitsandbytes peft>=0.7.0 torch
!pip install -q sentence-transformers scikit-learn numpy pandas

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

# Check LLM2Vec installation
try:
    from llm2vec import LLM2Vec
    print("✓ LLM2Vec installed successfully")
except ImportError:
    print("✗ LLM2Vec not found - installing from source")
    !pip install git+https://github.com/McGill-NLP/llm2vec.git
    from llm2vec import LLM2Vec
    print("✓ LLM2Vec installed from GitHub")
```

### 1.2 Authenticate with Hugging Face

```python
# Authenticate with Hugging Face
import os
HF_TOKEN = os.environ["HF_TOKEN"]  # set HF_TOKEN in your environment

from huggingface_hub import login
login(token=HF_TOKEN)
print("✓ Successfully authenticated with Hugging Face")
```

### 1.3 Mount Google Drive

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set paths
DARWIN_BASE_PATH = "/content/drive/My Drive/Darwin"
STAGE1_MODEL_PATH = f"{DARWIN_BASE_PATH}/darwin_mistral_7b_finetuned"
STAGE2_OUTPUT_PATH = f"{DARWIN_BASE_PATH}/darwin_mistral_7b_encoder"

print(f"Stage 1 model: {STAGE1_MODEL_PATH}")
print(f"Stage 2 output: {STAGE2_OUTPUT_PATH}")

# Verify Stage 1 model exists
if os.path.exists(STAGE1_MODEL_PATH):
    print("✓ Stage 1 model found")
else:
    print("✗ Stage 1 model NOT found - please run Stage 1 first")
```

### 1.4 GPU Configuration

```python
# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Verify A100
    gpu_name = torch.cuda.get_device_name(0)
    if "A100" not in gpu_name:
        print(f"⚠️  WARNING: Expected A100 but got {gpu_name}")
else:
    print("❌ ERROR: No GPU detected")
```

---

## 2. Load Fine-Tuned Model from Stage 1

### 2.1 Load Model with LoRA Adapters

```python
print("Loading Stage 1 fine-tuned model...")

# Base model name
base_model_name = "mistralai/Mistral-7B-v0.1"

# Load tokenizer from base model (Stage 1 tokenizer may be corrupted)
print("Loading tokenizer from base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("✓ Tokenizer loaded from base model")

# IMPORTANT: LLM2Vec works best with non-quantized models
# Load in full precision (bfloat16) instead of 4-bit
print("Loading base model in bfloat16 (no quantization)...")
print("⚠️  This requires ~14GB GPU memory for Mistral-7B")

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
    low_cpu_mem_usage=True
)
print("✓ Base model loaded")

# Load LoRA adapters from Stage 1
print("Loading LoRA adapters from Stage 1...")
model = PeftModel.from_pretrained(base_model, STAGE1_MODEL_PATH)
print("✓ Stage 1 LoRA adapters loaded")

# Merge adapters into base model for LLM2Vec
print("Merging LoRA adapters into base model...")
model = model.merge_and_unload()
print("✓ Model merged - ready for LLM2Vec conversion")
```

---

## 3. Initialize LLM2Vec

### 3.1 Save Merged Model & Initialize LLM2Vec

```python
print("Saving merged model temporarily for LLM2Vec...")

# Save merged model to temporary directory
temp_model_path = "/content/temp_merged_model"
model.save_pretrained(temp_model_path)
tokenizer.save_pretrained(temp_model_path)
print(f"✓ Merged model saved to {temp_model_path}")

# CRITICAL: Free GPU memory before loading LLM2Vec
print("\nClearing GPU memory...")
del model
del base_model
import gc
gc.collect()
torch.cuda.empty_cache()
print("✓ GPU memory cleared")

# Check available memory
if torch.cuda.is_available():
    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    print(f"Available GPU memory: {free_memory / 1e9:.2f} GB")

# Initialize LLM2Vec from the saved merged model
print("\nInitializing LLM2Vec from merged model...")

l2v = LLM2Vec.from_pretrained(
    base_model_name_or_path=temp_model_path,
    peft_model_name_or_path=None,  # Already merged, no separate LoRA
    pooling_mode="mean",            # Mean pooling for sentence embeddings
    max_length=512                  # Match Stage 1 sequence length
)

# Move to GPU if available
if torch.cuda.is_available():
    l2v = l2v.to("cuda")
    print(f"✓ Model moved to GPU")

print("✓ LLM2Vec initialized")
print(f"Pooling mode: mean")
print(f"Max length: 512 tokens")
```

---

## 4. Save LLM2Vec Encoder

**Note:** LLM2Vec is a wrapper class - it doesn't have `.save_pretrained()` directly. We save the underlying model with a workaround.

### 4.1 Save Model & Configuration

```python
print("\n" + "="*60)
print("LLM2VEC CONFIGURATION")
print("="*60)
print("✓ Base model: Fine-tuned Mistral-7B (Stage 1)")
print("✓ Bidirectional attention: Enabled by LLM2Vec")
print("✓ Pooling mode: Mean pooling")
print("✓ Max sequence length: 512 tokens")
print("✓ Ready for embedding generation")
print("="*60)

# Save the underlying model (LLM2Vec wrapper doesn't have save_pretrained)
output_path = STAGE2_OUTPUT_PATH
print(f"\nSaving encoder model to {output_path}...")

# Create output directory if it doesn't exist
import os
os.makedirs(output_path, exist_ok=True)

# Save the underlying model and tokenizer
# Workaround for saving after LLM2Vec initialization
l2v.model._hf_peft_config_loaded = False  # Workaround for PEFT save issue
l2v.model.save_pretrained(output_path)
l2v.tokenizer.save_pretrained(output_path)

# Save LLM2Vec configuration
import json
llm2vec_config = {
    "pooling_mode": "mean",
    "max_length": 512,
    "model_type": "llm2vec",
    "base_model": "mistralai/Mistral-7B-v0.1",
    "stage1_model": STAGE1_MODEL_PATH
}
with open(f"{output_path}/llm2vec_config.json", 'w') as f:
    json.dump(llm2vec_config, f, indent=2)

print(f"✓ Encoder saved to {output_path}")
print(f"✓ LLM2Vec config saved")
```

---

## 5. Evaluate Encoder Quality

### 5.1 Use Current Encoder

```python
# Use the current LLM2Vec model for evaluation
encoder = l2v
print("✓ Using current LLM2Vec encoder for evaluation")
```

### 5.2 Generate Test Embeddings

```python
# Test sentences
test_texts = [
    "I have been observing the variation in species across different islands.",
    "The distribution of species on the Galapagos shows remarkable patterns.",
    "Natural selection acts on heritable variation within populations.",
    "My dear Hooker, thank you for your letter regarding the orchids.",
    "The cirripedes show fascinating anatomical structures."
]

print("Generating embeddings for test sentences...")
embeddings = encoder.encode(test_texts)

print(f"\nEmbedding shape: {embeddings.shape}")
print(f"Embedding dimension: {embeddings.shape[1]}")
```

### 5.3 Similarity Analysis

```python
from sklearn.metrics.pairwise import cosine_similarity

# Compute similarity matrix
similarity_matrix = cosine_similarity(embeddings)

print("\n" + "="*60)
print("SIMILARITY MATRIX")
print("="*60)
print("(0=Evolution/Variation, 1=Galapagos, 2=Selection, 3=Letter, 4=Barnacles)")
print()
print(np.round(similarity_matrix, 3))
print("="*60)

print("\nExpected patterns:")
print("  • Sentences 0, 1, 2 (scientific) should be similar")
print("  • Sentence 3 (letter greeting) should be different")
print("  • Sentence 4 (barnacles) should be moderately similar")
```

### 5.4 Dimensionality Reduction Visualization

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce to 2D for visualization
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100)

for i, txt in enumerate(test_texts):
    plt.annotate(f"{i}: {txt[:30]}...", (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                 fontsize=8, alpha=0.7)

plt.title("Darwin Embeddings - 2D Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save to Google Drive
plt.savefig(f"{DARWIN_BASE_PATH}/stage2_embeddings_visualization.png", dpi=150)
print(f"\n✓ Visualization saved to {DARWIN_BASE_PATH}/stage2_embeddings_visualization.png")
plt.show()
```

---

## 6. Benchmark Against Base Embeddings

### 6.1 Load Base Model Embeddings (No Fine-Tuning)

```python
print("Loading base model for comparison...")
print("(This may take a few minutes)")

# CRITICAL: Free GPU memory first
print("\nClearing fine-tuned encoder from GPU...")
del l2v
del encoder
import gc
gc.collect()
torch.cuda.empty_cache()

# Check available memory
free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
print(f"Available GPU memory: {free_memory / 1e9:.2f} GB")

# Load base Mistral with LLM2Vec but no Stage 1 fine-tuning
base_encoder = LLM2Vec.from_pretrained(
    base_model_name_or_path=base_model_name,
    peft_model_name_or_path=None,
    pooling_mode="mean",
    max_length=512,
    token=HF_TOKEN
)

if torch.cuda.is_available():
    base_encoder = base_encoder.to("cuda")

print("✓ Base model loaded")

# Generate embeddings with base model
base_embeddings = base_encoder.encode(test_texts)
base_similarity = cosine_similarity(base_embeddings)

print("✓ Base model embeddings generated")
```

### 6.2 Compare Similarity Patterns

```python
print("\n" + "="*60)
print("COMPARISON: BASE vs FINE-TUNED ENCODER")
print("="*60)

print("\nBASE MODEL SIMILARITY:")
print(np.round(base_similarity, 3))

print("\nFINE-TUNED LLM2VEC SIMILARITY:")
print(np.round(similarity_matrix, 3))

print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)
print("Fine-tuned encoder should show:")
print("  • Higher similarity for Darwin-specific scientific concepts")
print("  • Better discrimination between letter greetings and content")
print("  • Stronger clustering of related topics")
print("="*60)
```

---

## 7. Generate Embeddings at Scale

### 7.1 Load Darwin Letters Dataset

```python
# Load letters dataset
data_path = f"{DARWIN_BASE_PATH}/letters_date_fix_new.tsv"
df = pd.read_csv(data_path, sep='\t', on_bad_lines='skip')

# Filter Darwin's letters
darwin_letters = df[df['Sender'].str.contains('Darwin, C. R.', na=False)].copy()
print(f"Total Darwin letters: {len(darwin_letters)}")
```

### 7.2 Embed All Darwin Letters

```python
print("Generating embeddings for all Darwin letters...")
print("(This may take 15-30 minutes)")

# Get all letter texts
all_letters = darwin_letters['Letter'].tolist()
all_letters_clean = [str(text)[:1000] for text in all_letters if pd.notna(text) and len(str(text)) > 10]

print(f"Total letters to embed: {len(all_letters_clean)}")

# Generate embeddings in batches
batch_size = 32
all_embeddings = []

for i in range(0, len(all_letters_clean), batch_size):
    batch = all_letters_clean[i:i+batch_size]
    batch_embeddings = encoder.encode(batch, show_progress_bar=False)
    all_embeddings.append(batch_embeddings)

    if (i // batch_size) % 10 == 0:
        print(f"Processed {i}/{len(all_letters_clean)} letters...")

all_embeddings = np.vstack(all_embeddings)
print(f"\n✓ Generated {all_embeddings.shape[0]} embeddings")
print(f"Embedding shape: {all_embeddings.shape}")
```

### 7.3 Save Letter Embeddings

```python
# Save embeddings for Stage 3
embeddings_path = f"{DARWIN_BASE_PATH}/darwin_letters_embeddings.npy"
np.save(embeddings_path, all_embeddings)
print(f"✓ Letter embeddings saved to {embeddings_path}")

# Save metadata (letter IDs, years, correspondents)
embedding_metadata = darwin_letters[['File', 'Year', 'Sender', 'Receiver']].reset_index(drop=True)
metadata_df_path = f"{DARWIN_BASE_PATH}/darwin_letters_embeddings_metadata.csv"
embedding_metadata.to_csv(metadata_df_path, index=False)
print(f"✓ Embedding metadata saved to {metadata_df_path}")

# Summary statistics
print("\n" + "="*60)
print("EMBEDDING STATISTICS")
print("="*60)
print(f"Total embeddings: {all_embeddings.shape[0]}")
print(f"Embedding dimension: {all_embeddings.shape[1]}")
print(f"Mean norm: {np.linalg.norm(all_embeddings, axis=1).mean():.4f}")
print(f"Storage size: {all_embeddings.nbytes / 1e6:.2f} MB")
print("="*60)
```

---

## 8. Cleanup & Runtime Management

### 8.1 Clear GPU Memory

```python
# Clear memory
import gc

del model
del base_model
del encoder
if 'base_encoder' in locals():
    del base_encoder

torch.cuda.empty_cache()
gc.collect()

print("✓ GPU memory cleared")
```

### 8.2 Verify All Artifacts Saved

```python
print("\n" + "="*60)
print("RUNTIME UNASSIGNMENT CHECKLIST")
print("="*60)

# Check all files saved
checks = {
    "Encoder model": os.path.exists(STAGE2_OUTPUT_PATH),
    "LLM2Vec config": os.path.exists(f"{STAGE2_OUTPUT_PATH}/llm2vec_config.json"),
    "Letter embeddings": os.path.exists(embeddings_path),
    "Embedding metadata": os.path.exists(metadata_df_path),
    "Visualization": os.path.exists(f"{DARWIN_BASE_PATH}/stage2_embeddings_visualization.png")
}

for item, exists in checks.items():
    status = "✓" if exists else "✗"
    print(f"{status} {item}")

print("\n" + "="*60)
if all(checks.values()):
    print("✓ ALL ARTIFACTS SAVED TO GOOGLE DRIVE")
    print("\nNEXT STEPS:")
    print("1. Go to Runtime → Manage sessions")
    print("2. Terminate this session")
    print("3. Disconnect from runtime")
else:
    print("✗ WARNING: Some artifacts may not be saved!")
    print("Please verify before disconnecting.")
print("="*60)
```

---

## 9. Summary & Next Steps

### 9.1 Stage 2 Summary

```python
print("\n" + "="*60)
print("STAGE 2 COMPLETE - SUMMARY")
print("="*60)
print(f"✓ Fine-tuned decoder converted to bidirectional encoder")
print(f"✓ LLM2Vec initialized with mean pooling")
print(f"✓ Generated embeddings for {all_embeddings.shape[0]} Darwin letters")
print(f"✓ Embedding dimension: {all_embeddings.shape[1]}")
print(f"✓ All artifacts saved to Google Drive")
print("\n" + "="*60)
print("WHAT WE ACHIEVED:")
print("="*60)
print("• Stage 1 fine-tuned model → Bidirectional encoder via LLM2Vec")
print("• Bidirectional attention enabled automatically")
print("• High-quality semantic embeddings for all Darwin letters")
print("• Ready for fine-grained letter-publication matching")
print("\n" + "="*60)
print("APPROACH:")
print("="*60)
print("• Used simplified LLM2Vec (Option A)")
print("• Benefits: Stage 1 domain adaptation + bidirectional attention")
print("• Skipped MNTP/contrastive training (can be added later if needed)")
print("\n" + "="*60)
print("NEXT STEPS:")
print("="*60)
print("⏭️  STAGE 3: Letter-Publication Matching")
print("   - Download/embed Darwin's publications")
print("   - Implement chunking strategy (paragraph-based)")
print("   - Match publication chunks to relevant letters")
print("   - Evaluate against ground truth timeline")
print("="*60)
```

---

## Notes

- **Approach**: Simplified LLM2Vec (Option A) - bidirectional attention without additional training
- **Processing time**: Approximately 30-60 minutes on A100
- **Memory usage**: ~28GB GPU memory (for full precision model)
- **LLM2Vec benefits**: Bidirectional attention + Stage 1 domain adaptation
- **Stage 2 output**: Encoder model + embeddings for all Darwin letters (~9,600 letters)
- **Critical for Stage 3**: These embeddings enable semantic matching at scale
- **Future improvement**: Can add MNTP/contrastive training using run_mntp.py and run_simcse.py scripts

---

## Troubleshooting

### If OOM During Model Loading:
- The merged model requires ~14GB GPU memory
- Ensure previous models are deleted before loading LLM2Vec
- Check GPU memory with `torch.cuda.memory_allocated()`

### If LLM2Vec Installation Fails:
- Install from source: `!pip install git+https://github.com/McGill-NLP/llm2vec.git`
- Check transformers version compatibility (needs >=4.36.0)

### If Embedding Generation is Too Slow:
- Reduce batch_size from 32 to 16
- Use smaller max_length (256 instead of 512)
- Process fewer letters for testing first

### If Save Fails:
- Use the workaround: `model._hf_peft_config_loaded = False`
- Alternatively, skip saving and use model directly for embedding generation

### For Full LLM2Vec Training:
- Refer to: https://github.com/McGill-NLP/llm2vec
- Use run_mntp.py for MNTP training
- Use run_simcse.py for contrastive learning
- Requires JSON config files and additional 4-6 hours
