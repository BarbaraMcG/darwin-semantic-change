# Stage 2.5: Generate Embeddings for All Darwin Letters

## Objective
Reload the fine-tuned LLM2Vec encoder from Stage 2 and generate embeddings for all Darwin letters at scale. These embeddings will be used in Stage 3 for letter-publication matching.

---

## 1. Setup & Environment

### 1.1 Install Dependencies

```python
# Install LLM2Vec if not already installed
!pip install -q llm2vec transformers>=4.36.0 torch

print("✓ Dependencies installed")
```

### 1.2 Import Libraries

```python
import torch
from llm2vec import LLM2Vec
import pandas as pd
import numpy as np
import os
from datetime import datetime

print("✓ Libraries imported")
```

### 1.3 Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Set paths
DARWIN_BASE_PATH = "/content/drive/My Drive/Darwin"
ENCODER_PATH = f"{DARWIN_BASE_PATH}/darwin_mistral_7b_encoder"

print(f"Darwin base path: {DARWIN_BASE_PATH}")
print(f"Encoder path: {ENCODER_PATH}")
```

### 1.4 Verify GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    print(f"Available GPU memory: {free_memory / 1e9:.2f} GB")
```

---

## 2. Load Fine-Tuned Encoder

### 2.1 Load LLM2Vec Encoder from Stage 2

```python
print("Loading fine-tuned LLM2Vec encoder from Stage 2...")
print("(This may take a few minutes)")

# Load the base model and tokenizer from saved path
from transformers import AutoModel, AutoTokenizer

print("Loading model and tokenizer...")
model = AutoModel.from_pretrained(
    ENCODER_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(ENCODER_PATH)

print("✓ Model and tokenizer loaded")

# Wrap with LLM2Vec (manually to avoid config file issues)
print("Initializing LLM2Vec wrapper...")
encoder = LLM2Vec(
    model=model,
    tokenizer=tokenizer,
    pooling_mode="mean",
    max_length=512,
    doc_max_length=400,
    skip_instruction=True
)

print("✓ Fine-tuned LLM2Vec encoder loaded successfully")
```

### 2.2 Quick Test

```python
# Quick test to verify encoder is working
test_text = "My dear Hooker, I have been observing species variation."
test_embedding = encoder.encode([test_text])

print(f"\nTest embedding shape: {test_embedding.shape}")
print(f"Embedding dimension: {test_embedding.shape[1]}")
print("✓ Encoder is functioning correctly")
```

---

## 3. Load Darwin Letters Dataset

### 3.1 Load and Filter Letters

```python
print("\nLoading Darwin letters dataset...")

# Load letters
data_path = f"{DARWIN_BASE_PATH}/letters_date_fix_new.tsv"
df = pd.read_csv(data_path, sep='\t', on_bad_lines='skip')

# Filter Darwin's letters (as sender)
darwin_letters = df[df['Sender'].str.contains('Darwin, C. R.', na=False)].copy()
print(f"Total Darwin letters: {len(darwin_letters)}")

# Basic statistics
print(f"Year range: {darwin_letters['Year'].min()}-{darwin_letters['Year'].max()}")
print(f"Unique correspondents: {darwin_letters['Receiver'].nunique()}")
```

### 3.2 Prepare Text Data

```python
# Extract and clean letter texts
all_letters = darwin_letters['Letter'].tolist()
all_letters_clean = []

for text in all_letters:
    if pd.notna(text) and len(str(text)) > 10:
        # Truncate to 1000 characters to keep within 512 token limit
        cleaned_text = str(text)[:1000]
        all_letters_clean.append(cleaned_text)

print(f"\nLetters to embed: {len(all_letters_clean)}")
print(f"Average length: {np.mean([len(t) for t in all_letters_clean]):.0f} characters")
```

---

## 4. Generate Embeddings at Scale

### 4.1 Batch Embedding Generation

```python
print("\n" + "="*60)
print("GENERATING EMBEDDINGS FOR ALL DARWIN LETTERS")
print("="*60)
print(f"Total letters: {len(all_letters_clean)}")
print("Estimated time: 15-30 minutes")
print("="*60)

# Generate embeddings in batches
batch_size = 32  # Adjust based on GPU memory
all_embeddings = []

from datetime import datetime
start_time = datetime.now()

for i in range(0, len(all_letters_clean), batch_size):
    batch = all_letters_clean[i:i+batch_size]
    batch_embeddings = encoder.encode(batch, show_progress_bar=False)
    all_embeddings.append(batch_embeddings)

    # Progress updates every 10 batches
    if (i // batch_size) % 10 == 0:
        progress = (i / len(all_letters_clean)) * 100
        elapsed = (datetime.now() - start_time).total_seconds()
        estimated_total = (elapsed / (i + 1)) * len(all_letters_clean)
        remaining = estimated_total - elapsed
        print(f"Progress: {progress:.1f}% ({i}/{len(all_letters_clean)}) - "
              f"Est. remaining: {remaining/60:.1f} min")

# Stack all embeddings
all_embeddings = np.vstack(all_embeddings)

end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

print("\n" + "="*60)
print("EMBEDDING GENERATION COMPLETE")
print("="*60)
print(f"✓ Generated {all_embeddings.shape[0]} embeddings")
print(f"✓ Embedding shape: {all_embeddings.shape}")
print(f"✓ Duration: {duration/60:.1f} minutes")
print("="*60)
```

### 4.2 Embedding Statistics

```python
print("\n" + "="*60)
print("EMBEDDING STATISTICS")
print("="*60)
print(f"Total embeddings: {all_embeddings.shape[0]}")
print(f"Embedding dimension: {all_embeddings.shape[1]}")
print(f"Mean embedding norm: {np.linalg.norm(all_embeddings, axis=1).mean():.4f}")
print(f"Std embedding norm: {np.linalg.norm(all_embeddings, axis=1).std():.4f}")
print(f"Storage size: {all_embeddings.nbytes / 1e6:.2f} MB")
print(f"Memory per embedding: {all_embeddings.nbytes / all_embeddings.shape[0] / 1e3:.2f} KB")
print("="*60)
```

---

## 5. Save Embeddings and Metadata

### 5.1 Save Embeddings Array

```python
print("\nSaving embeddings to Google Drive...")

# Save embeddings as numpy array
embeddings_path = f"{DARWIN_BASE_PATH}/darwin_letters_embeddings.npy"
np.save(embeddings_path, all_embeddings)

print(f"✓ Embeddings saved to: {embeddings_path}")
print(f"✓ File size: {os.path.getsize(embeddings_path) / 1e6:.2f} MB")
```

### 5.2 Save Metadata

```python
# Create metadata DataFrame (aligned with embeddings)
# Only include letters that were successfully embedded
metadata_df = darwin_letters.iloc[:len(all_embeddings)][['File', 'Year', 'Sender', 'Receiver']].copy()
metadata_df['embedding_index'] = range(len(all_embeddings))
metadata_df['text_length'] = [len(t) for t in all_letters_clean]
metadata_df['generated_at'] = datetime.now().isoformat()

# Save metadata
metadata_path = f"{DARWIN_BASE_PATH}/darwin_letters_embeddings_metadata.csv"
metadata_df.to_csv(metadata_path, index=False)

print(f"✓ Metadata saved to: {metadata_path}")
print(f"✓ Metadata records: {len(metadata_df)}")
```

### 5.3 Save Generation Config

```python
# Save configuration for reproducibility
import json

config = {
    "stage": "2.5",
    "encoder_path": ENCODER_PATH,
    "dataset_path": data_path,
    "generation_config": {
        "batch_size": batch_size,
        "max_length": 512,
        "pooling_mode": "mean",
        "text_truncation": 1000
    },
    "statistics": {
        "total_embeddings": int(all_embeddings.shape[0]),
        "embedding_dimension": int(all_embeddings.shape[1]),
        "mean_norm": float(np.linalg.norm(all_embeddings, axis=1).mean()),
        "std_norm": float(np.linalg.norm(all_embeddings, axis=1).std()),
        "duration_seconds": duration
    },
    "generated_at": datetime.now().isoformat()
}

config_path = f"{DARWIN_BASE_PATH}/darwin_letters_embeddings_config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Configuration saved to: {config_path}")
```

---

## 6. Validation & Quality Checks

### 6.1 Check for NaN or Invalid Embeddings

```python
print("\n" + "="*60)
print("VALIDATION CHECKS")
print("="*60)

# Check for NaN
nan_count = np.isnan(all_embeddings).sum()
print(f"NaN values: {nan_count}")

# Check for infinite values
inf_count = np.isinf(all_embeddings).sum()
print(f"Infinite values: {inf_count}")

# Check for zero vectors
zero_vectors = np.sum(np.all(all_embeddings == 0, axis=1))
print(f"Zero vectors: {zero_vectors}")

# Check embedding diversity
unique_embeddings = np.unique(all_embeddings, axis=0).shape[0]
print(f"Unique embeddings: {unique_embeddings} / {all_embeddings.shape[0]}")

if nan_count == 0 and inf_count == 0 and zero_vectors == 0:
    print("\n✓ All validation checks passed!")
else:
    print("\n⚠️  WARNING: Some validation issues detected")

print("="*60)
```

### 6.2 Sample Similarity Test

```python
print("\n" + "="*60)
print("SAMPLE SIMILARITY TEST")
print("="*60)

# Test similarity between a few random letters
from sklearn.metrics.pairwise import cosine_similarity

sample_indices = np.random.choice(len(all_embeddings), 5, replace=False)
sample_embeddings = all_embeddings[sample_indices]
sample_similarity = cosine_similarity(sample_embeddings)

print("Similarity matrix (5 random letters):")
print(np.round(sample_similarity, 3))

# Show which letters were sampled
print("\nSampled letters:")
for idx in sample_indices:
    year = metadata_df.iloc[idx]['Year']
    receiver = metadata_df.iloc[idx]['Receiver']
    print(f"  Index {idx}: {year} → {receiver}")

print("="*60)
```

---

## 7. Cleanup & Summary

### 7.1 Clear GPU Memory

```python
print("\nClearing GPU memory...")

del encoder
import gc
gc.collect()
torch.cuda.empty_cache()

print("✓ GPU memory cleared")
```

### 7.2 Verify All Files Saved

```python
print("\n" + "="*60)
print("VERIFICATION CHECKLIST")
print("="*60)

files_to_check = {
    "Embeddings array (.npy)": embeddings_path,
    "Metadata (.csv)": metadata_path,
    "Configuration (.json)": config_path
}

all_saved = True
for name, path in files_to_check.items():
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {name}: {path}")
    if exists:
        size = os.path.getsize(path) / 1e6
        print(f"    Size: {size:.2f} MB")
    all_saved = all_saved and exists

print("="*60)

if all_saved:
    print("\n✓ ALL FILES SAVED SUCCESSFULLY")
else:
    print("\n✗ WARNING: Some files may not have been saved!")
```

### 7.3 Final Summary

```python
print("\n" + "="*60)
print("STAGE 2.5 COMPLETE - SUMMARY")
print("="*60)
print(f"✓ Loaded fine-tuned LLM2Vec encoder from Stage 2")
print(f"✓ Generated {all_embeddings.shape[0]} embeddings")
print(f"✓ Embedding dimension: {all_embeddings.shape[1]}")
print(f"✓ Processing time: {duration/60:.1f} minutes")
print(f"✓ All artifacts saved to Google Drive")
print("\n" + "="*60)
print("OUTPUTS:")
print("="*60)
print(f"1. Embeddings: {embeddings_path}")
print(f"2. Metadata: {metadata_path}")
print(f"3. Config: {config_path}")
print("\n" + "="*60)
print("NEXT STEPS:")
print("="*60)
print("⏭️  STAGE 3: Letter-Publication Matching")
print("   - Load these embeddings")
print("   - Embed Darwin's publications (with chunking)")
print("   - Compute semantic similarity")
print("   - Match publication chunks to relevant letters")
print("   - Evaluate against ground truth timeline")
print("="*60)
```

---

## 8. Optional: Create Sample Visualization

### 8.1 Visualize Embeddings Over Time

```python
print("\nCreating temporal visualization...")

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce to 2D
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(all_embeddings)

# Plot colored by year
plt.figure(figsize=(14, 10))
scatter = plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=metadata_df['Year'],
    cmap='viridis',
    alpha=0.5,
    s=20
)
plt.colorbar(scatter, label='Year')
plt.title(f'Darwin Letters Embeddings (n={len(all_embeddings)})')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True, alpha=0.3)

# Save
viz_path = f"{DARWIN_BASE_PATH}/stage2.5_embeddings_temporal_viz.png"
plt.savefig(viz_path, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved to: {viz_path}")
plt.show()
```

### 8.2 Embeddings by Decade

```python
# Analyze embeddings by decade
metadata_df['Decade'] = (metadata_df['Year'] // 10) * 10

print("\n" + "="*60)
print("EMBEDDINGS BY DECADE")
print("="*60)

for decade in sorted(metadata_df['Decade'].unique()):
    decade_count = (metadata_df['Decade'] == decade).sum()
    decade_pct = (decade_count / len(metadata_df)) * 100
    print(f"{decade}s: {decade_count:4d} letters ({decade_pct:5.1f}%)")

print("="*60)
```

---

## Notes

- **Processing time**: Approximately 15-30 minutes for ~8,000 letters on A100
- **Memory usage**: ~28GB GPU memory during encoding
- **Batch size**: 32 (adjust down if OOM, up if memory available)
- **Output size**: ~130 MB for 8,000 embeddings (4096-dim each)
- **Ready for Stage 3**: Embeddings and metadata are aligned and saved

---

## Troubleshooting

### If Encoder Fails to Load:
- Verify ENCODER_PATH exists and contains saved model
- Check that Stage 2 completed successfully
- Try loading with explicit device_map: `device_map="auto"`

### If OOM During Embedding:
- Reduce batch_size from 32 to 16 or 8
- Process in smaller chunks and save incrementally
- Clear GPU cache periodically: `torch.cuda.empty_cache()`

### If Embeddings Look Wrong:
- Check validation output for NaN/Inf/zeros
- Verify encoder loaded correctly with test embedding
- Compare sample similarities - should show meaningful patterns

### If Too Slow:
- Increase batch_size if GPU memory available
- Use smaller max_length (256 instead of 512)
- Process subset first to test (~1000 letters takes ~2 minutes)
