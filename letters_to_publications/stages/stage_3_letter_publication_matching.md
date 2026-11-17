# Stage 3: Letter-Publication Matching

## Objective
Implement fine-grained semantic matching between Darwin's letters (from Stage 2.5) and publication passages (chunked text) to identify which specific letter passages influenced or relate to which publication sections. Uses the strategy documented in `stage_3_strategy.md`.

---

## Background

We now have:
- ✅ **8,159 Darwin letter embeddings** from Stage 2.5 (4096-dim)
- ✅ **12 Darwin publications** as plain text (~10.7 MB)
- ✅ **Ground truth timeline** showing when Darwin worked on each publication

**Goal**: Match publication chunks → relevant letters with temporal filtering.

---

## 1. Setup & Environment

### 1.1 Install Dependencies

```python
!pip install -q llm2vec transformers torch numpy pandas scikit-learn faiss-cpu

print("✓ Dependencies installed")
```

### 1.2 Import Libraries

```python
import torch
from llm2vec import LLM2Vec
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import os
import json
import re
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import faiss

print("✓ Libraries imported")
```

### 1.3 Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Set paths
DARWIN_BASE_PATH = "/content/drive/My Drive/Darwin"
ENCODER_PATH = f"{DARWIN_BASE_PATH}/darwin_mistral_7b_encoder"
PUBLICATIONS_PATH = f"{DARWIN_BASE_PATH}/Publications"
EMBEDDINGS_PATH = f"{DARWIN_BASE_PATH}/darwin_letters_embeddings.npy"
METADATA_PATH = f"{DARWIN_BASE_PATH}/darwin_letters_embeddings_metadata.csv"

print(f"Darwin base: {DARWIN_BASE_PATH}")
print(f"Publications: {PUBLICATIONS_PATH}")
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

## 2. Load Encoder and Letter Embeddings

### 2.1 Load LLM2Vec Encoder

```python
print("Loading LLM2Vec encoder...")

# Load model and tokenizer
model = AutoModel.from_pretrained(
    ENCODER_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(ENCODER_PATH)

# Wrap with LLM2Vec
encoder = LLM2Vec(
    model=model,
    tokenizer=tokenizer,
    pooling_mode="mean",
    max_length=512,
    doc_max_length=400,
    skip_instruction=True
)

print("✓ Encoder loaded successfully")
```

### 2.2 Load Letter Embeddings

```python
print("\nLoading pre-computed letter embeddings...")

# Load embeddings
letter_embeddings = np.load(EMBEDDINGS_PATH)
print(f"✓ Loaded {letter_embeddings.shape[0]} letter embeddings")
print(f"  Embedding dimension: {letter_embeddings.shape[1]}")

# Load metadata
letter_metadata = pd.read_csv(METADATA_PATH)
print(f"✓ Loaded metadata for {len(letter_metadata)} letters")
print(f"  Year range: {letter_metadata['Year'].min()}-{letter_metadata['Year'].max()}")
```

---

## 3. Load and Preprocess Publications

### 3.1 Define Publication Metadata

```python
# Publication metadata with temporal information
publications = [
    {"name": "Journal of Researches", "year": 1839, "file": "journal_of_researches_1839.txt", "short": "Journal"},
    {"name": "Cirripedia (Fossil Lepadidae)", "year": 1851, "file": "cirripedia_fossil_lepadidae_1851.txt", "short": "Cirripedia"},
    {"name": "On the Origin of Species", "year": 1859, "file": "origin_of_species_1859.txt", "short": "Origin"},
    {"name": "Fertilisation of Orchids", "year": 1862, "file": "orchids_1862.txt", "short": "Orchids"},
    {"name": "Climbing Plants", "year": 1865, "file": "climbing_plants_1865.txt", "short": "Climbing"},
    {"name": "The Descent of Man", "year": 1871, "file": "descent_of_man_1871.txt", "short": "Descent"},
    {"name": "Expression of Emotions", "year": 1872, "file": "expression_of_emotions_1872.txt", "short": "Expression"},
    {"name": "Insectivorous Plants", "year": 1875, "file": "insectivorous_plants_1875.txt", "short": "Insectivorous"},
    {"name": "Cross and Self Fertilisation", "year": 1876, "file": "cross_and_self_fertilisation_1876.txt", "short": "Fertilisation"},
    {"name": "Forms of Flowers", "year": 1877, "file": "forms_of_flowers_1877.txt", "short": "FormsFlowers"},
    {"name": "Erasmus Darwin", "year": 1879, "file": "erasmus_darwin_1879.txt", "short": "Erasmus"},
    {"name": "Movement in Plants", "year": 1880, "file": "movement_in_plants_1880.txt", "short": "Movement"}
]

publications_df = pd.DataFrame(publications)
print(f"Total publications: {len(publications_df)}")
print(publications_df[['short', 'year', 'file']])
```

### 3.2 Load and Clean Publications

```python
def strip_gutenberg_header(text):
    """Strip Project Gutenberg header and footer."""
    # Find start marker
    start_marker = "*** START OF"
    start_idx = text.find(start_marker)
    if start_idx != -1:
        # Find end of that line
        start_idx = text.find('\n', start_idx) + 1
    else:
        start_idx = 0

    # Find end marker
    end_marker = "*** END OF"
    end_idx = text.find(end_marker)
    if end_idx == -1:
        end_idx = len(text)

    return text[start_idx:end_idx].strip()

# Load all publications
publications_text = {}

print("\nLoading publications...")
for idx, pub in publications_df.iterrows():
    file_path = f"{PUBLICATIONS_PATH}/{pub['file']}"

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Strip PG header/footer
    cleaned_text = strip_gutenberg_header(text)
    publications_text[pub['short']] = cleaned_text

    print(f"✓ {pub['short']:15s} ({pub['year']}): {len(cleaned_text):,} chars")

print(f"\n✓ Loaded {len(publications_text)} publications")
total_chars = sum(len(t) for t in publications_text.values())
print(f"  Total text: {total_chars / 1e6:.2f}M characters")
```

---

## 4. Chunk Publications (Paragraph-Based)

### 4.1 Paragraph-Level Chunking Function

```python
def chunk_publication_by_paragraph(text, min_chunk_size=100, max_chunk_size=1000, overlap=50):
    """
    Chunk text by paragraphs with context window.

    Strategy from stage_3_strategy.md:
    - Base unit: Natural paragraph boundaries
    - Context window: Include ±1 paragraph for semantic coherence
    - Minimum chunk size: 100 chars
    - Maximum chunk size: 1000 chars (to fit in 512 tokens)
    - Overlap: 50 chars between chunks
    """
    # Split by double newline (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > min_chunk_size]

    chunks = []

    for i in range(len(paragraphs)):
        # Current paragraph
        current = paragraphs[i]

        # Add context: previous paragraph
        if i > 0:
            prev_context = paragraphs[i-1][-overlap:] if len(paragraphs[i-1]) > overlap else paragraphs[i-1]
            current = prev_context + " " + current

        # Add context: next paragraph
        if i < len(paragraphs) - 1:
            next_context = paragraphs[i+1][:overlap] if len(paragraphs[i+1]) > overlap else paragraphs[i+1]
            current = current + " " + next_context

        # Truncate if too long (keep within token limit)
        if len(current) > max_chunk_size:
            current = current[:max_chunk_size]

        if len(current) >= min_chunk_size:
            chunks.append({
                'text': current,
                'paragraph_index': i,
                'char_length': len(current)
            })

    return chunks

print("✓ Chunking function defined")
```

### 4.2 Chunk All Publications

```python
print("\n" + "="*60)
print("CHUNKING ALL PUBLICATIONS")
print("="*60)

all_chunks = []

for pub in publications_df.itertuples():
    text = publications_text[pub.short]

    # Chunk the publication
    chunks = chunk_publication_by_paragraph(text)

    # Add metadata to each chunk
    for chunk in chunks:
        chunk['publication'] = pub.short
        chunk['pub_name'] = pub.name
        chunk['pub_year'] = pub.year
        chunk['pub_file'] = pub.file

    all_chunks.extend(chunks)

    print(f"{pub.short:15s}: {len(chunks):4d} chunks (avg {np.mean([c['char_length'] for c in chunks]):.0f} chars)")

print("="*60)
print(f"✓ Total chunks: {len(all_chunks)}")
print(f"  Average chunk size: {np.mean([c['char_length'] for c in all_chunks]):.0f} chars")
print(f"  Min chunk size: {np.min([c['char_length'] for c in all_chunks])} chars")
print(f"  Max chunk size: {np.max([c['char_length'] for c in all_chunks])} chars")
print("="*60)
```

---

## 5. Generate or Load Publication Chunk Embeddings

### 5.1 Check for Existing Embeddings

```python
# Define paths
chunk_embeddings_path = f"{DARWIN_BASE_PATH}/publication_chunks_embeddings.npy"
chunks_metadata_path = f"{DARWIN_BASE_PATH}/publication_chunks_metadata.csv"

# Check if embeddings already exist
embeddings_exist = os.path.exists(chunk_embeddings_path) and os.path.exists(chunks_metadata_path)

if embeddings_exist:
    print("\n" + "="*60)
    print("EXISTING EMBEDDINGS FOUND")
    print("="*60)
    print(f"✓ Embeddings file: {chunk_embeddings_path}")
    print(f"✓ Metadata file: {chunks_metadata_path}")

    # Load existing embeddings
    chunk_embeddings = np.load(chunk_embeddings_path)
    chunks_df = pd.DataFrame(all_chunks)  # Use current chunks
    existing_chunks_df = pd.read_csv(chunks_metadata_path)

    print(f"\n✓ Loaded {chunk_embeddings.shape[0]} pre-computed embeddings")
    print(f"  Embedding dimension: {chunk_embeddings.shape[1]}")

    # Verify chunk count matches
    if len(chunks_df) == len(existing_chunks_df):
        print(f"✓ Chunk count matches: {len(chunks_df)}")
        print("\n⚡ Skipping embedding generation (using cached embeddings)")
        skip_embedding = True
    else:
        print(f"⚠️  Chunk count mismatch: current={len(chunks_df)}, cached={len(existing_chunks_df)}")
        print("   Re-generating embeddings...")
        skip_embedding = False
else:
    print("\n" + "="*60)
    print("NO EXISTING EMBEDDINGS FOUND")
    print("="*60)
    print("Will generate new embeddings...")
    skip_embedding = False
```

### 5.2 Generate Embeddings (if needed)

```python
if not skip_embedding:
    print("\n" + "="*60)
    print("GENERATING EMBEDDINGS FOR PUBLICATION CHUNKS")
    print("="*60)
    print(f"Total chunks to embed: {len(all_chunks)}")
    print("Estimated time: 10-20 minutes")
    print("="*60)

    # Extract chunk texts
    chunk_texts = [chunk['text'] for chunk in all_chunks]

    # Generate embeddings in batches
    batch_size = 32
    chunk_embeddings_list = []

    from datetime import datetime
    start_time = datetime.now()

    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i+batch_size]
        batch_embeddings = encoder.encode(batch, show_progress_bar=False)
        chunk_embeddings_list.append(batch_embeddings)

        # Progress update every 50 batches
        if (i // batch_size) % 50 == 0:
            progress = (i / len(chunk_texts)) * 100
            elapsed = (datetime.now() - start_time).total_seconds()
            estimated_total = (elapsed / (i + 1)) * len(chunk_texts)
            remaining = estimated_total - elapsed
            print(f"Progress: {progress:.1f}% ({i}/{len(chunk_texts)}) - Est. remaining: {remaining/60:.1f} min")

    # Stack all embeddings
    chunk_embeddings = np.vstack(chunk_embeddings_list)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "="*60)
    print("EMBEDDING GENERATION COMPLETE")
    print("="*60)
    print(f"✓ Generated {chunk_embeddings.shape[0]} embeddings")
    print(f"✓ Embedding dimension: {chunk_embeddings.shape[1]}")
    print(f"✓ Duration: {duration/60:.1f} minutes")
    print(f"✓ Speed: {len(chunk_texts) / duration:.1f} chunks/sec")
    print("="*60)

    # Save embeddings
    np.save(chunk_embeddings_path, chunk_embeddings)
    print(f"\n✓ Chunk embeddings saved: {chunk_embeddings_path}")

    # Save chunk metadata
    chunks_df = pd.DataFrame(all_chunks)
    chunks_df.to_csv(chunks_metadata_path, index=False)
    print(f"✓ Chunk metadata saved: {chunks_metadata_path}")
else:
    print("\n✓ Using cached embeddings - saved ~10-20 minutes!")
```

---

## 6. Build FAISS Index for Efficient Retrieval

### 6.1 Create FAISS Index

```python
print("\n" + "="*60)
print("BUILDING FAISS INDEX FOR FAST SIMILARITY SEARCH")
print("="*60)

# Normalize embeddings for cosine similarity
letter_embeddings_normalized = letter_embeddings / np.linalg.norm(letter_embeddings, axis=1, keepdims=True)
chunk_embeddings_normalized = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)

# Create FAISS index
dimension = letter_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product (for normalized = cosine similarity)
index.add(letter_embeddings_normalized.astype('float32'))

print(f"✓ FAISS index built with {index.ntotal} letter embeddings")
print(f"  Index type: Flat (exact search)")
print(f"  Dimension: {dimension}")
print("="*60)
```

---

## 7. Implement Two-Phase Matching

### 7.1 Define Temporal Filtering

```python
def get_relevant_letters_by_time(pub_year, letter_metadata, window_years=5):
    """
    Get letters written BEFORE or during publication year + window.

    Strategy: Only match letters written before publication date.
    Add small window (5 years) to capture late correspondence.
    """
    # Letters must be before or during publication
    max_year = pub_year + window_years

    relevant_indices = letter_metadata[
        (letter_metadata['Year'] <= max_year)
    ].index.tolist()

    return relevant_indices

print("✓ Temporal filtering function defined")
```

### 7.2 Implement Top-K Retrieval with Temporal Filter

```python
def match_chunk_to_letters(chunk_embedding, chunk_pub_year, index, letter_metadata,
                          top_k=50, temporal_window=5):
    """
    Match a publication chunk to relevant letters.

    Phase 1: Coarse retrieval with temporal filtering
    - Retrieve top-K similar letters
    - Filter by publication date (letters must be before publication)

    Returns: List of (letter_idx, similarity_score) tuples
    """
    # Search in FAISS index
    chunk_emb = chunk_embedding.reshape(1, -1).astype('float32')
    similarities, indices = index.search(chunk_emb, top_k * 2)  # Get more to account for filtering

    # Convert to list of results
    results = []
    for sim, idx in zip(similarities[0], indices[0]):
        letter_year = letter_metadata.iloc[idx]['Year']

        # Temporal filter: letter must be before publication (+window)
        if letter_year <= chunk_pub_year + temporal_window:
            results.append({
                'letter_idx': int(idx),
                'similarity': float(sim),
                'letter_year': int(letter_year),
                'temporal_gap': int(chunk_pub_year - letter_year)
            })

    # Return top-K after filtering
    return results[:top_k]

print("✓ Matching function defined")
```

### 7.3 Match All Chunks to Letters

```python
print("\n" + "="*60)
print("MATCHING PUBLICATION CHUNKS TO LETTERS")
print("="*60)
print(f"Chunks to match: {len(chunk_embeddings)}")
print(f"Letters in index: {len(letter_embeddings)}")
print("="*60)

all_matches = []

for chunk_idx, chunk in enumerate(chunks_df.itertuples()):
    # Get chunk embedding
    chunk_emb_normalized = chunk_embeddings_normalized[chunk_idx]

    # Match to letters
    matches = match_chunk_to_letters(
        chunk_emb_normalized,
        chunk.pub_year,
        index,
        letter_metadata,
        top_k=50,
        temporal_window=5
    )

    # Store matches with chunk info
    for match in matches:
        all_matches.append({
            'chunk_idx': chunk_idx,
            'publication': chunk.publication,
            'pub_year': chunk.pub_year,
            'paragraph_index': chunk.paragraph_index,
            'chunk_text': chunk.text[:200],  # First 200 chars
            'letter_idx': match['letter_idx'],
            'letter_id': letter_metadata.iloc[match['letter_idx']]['File'],
            'letter_year': match['letter_year'],
            'letter_receiver': letter_metadata.iloc[match['letter_idx']]['Receiver'],
            'similarity': match['similarity'],
            'temporal_gap': match['temporal_gap']
        })

    # Progress update
    if (chunk_idx + 1) % 500 == 0:
        progress = ((chunk_idx + 1) / len(chunks_df)) * 100
        print(f"Progress: {progress:.1f}% ({chunk_idx + 1}/{len(chunks_df)} chunks matched)")

print("\n" + "="*60)
print("MATCHING COMPLETE")
print("="*60)
print(f"✓ Total matches: {len(all_matches)}")
print(f"✓ Average matches per chunk: {len(all_matches) / len(chunks_df):.1f}")
print("="*60)
```

---

## 8. Apply Multi-Tier Similarity Thresholds

### 8.1 Classify Match Confidence

```python
def classify_match_confidence(similarity):
    """
    Multi-tier thresholding from stage_3_strategy.md:
    - High confidence: ≥0.75
    - Moderate confidence: 0.60-0.75
    - Low confidence: 0.45-0.60
    - Discard: <0.45
    """
    if similarity >= 0.75:
        return "high"
    elif similarity >= 0.60:
        return "moderate"
    elif similarity >= 0.45:
        return "low"
    else:
        return "discard"

# Add confidence classification
matches_df = pd.DataFrame(all_matches)
matches_df['confidence'] = matches_df['similarity'].apply(classify_match_confidence)

# Filter out discarded matches
matches_filtered = matches_df[matches_df['confidence'] != 'discard'].copy()

print("\n" + "="*60)
print("CONFIDENCE DISTRIBUTION")
print("="*60)
print(matches_filtered['confidence'].value_counts())
print(f"\nTotal filtered matches: {len(matches_filtered)}")
print(f"Discarded: {len(matches_df) - len(matches_filtered)}")
print("="*60)
```

---

## 9. Save Results

### 9.1 Save All Matches

```python
# Save filtered matches
matches_output_path = f"{DARWIN_BASE_PATH}/letter_publication_matches.csv"
matches_filtered.to_csv(matches_output_path, index=False)
print(f"✓ Matches saved: {matches_output_path}")

# Save high-confidence matches separately
high_conf_matches = matches_filtered[matches_filtered['confidence'] == 'high']
high_conf_path = f"{DARWIN_BASE_PATH}/letter_publication_matches_high_confidence.csv"
high_conf_matches.to_csv(high_conf_path, index=False)
print(f"✓ High-confidence matches saved: {high_conf_path}")
```

### 9.2 Generate Summary Statistics

```python
print("\n" + "="*60)
print("MATCHING SUMMARY BY PUBLICATION")
print("="*60)

summary = matches_filtered.groupby(['publication', 'pub_year']).agg({
    'letter_idx': 'count',
    'similarity': ['mean', 'max'],
    'temporal_gap': 'mean'
}).round(3)

summary.columns = ['total_matches', 'avg_similarity', 'max_similarity', 'avg_temporal_gap']
summary = summary.sort_values('pub_year')

print(summary)
print("="*60)
```

---

## 10. Validation Against Ground Truth

### 10.1 Load Ground Truth Timeline

```python
# Ground truth from publications_worked_on_overtime.md
ground_truth = {
    'Journal': {'active_periods': [(1830, 1840), (1855, 1860)]},
    'Cirripedia': {'active_periods': [(1840, 1855)]},
    'Origin': {'active_periods': [(1840, 1865)]},
    'Orchids': {'active_periods': [(1835, 1862)]},
    'Climbing': {'active_periods': [(1860, 1870)]},
    'Descent': {'active_periods': [(1860, 1875)]},
    'Expression': {'active_periods': [(1830, 1845), (1870, 1875)]},
    'Insectivorous': {'active_periods': [(1860, 1880)]},
    'Fertilisation': {'active_periods': [(1870, 1876)]},
    'FormsFlowers': {'active_periods': [(1855, 1877)]},
    'Erasmus': {'active_periods': [(1870, 1879)]},
    'Movement': {'active_periods': [(1860, 1880), (1875, 1882)]}
}

def is_in_active_period(year, publication):
    """Check if a letter year falls within active work periods."""
    if publication not in ground_truth:
        return False

    for start, end in ground_truth[publication]['active_periods']:
        if start <= year <= end:
            return True
    return False

# Add ground truth validation
matches_filtered['in_active_period'] = matches_filtered.apply(
    lambda row: is_in_active_period(row['letter_year'], row['publication']),
    axis=1
)

print("✓ Ground truth validation added")
```

### 10.2 Calculate Temporal Accuracy

```python
print("\n" + "="*60)
print("TEMPORAL ACCURACY (vs Ground Truth)")
print("="*60)

accuracy_by_pub = matches_filtered.groupby('publication').agg({
    'in_active_period': ['sum', 'count', 'mean']
}).round(3)

accuracy_by_pub.columns = ['in_active_period', 'total_matches', 'temporal_accuracy']
accuracy_by_pub = accuracy_by_pub.sort_values('temporal_accuracy', ascending=False)

print(accuracy_by_pub)

overall_accuracy = matches_filtered['in_active_period'].mean()
print(f"\n✓ Overall temporal accuracy: {overall_accuracy:.1%}")
print("="*60)

# Success check
if overall_accuracy >= 0.70:
    print("✅ SUCCESS: ≥70% temporal accuracy achieved!")
else:
    print("⚠️  Below 70% target - may need refinement")
```

---

## 11. Generate Sample Output

### 11.1 Show Top Matches for Origin of Species

```python
print("\n" + "="*60)
print("SAMPLE: TOP MATCHES FOR 'ORIGIN OF SPECIES'")
print("="*60)

origin_matches = matches_filtered[
    (matches_filtered['publication'] == 'Origin') &
    (matches_filtered['confidence'] == 'high')
].sort_values('similarity', ascending=False).head(10)

for idx, match in origin_matches.iterrows():
    print(f"\nChunk {match['chunk_idx']} (similarity: {match['similarity']:.3f})")
    print(f"  Letter: {match['letter_id']} ({match['letter_year']}) → {match['letter_receiver']}")
    print(f"  Temporal gap: {match['temporal_gap']} years")
    print(f"  Active period: {'✓' if match['in_active_period'] else '✗'}")
    print(f"  Text: {match['chunk_text'][:150]}...")
    print("-" * 60)
```

---

## 12. Cleanup & Summary

### 12.1 Clear GPU Memory

```python
del encoder
del model
import gc
gc.collect()
torch.cuda.empty_cache()

print("✓ GPU memory cleared")
```

### 12.2 Final Summary

```python
print("\n" + "="*70)
print("STAGE 3 COMPLETE - FINAL SUMMARY")
print("="*70)
print(f"✓ Processed {len(publications_df)} Darwin publications")
print(f"✓ Generated {len(chunks_df)} publication chunks")
print(f"✓ Matched against {len(letter_embeddings)} Darwin letters")
print(f"✓ Total matches (filtered): {len(matches_filtered)}")
print(f"✓ High-confidence matches: {len(high_conf_matches)}")
print(f"✓ Temporal accuracy: {overall_accuracy:.1%}")
print("\n" + "="*70)
print("OUTPUTS:")
print("="*70)
print(f"1. Chunk embeddings: {chunk_embeddings_path}")
print(f"2. Chunk metadata: {chunks_metadata_path}")
print(f"3. All matches: {matches_output_path}")
print(f"4. High-confidence matches: {high_conf_path}")
print("\n" + "="*70)
print("WHAT WE ACHIEVED:")
print("="*70)
print("• Fine-grained letter-publication matching at paragraph level")
print("• Temporal filtering to ensure chronological validity")
print("• Multi-tier confidence classification")
print("• Validation against ground truth timeline")
print("• Ready for historical analysis and visualization")
print("="*70)
```

---

## Notes

- **Processing time**: ~30-45 minutes total (10-20 min chunking/embedding, 10-15 min matching)
- **Memory usage**: ~30GB GPU memory for encoder
- **Chunking**: ~5,000-10,000 chunks from 12 publications (avg ~600 chunks/pub)
- **Matching**: 50 top letters per chunk, filtered by temporal validity
- **Success metric**: ≥70% temporal accuracy (matches in ground truth periods)

---

## Troubleshooting

### If Chunking Creates Too Many/Few Chunks:
- Adjust `min_chunk_size` (default 100) and `max_chunk_size` (default 1000)
- Adjust `overlap` (default 50 chars)

### If Embedding Generation is Slow:
- Reduce `batch_size` from 32 to 16
- Process one publication at a time for testing

### If Too Many Low-Quality Matches:
- Increase similarity thresholds in `classify_match_confidence`
- Reduce `temporal_window` from 5 to 2 years

### If FAISS Index Fails:
- Fallback to scikit-learn: `cosine_similarity(chunk_embeddings, letter_embeddings)`
- Slower but works without FAISS

### If Temporal Accuracy is Low (<70%):
- Review ground truth periods (may need adjustment)
- Check if letters are correctly dated
- Consider adding correspondent weighting (Hooker, Wallace, etc.)
