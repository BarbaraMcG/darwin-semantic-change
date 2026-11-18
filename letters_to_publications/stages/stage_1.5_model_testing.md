# Stage 1.5: Testing & Validating the Fine-Tuned Darwin Model

## Objective
Load the fine-tuned Mistral-7B model from Stage 1 and validate that it has successfully learned Darwin's writing style through generation tests, perplexity analysis, and qualitative evaluation.

---

## 1. Setup & Load Model

### 1.1 Install Dependencies

```python
!pip install -q transformers>=4.36.0 datasets accelerate bitsandbytes peft>=0.7.0 torch

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import pandas as pd
import numpy as np
import json
from datetime import datetime
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
# Mount Google Drive to access fine-tuned model
from google.colab import drive
drive.mount('/content/drive')

# Set base path for Darwin project
DARWIN_BASE_PATH = "/content/drive/My Drive/Darwin"
MODEL_PATH = f"{DARWIN_BASE_PATH}/darwin_mistral_7b_finetuned"

print(f"Model path: {MODEL_PATH}")

# Verify model exists
import os
if os.path.exists(MODEL_PATH):
    print(f"✓ Model directory found")
    # List model files
    model_files = os.listdir(MODEL_PATH)
    print(f"Files: {model_files[:10]}...")  # Show first 10 files
else:
    print(f"✗ Model NOT found at {MODEL_PATH}")
```

### 1.4 GPU Configuration

```python
# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  No GPU detected - inference will be slow")
```

---

## 2. Load Fine-Tuned Model

### 2.1 Load Base Model with LoRA Adapters

```python
# Base model name
base_model_name = "mistralai/Mistral-7B-v0.1"

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
print("✓ Tokenizer loaded")

# Load base model with 4-bit quantization
print("Loading base model with quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN
)
print("✓ Base model loaded")

# Load fine-tuned LoRA adapters
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()  # Set to evaluation mode
print("✓ Fine-tuned model ready for inference")
```

### 2.2 Load Training Metadata (Optional)

```python
# Load training config if available
config_path = f"{MODEL_PATH}/training_config.json"
if os.path.exists(config_path):
    try:
        with open(config_path, 'r') as f:
            training_config = json.load(f)
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Training samples: {training_config.get('training_samples', 'N/A')}")
        print(f"Validation samples: {training_config.get('validation_samples', 'N/A')}")
        print(f"Epochs: {training_config.get('training_args', {}).get('epochs', 'N/A')}")
        eval_loss = training_config.get('final_metrics', {}).get('eval_loss', 'N/A')
        if eval_loss != 'N/A':
            print(f"Final eval loss: {eval_loss:.4f}")
        else:
            print(f"Final eval loss: {eval_loss}")
        print(f"Trained on: {training_config.get('trained_on', 'N/A')}")
        print("="*60)
    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠️  Training config exists but couldn't be loaded: {e}")
        print("This is OK - the model itself is fine")
else:
    print("Training config not found (this is OK)")
```

---

## 3. Text Generation Tests

### 3.1 Define Generation Function

```python
def generate_darwin_text(prompt, max_new_tokens=200, temperature=0.7, top_p=0.9, do_sample=True):
    """
    Generate text in Darwin's style given a prompt.

    Args:
        prompt: Starting text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more creative)
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling (vs greedy decoding)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

print("✓ Generation function ready")
```

### 3.2 Test 1: Letter to Joseph Hooker (Darwin's Top Correspondent)

```python
prompt_1 = """Date: 1859
From: Darwin, C. R.
To: Hooker, J. D.

Letter:
My dear Hooker,
I have been thinking deeply about the distribution of species across"""

print("="*70)
print("TEST 1: Letter to Joseph Hooker")
print("="*70)
print("\nPROMPT:")
print(prompt_1)
print("\n" + "-"*70)
print("GENERATED:")
print("-"*70)
generated_1 = generate_darwin_text(prompt_1, max_new_tokens=150, temperature=0.7)
print(generated_1)
print("="*70)
```

### 3.3 Test 2: Letter to Alfred Russel Wallace (Natural Selection)

```python
prompt_2 = """Date: 1858
From: Darwin, C. R.
To: Wallace, A. R.

Letter:
Dear Wallace,
Your observations on natural selection and the struggle for existence are"""

print("\n\n" + "="*70)
print("TEST 2: Letter to Alfred Russel Wallace")
print("="*70)
print("\nPROMPT:")
print(prompt_2)
print("\n" + "-"*70)
print("GENERATED:")
print("-"*70)
generated_2 = generate_darwin_text(prompt_2, max_new_tokens=150, temperature=0.7)
print(generated_2)
print("="*70)
```

### 3.4 Test 3: "I believe in" Statement (Belief/Theory Expression)

```python
prompt_3a = """Date: 1860
From: Darwin, C. R.
To: Lyell, Charles

Letter:
My dear Lyell,
After all my observations and experiments, I believe in"""

print("\n\n" + "="*70)
print("TEST 3: 'I believe in' Statement")
print("="*70)
print("\nPROMPT:")
print(prompt_3a)
print("\n" + "-"*70)
print("GENERATED:")
print("-"*70)
generated_3a = generate_darwin_text(prompt_3a, max_new_tokens=150, temperature=0.7)
print(generated_3a)
print("="*70)
print("\nNote: This tests how the model completes Darwin's statements of belief/theory.")
```

### 3.6 Test 5: Letter About Barnacles (Darwin's Extended Study)

```python
prompt_3 = """Date: 1851
From: Darwin, C. R.
To: Gray, Asa

Letter:
My dear Gray,
I must tell you about my recent work on the cirripedes. The anatomical structures"""

print("\n\n" + "="*70)
print("TEST 5: Letter About Barnacles")
print("="*70)
print("\nPROMPT:")
print(prompt_3)
print("\n" + "-"*70)
print("GENERATED:")
print("-"*70)
generated_3 = generate_darwin_text(prompt_3, max_new_tokens=150, temperature=0.7)
print(generated_3)
print("="*70)
```

### 3.8 Test 7: Scientific Observation (High Temperature - More Creative)

```python
prompt_4 = """Date: 1871
From: Darwin, C. R.
To: Huxley, T. H.

Letter:
Dear Huxley,
I have observed in my experiments on insectivorous plants that"""

print("\n\n" + "="*70)
print("TEST 7: Scientific Observation (Creative)")
print("="*70)
print("\nPROMPT:")
print(prompt_4)
print("\n" + "-"*70)
print("GENERATED:")
print("-"*70)
generated_4 = generate_darwin_text(prompt_4, max_new_tokens=150, temperature=0.9)  # Higher temp
print(generated_4)
print("="*70)
```

### 3.10 Test 9: Greedy Decoding (Most Likely Output)

```python
prompt_5 = """Date: 1872
From: Darwin, C. R.
To: Darwin, Francis

Letter:
My dear Frank,
Thank you for your letter. Regarding your question about the expression of emotions in"""

print("\n\n" + "="*70)
print("TEST 9: Greedy Decoding (Most Likely)")
print("="*70)
print("\nPROMPT:")
print(prompt_5)
print("\n" + "-"*70)
print("GENERATED:")
print("-"*70)
generated_5 = generate_darwin_text(prompt_5, max_new_tokens=150, temperature=0.7, do_sample=False)  # Greedy
print(generated_5)
print("="*70)
```

---

## 4. Qualitative Analysis

### 4.1 Style Characteristics Checklist

```python
def analyze_style(generated_text):
    """
    Analyze generated text for Darwin-specific characteristics.
    """
    characteristics = {
        "Formal address": any(phrase in generated_text for phrase in ["My dear", "Dear", "My dear Sir"]),
        "First person": " I " in generated_text or generated_text.startswith("I "),
        "Scientific terms": any(word in generated_text.lower() for word in
                                ["species", "variation", "selection", "observations", "experiments"]),
        "Tentative language": any(phrase in generated_text.lower() for phrase in
                                  ["it seems", "i believe", "perhaps", "possibly", "may be"]),
        "Gratitude/politeness": any(phrase in generated_text.lower() for phrase in
                                    ["thank you", "grateful", "kindly", "obliged"]),
        "References to work": any(word in generated_text.lower() for word in
                                  ["manuscript", "chapter", "work", "study", "paper"])
    }
    return characteristics

print("\n\n" + "="*70)
print("STYLE CHARACTERISTICS ANALYSIS")
print("="*70)

all_generated = [generated_1, generated_2, generated_3a, generated_3, generated_4, generated_5]
for i, gen_text in enumerate(all_generated, 1):
    print(f"\n--- Test {i} ---")
    analysis = analyze_style(gen_text)
    for characteristic, present in analysis.items():
        status = "✓" if present else "✗"
        print(f"{status} {characteristic}")
```

### 4.2 Length Statistics

```python
print("\n\n" + "="*70)
print("LENGTH STATISTICS")
print("="*70)

for i, gen_text in enumerate(all_generated, 1):
    # Extract only the generated portion (after prompt)
    prompts = [prompt_1, prompt_2, prompt_3a, prompt_3, prompt_4, prompt_5]
    generated_only = gen_text[len(prompts[i-1]):]

    word_count = len(generated_only.split())
    char_count = len(generated_only)

    print(f"\nTest {i}:")
    print(f"  Words: {word_count}")
    print(f"  Characters: {char_count}")
    print(f"  Avg word length: {char_count / word_count if word_count > 0 else 0:.1f}")
```

---

## 5. Perplexity Evaluation

### 5.1 Load Validation Data

```python
# Load a sample of validation letters for perplexity calculation
data_path = f"{DARWIN_BASE_PATH}/letters_date_fix_new.tsv"
df = pd.read_csv(data_path, sep='\t', on_bad_lines='skip')

# Get Darwin's letters from late period (validation set)
darwin_letters = df[df['Sender'].str.contains('Darwin, C. R.', na=False)].copy()
darwin_letters = darwin_letters.sort_values('Year')

# Take last 50 letters as validation sample
val_sample = darwin_letters.tail(50).copy()

print(f"Validation sample: {len(val_sample)} letters")
print(f"Year range: {val_sample['Year'].min()}-{val_sample['Year'].max()}")
```

### 5.2 Calculate Perplexity

```python
def calculate_perplexity(model, tokenizer, texts, max_length=512):
    """
    Calculate perplexity on a set of texts.
    Lower perplexity = better fit to the data.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0

    for text in texts:
        if pd.isna(text) or len(str(text)) < 10:
            continue

        # Tokenize
        encodings = tokenizer(str(text), return_tensors='pt', truncation=True,
                             max_length=max_length, padding=True)
        input_ids = encodings.input_ids.to(model.device)

        # Calculate loss
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        # Accumulate
        total_loss += loss.item() * input_ids.size(1)
        total_tokens += input_ids.size(1)

    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity, avg_loss

print("Calculating perplexity on validation set...")
print("(This may take a few minutes)")

val_texts = val_sample['Letter'].tolist()
perplexity, avg_loss = calculate_perplexity(model, tokenizer, val_texts)

print("\n" + "="*60)
print("PERPLEXITY RESULTS")
print("="*60)
print(f"Average Loss: {avg_loss:.4f}")
print(f"Perplexity: {perplexity:.2f}")
print("="*60)
print("\nInterpretation:")
print(f"  • Lower perplexity = better fit to Darwin's style")
print(f"  • Expected range: 5-15 for well-fine-tuned model")
print(f"  • Your model: {'✓ Good' if perplexity < 15 else '⚠️ May need more training'}")
```

---

## 6. Comparison: Base vs Fine-Tuned

### 6.1 Load Base Model (No Fine-Tuning)

```python
print("Loading base Mistral model (no fine-tuning) for comparison...")

base_model_comparison = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN
)
base_model_comparison.eval()

print("✓ Base model loaded")
```

### 6.2 Side-by-Side Comparison

```python
def generate_base_text(prompt, max_new_tokens=150):
    """Generate from base model (no fine-tuning)"""
    inputs = tokenizer(prompt, return_tensors="pt").to(base_model_comparison.device)

    with torch.no_grad():
        outputs = base_model_comparison.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test prompt
comparison_prompt = """Date: 1859
From: Darwin, C. R.
To: Hooker, J. D.

Letter:
My dear Hooker,
I have been considering the question of species variation and"""

print("="*70)
print("COMPARISON: BASE MODEL vs FINE-TUNED MODEL")
print("="*70)
print("\nPROMPT:")
print(comparison_prompt)

print("\n" + "-"*70)
print("BASE MODEL (No Fine-Tuning):")
print("-"*70)
base_output = generate_base_text(comparison_prompt)
print(base_output)

print("\n" + "-"*70)
print("FINE-TUNED MODEL:")
print("-"*70)
finetuned_output = generate_darwin_text(comparison_prompt, max_new_tokens=150)
print(finetuned_output)

print("\n" + "="*70)
print("ANALYSIS:")
print("="*70)
print("Expected differences:")
print("  • Fine-tuned should use more Darwin-specific vocabulary")
print("  • Fine-tuned should match 19th-century letter style")
print("  • Fine-tuned should maintain scientific focus")
print("="*70)
```

---

## 7. Save Test Results

### 7.1 Export Test Outputs

```python
# Compile all test results
test_results = {
    "test_date": datetime.now().isoformat(),
    "model_path": MODEL_PATH,
    "validation_perplexity": float(perplexity),
    "validation_loss": float(avg_loss),
    "generation_tests": [
        {"test_id": 1, "prompt": prompt_1, "generated": generated_1},
        {"test_id": 2, "prompt": prompt_2, "generated": generated_2},
        {"test_id": 3, "prompt": prompt_3a, "generated": generated_3a, "note": "I believe in statement"},
        {"test_id": 4, "prompt": prompt_3, "generated": generated_3},
        {"test_id": 5, "prompt": prompt_4, "generated": generated_4},
        {"test_id": 6, "prompt": prompt_5, "generated": generated_5}
    ],
    "comparison": {
        "prompt": comparison_prompt,
        "base_model": base_output,
        "fine_tuned": finetuned_output
    }
}

# Save to Google Drive
results_path = f"{DARWIN_BASE_PATH}/stage_1_test_results.json"
with open(results_path, 'w') as f:
    json.dump(test_results, f, indent=2)

print(f"✓ Test results saved to {results_path}")
```

---

## 8. Cleanup

### 8.1 Clear GPU Memory

```python
# Clear memory
import gc

del model
del base_model
del base_model_comparison

torch.cuda.empty_cache()
gc.collect()

print("✓ GPU memory cleared")
```

### 8.2 Summary & Next Steps

```python
print("\n" + "="*70)
print("STAGE 1.5 COMPLETE - SUMMARY")
print("="*70)
print(f"✓ Fine-tuned model loaded successfully")
print(f"✓ Generated {len(all_generated)} test samples")
print(f"✓ Validation perplexity: {perplexity:.2f}")
print(f"✓ Test results saved to Google Drive")
print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. Review generated samples for quality")
print("2. If perplexity < 15 and style looks good → Proceed to Stage 2")
print("3. If results are poor → Consider retraining with adjusted hyperparameters")
print("\n⏭️  READY FOR STAGE 2: LLM2Vec Encoder Conversion")
print("="*70)
```

---

## Notes

- This notebook tests the fine-tuned model without modifying it
- All tests are performed in evaluation mode (no training)
- Results are saved to Google Drive for documentation
- No A100 required (inference can run on T4 or even CPU, just slower)
- Use this to validate before proceeding to Stage 2
