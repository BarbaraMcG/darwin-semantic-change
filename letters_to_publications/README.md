# Letters to publications: matching Darwin's correspondence to his published works

This folder contains the code for Section 7 of *Darwin's Language Evolution: decomposing computational methods in the age of AI* ("From letters to publications"). The code was developed by Sean Lyons.

The pipeline fine-tunes Mistral-7B on the Darwin Correspondence Project letters, converts the fine-tuned model into a bidirectional text encoder with LLM2Vec, embeds every letter and every paragraph of twelve of Darwin's publications, and then matches letters to publication passages by cosine similarity.

Mistral is used only to produce embeddings. Apart from a sanity check in Stage 1.5, where the fine-tuned model is given a few letter openings to confirm it has learned Darwin's style, no prompts or instructions are used, and no generated text enters the analysis.

## Stages

Each stage has a notebook and a Markdown walkthrough with the same code and an explanation. Stage 1 exists only as a walkthrough.

| Stage | Files | What it does |
|---|---|---|
| 1 | `stages/stage_1_decoder_finetuning.md` | LoRA fine-tuning (r = 64, 3 epochs, 4-bit quantised) of `mistralai/Mistral-7B-v0.1` (base model, not instruction-tuned) on the letters, using next-token prediction |
| 1.5 | `stages/stage_1.5.ipynb`, `stages/stage_1.5_model_testing.md` | Checks that fine-tuning worked: sample generations, perplexity, and comparison with the base model |
| 2 | `stages/stage_2_llm2vec.ipynb`, `stages/stage_2_llm2vec_conversion.md` | Merges the LoRA weights and enables bidirectional attention with LLM2Vec (no extra MNTP or SimCSE training) |
| 2.5 | `stages/stage_2.5.ipynb`, `stages/stage_2.5_generate_embeddings.md` | Mean-pooled embeddings for every letter (no instruction prefix) |
| 3 | `stages/stage_3.ipynb`, `stages/stage_3_letter_publication_matching.md` | Splits the publications into paragraphs, embeds them, and matches letters to passages |
| 3.5 | `stages/stage_3.5.ipynb`, `stages/stage_3.5_results_analysis.md` | Correspondent, temporal-gap, and similarity-versus-gap analyses, plus figures |

`results/figures/` holds the figures produced by Stage 3.5, and `results/*.json` hold the summary statistics reported in the paper. One example is the temporal coherence of 0.753 for the LLM2Vec encoder.

## Data

- **Letters.** The stages expect `letters_date_fix_new.tsv` with the columns `File, Year, Sender, Receiver, Letter, Footnote`, which has one row per letter (15,107 letters). It was built from the Darwin Correspondence Project TEI-XML (https://www.darwinproject.ac.uk/). The transcriptions are not redistributed here, so please obtain them from the DCP.
- **Publications.** `publications/download_publications.py` downloads the twelve publications from Project Gutenberg, Darwin Online and the Internet Archive. `publications/publications_metadata.md` lists them.
- **Intermediate outputs.** The fine-tuned model, the encoder, the embeddings and the match tables (about 250 MB of CSV) are not included because of their size.

## Running

The notebooks were run on Google Colab with an A100 (40 GB). They read and write everything under `DARWIN_BASE_PATH = "/content/drive/My Drive/Darwin"`, so change that path if you run them elsewhere.

Mistral-7B is a gated model on Hugging Face. Accept its terms on the model page and set your token as an environment variable before running:

```bash
export HF_TOKEN=...   # your own Hugging Face token
```

Main dependencies, with each stage's notebook installing its own:

```bash
pip install "transformers>=4.36.0" datasets accelerate bitsandbytes "peft>=0.7.0" trl sentencepiece torch
pip install git+https://github.com/McGill-NLP/llm2vec.git
pip install numpy pandas scipy scikit-learn faiss-cpu matplotlib seaborn plotly networkx
```
