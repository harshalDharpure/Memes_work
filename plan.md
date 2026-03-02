# CAPMeme Project — Complete Explanation for Mentor

This document explains the **entire project**: what it does, which dataset and ground truth we use, which models we have, how they are trained and evaluated, and how to run everything.

---

## 1. Project in One Sentence

**CAPMeme** is a **multimodal sarcasm detection** system for **Hindi memes** that predicts whether a meme is sarcastic (binary) by modelling **affective incongruity** between the image and the text, using **hierarchical emotion labels** as auxiliary supervision and optionally **ConceptNet** as external knowledge.

---

## 2. Task and Goal

| Item | Description |
|------|-------------|
| **Task** | Binary classification: **Sarcastic (1) vs Non-Sarcastic (0)**. |
| **Input** | One meme = **one image (PNG)** + **one Hindi text** (caption/overlay). |
| **Output** | A single label: 0 or 1 (sarcasm probability from the model is also computed). |
| **Goal** | Train and compare multiple models (one proposed + baselines and ablations) on the same data and report metrics for a research paper (e.g. top NLP/multimodal venue). |

---

## 3. Dataset: What We Use

### 3.1 Dataset Name and Location

| Item | Value |
|------|--------|
| **Dataset name** | **EMOFF_MEME** (Hindi meme dataset with emotion and sarcasm labels). |
| **CSV file** | `EMOFF_MEME.csv` (one row per meme). |
| **Images** | PNG files in folder `my_meme_data/my_meme_data/`. Image filename in each row is in the `Name` column (e.g. `hindi_meme0.png`). |

### 3.2 CSV Columns (Ground Truth and Metadata)

| Column | Description | Use in project |
|--------|-------------|----------------|
| **Name** | Image filename (e.g. `hindi_meme0.png`) | Used to load the image; must match a file in the image folder. |
| **text** | Hindi meme text (caption/overlay) | Input to the text encoder. |
| **Level1** | **0 or 1** — **main ground truth**: 0 = non-sarcastic, 1 = sarcastic | **Primary label for training and evaluation.** All models are trained to predict this. |
| **Level2** | Coarser label (not used in current code) | — |
| **Level 3(Emotion1)** | First emotion (e.g. Joy, Rage, Envy, Neglect, none) | Used to build **multi-hot emotion** target. |
| **Level 4(Emotion2)** | Second emotion | Same. |
| **Level 5(Emotion3)** | Third emotion | Same. |

- **Ground truth for the main task:** **Level1** (binary sarcasm).  
- **Auxiliary ground truth:** Level 3–5 are combined into a **multi-hot emotion vector** (all non-`"none"` emotions from these three columns, lowercased). This is used only by models that have an **affect (CAP)** module to supervise the emotion space.

### 3.3 Dataset Statistics (after preprocessing)

| Statistic | Value |
|-----------|--------|
| **Rows in CSV** | ~7,416 |
| **Unique image names** | ~7,406 |
| **Images actually present** | ~7,376 (some rows have missing images) |
| **After `filter_missing`** | **7,386** samples (rows with missing images dropped) |
| **Class balance (Level1)** | Imbalanced: ~5,632 non-sarcastic (0), ~1,784 sarcastic (1) (~76% / 24%) |
| **Emotion types (Level 3–5)** | 18 distinct emotions (e.g. Joy, Rage, Envy, Neglect, Fear, Disappointment, Shame, Suffering, etc.); `"none"` is excluded from the emotion vocabulary. |
| **Language** | Hindi (meme text). |

### 3.4 Data Splits (reproducible)

- **Method:** Stratified split by **Level1** so that train/val/test have similar sarcasm ratio.
- **Default ratios:** Train **70%**, validation **15%**, test **15%** (one fixed seed, e.g. 42).
- **Reproducibility:** Split indices can be saved to a JSON file (e.g. `outputs/splits.json`) and reused for all experiments so every model sees the same train/val/test sets.

---

## 4. Models: What Each One Is and What It Uses

We have **7 model variants**: 1 main model (in 2 settings), 2 ablations of it, and 3 baselines. All are trained to predict **Level1 (sarcasm)**; only the CAPMeme family uses **Level 3–5 (emotion)** as auxiliary supervision.

### 4.1 Encoders Used Across Models

| Modality | Encoder | Default name | Output size | Role |
|----------|---------|--------------|-------------|------|
| **Vision** | CLIP Vision | `openai/clip-vit-base-patch32` | 768 (E_v) | Encodes the meme image. |
| **Text** | BERT-style encoder | `bert-base-multilingual-cased` (or `ai4bharat/indic-bert` if available) | 768 (E_t) | Encodes the Hindi text; [CLS] representation used. |

- **E_v:** CLIP vision encoder output (pooler or first token).  
- **E_t:** Text encoder [CLS] token (`last_hidden_state[:, 0]`).

### 4.2 Model 1: CAPMeme (full) — `capmeme`

- **What it is:** The main proposed model.
- **Inputs:** Image → CLIP → E_v; Text → text encoder → E_t; optional ConceptNet → E_kg.
- **Core idea:** Map E_v and E_t to an **emotion space** (A_v, A_t), compute **affective contrast** F_contrast = |A_v − A_t|, fuse with E_kg and classify.
- **Ground truth used:** **Level1** (sarcasm) for the classifier; **Level 3–5** (multi-hot emotion) for supervising A_v and A_t.
- **When to use:** Run with `--use_kg` if you want ConceptNet; data loader will call ConceptNet API per sample (slower).

### 4.3 Model 2: CAPMeme without KG — `capmeme_no_kg`

- **What it is:** Same as CAPMeme but **no ConceptNet**; E_kg is replaced by a learned linear layer from E_t (fallback).
- **Ground truth:** Same as CAPMeme (Level1 + Level 3–5 emotion).
- **When to use:** Default choice for most runs (faster, no API).

### 4.4 Model 3: CAPMeme without emotion loss — `capmeme_no_emotion`

- **What it is:** Same architecture as CAPMeme, but **affect_weight = 0**: no MSE loss on A_v, A_t. Only BCE for sarcasm.
- **Ground truth:** Only **Level1** (sarcasm). Level 3–5 are in the data but not used in the loss.
- **Purpose:** Ablation to show the benefit of emotion supervision.

### 4.5 Model 4: CAPMeme with concat fusion (ablation) — `capmeme_concat_fusion`

- **What it is:** Same encoders and KG module as CAPMeme, but instead of **F_contrast** the fusion input is **concat(E_v, E_t, E_kg)**. So no affective contrast feature; just concatenated representations.
- **Ground truth:** Level1 + Level 3–5 (emotion loss still applied to A_v, A_t).
- **Purpose:** Ablation to show that **F_contrast** (affective incongruity) helps over simple concatenation.

### 4.6 Model 5: Text-only baseline — `text_only`

- **What it is:** Only the **text encoder**; [CLS] → linear layers → sarcasm logit. No image, no CAP, no KG.
- **Ground truth:** Only **Level1**.
- **Purpose:** Baseline to show how much the image modality adds.

### 4.7 Model 6: Image-only baseline — `image_only`

- **What it is:** Only **CLIP vision**; E_v → linear layers → sarcasm logit. No text encoder, no CAP, no KG.
- **Ground truth:** Only **Level1**.
- **Purpose:** Baseline to show how much the text modality adds.

### 4.8 Model 7: Late fusion baseline — `late_fusion`

- **What it is:** **Concat(E_v, E_t)** → MLP → sarcasm logit. Same encoders as CAPMeme, but **no** CAP (no A_v, A_t, no F_contrast), **no** emotion supervision, **no** KG.
- **Ground truth:** Only **Level1**.
- **Purpose:** Simple multimodal baseline; comparison shows the benefit of CAP + emotion.

---

## 5. Summary Table: Models vs Ground Truth vs Components

| Model | Image | Text | CAP (A_v, A_t, F_contrast) | Emotion loss (Level 3–5) | KG (ConceptNet) | Main GT |
|-------|-------|------|-----------------------------|--------------------------|-----------------|--------|
| capmeme | ✓ | ✓ | ✓ | ✓ | ✓ (optional) | Level1 |
| capmeme_no_kg | ✓ | ✓ | ✓ | ✓ | ✗ (fallback) | Level1 |
| capmeme_no_emotion | ✓ | ✓ | ✓ | ✗ | ✗ (fallback) | Level1 |
| capmeme_concat_fusion | ✓ | ✓ | concat(E_v,E_t) not F_contrast | ✓ | ✗ (fallback) | Level1 |
| text_only | ✗ | ✓ | ✗ | ✗ | ✗ | Level1 |
| image_only | ✓ | ✗ | ✗ | ✗ | ✗ | Level1 |
| late_fusion | ✓ | ✓ | ✗ | ✗ | ✗ | Level1 |

---

## 6. Loss Function

- For models **with** CAP and emotion supervision (capmeme, capmeme_no_kg, capmeme_concat_fusion, and capmeme_no_emotion only for the BCE part):
  - **L = λ₁ · L_BCE + λ₂ · L_affect**
  - **L_BCE:** Binary cross-entropy with logits vs **Level1** (sarcasm).
  - **L_affect:** MSE(A_v, emotion_target) + MSE(A_t, emotion_target), where emotion_target is the multi-hot from Level 3–5.
- For **capmeme_no_emotion:** λ₂ = 0, so only L_BCE.
- For **text_only, image_only, late_fusion:** Only L_BCE (no A_v, A_t).
- Default: `bce_weight=1.0`, `affect_weight=0.5` (overridable in training script).

---

## 7. Training and Evaluation

- **Optimizer:** AdamW.  
- **Scheduler:** CosineAnnealingLR over epochs.  
- **Best model:** Selected by **validation binary F1**; checkpoint saved (e.g. `outputs/<run_name>_best.pt`).  
- **Final evaluation:** On the **test set**; metrics saved to JSON (e.g. `outputs/<run_name>_metrics.json`).  
- **Metrics reported:** Accuracy, Macro F1, Weighted F1, **Binary F1** (sarcasm class), Precision, Recall, ROC-AUC, PR-AUC, and confusion matrix.  
- **Seeds:** We run each model with multiple seeds (e.g. 42, 123, 456) and can report mean ± std for paper.

---

## 8. File Structure and Role of Each File

| File | Role |
|------|------|
| **EMOFF_MEME.csv** | Dataset: one row per meme; columns Name, text, Level1, Level 3–5 emotions. |
| **my_meme_data/my_meme_data/** | Folder containing PNG images; filenames must match `Name` in CSV. |
| **dataset.py** | PyTorch `MemeDataset`: loads CSV + images, builds emotion vocab from Level 3–5, multi-hot emotion target, CLIP processor for images, text tokenizer; optional KG embedding per sample. |
| **model.py** | Defines CAPMeme, KGModule, and all 7 model variants (including baselines); `build_model(model_name, ...)` returns the right model. |
| **loss.py** | `joint_capmeme_loss`: BCE for sarcasm + optional MSE for A_v, A_t vs emotion target. |
| **data_utils.py** | `filter_missing_images`, `stratified_split`, `save_splits`, `load_splits` for reproducible train/val/test. |
| **metrics.py** | `compute_metrics`: accuracy, F1 variants, precision, recall, ROC-AUC, PR-AUC, confusion matrix. |
| **kg_extractor.py** | ConceptNet API (Hindi): fetch concept labels, build E_kg (hash-based, fixed dim). |
| **train.py** | Single-entry training: loads data (with optional filter_missing and splits file), builds model by name, trains, evaluates on val/test, saves best checkpoint and test metrics JSON. |
| **run_all.py** | Runs multiple models × multiple seeds; first run can save splits, later runs reuse same splits; writes `run_all_summary.json`. |
| **requirements.txt** | Python dependencies (torch, transformers, pandas, scikit-learn, etc.). |
| **project.md** | High-level CAPMeme architecture and pipeline description. |
| **RESEARCH_PLAN_TOP_JOURNAL.md** | Research plan: venues, baselines, ablations, metrics, paper structure. |
| **PROJECT_EXPLANATION_FOR_MENTOR.md** | This document: full project explanation for mentor. |

---

## 9. How to Run

### 9.1 Single run (one model, one seed)

```bash
cd /path/to/Memes_work
python3 train.py --csv EMOFF_MEME.csv --image_dir my_meme_data/my_meme_data \
  --model capmeme_no_kg --seed 42 --filter_missing --epochs 10 --output_dir outputs
```

- **--filter_missing:** Drops rows whose image file is missing (recommended).  
- **--text_model:** Default `bert-base-multilingual-cased`; use `ai4bharat/indic-bert` if you have Hugging Face access.  
- **--use_kg:** Only for `--model capmeme` if you want ConceptNet (slower).

### 9.2 Full pipeline (all 7 models × 3 seeds, same splits)

```bash
python3 run_all.py --filter_missing --save_splits outputs/splits.json --epochs 10 --output_dir outputs
```

- First run creates and saves splits; subsequent runs in the same invocation reuse them.  
- Results: `outputs/<model>_seed<seed>_best.pt`, `outputs/<model>_seed<seed>_metrics.json`, and `outputs/run_all_summary.json`.

---

## 10. Outputs Produced

| Output | Description |
|--------|-------------|
| **outputs/splits.json** | Train/val/test indices (optional; for reproducibility). |
| **outputs/<run_name>_best.pt** | Best model state dict (by val F1) for that run. |
| **outputs/<run_name>_metrics.json** | Test-set metrics: accuracy, macro_f1, weighted_f1, binary_f1, precision, recall, roc_auc, pr_auc, confusion_matrix, etc. |
| **outputs/run_all_summary.json** | Summary of all runs (model, seed, status, and if available test_binary_f1, test_macro_f1). |

---

## 11. Ground Truth Summary (for mentor)

- **Primary ground truth:** **Level1** in `EMOFF_MEME.csv` — binary sarcasm (0/1). This is the only label used for the main task and for all baselines.  
- **Auxiliary ground truth:** **Level 3, 4, 5** — three emotion labels per sample. Combined into one **multi-hot emotion vector** (vocabulary built from all non-`"none"` emotions in the dataset). Used only to supervise the **affect (CAP)** part of CAPMeme (A_v and A_t).  
- **No ground truth:** ConceptNet is used as an optional external resource (no separate labels); when ConceptNet is not used, the model uses a learned fallback from E_t.

---

## 12. Dependencies (high level)

- **torch**, **torchvision** — models and data.  
- **transformers** — CLIP, BERT/text encoder, tokenizers.  
- **pandas** — CSV and data handling.  
- **Pillow** — image loading.  
- **scikit-learn** — stratified split, metrics (F1, accuracy, etc.).  
- **requests** — ConceptNet API (when `use_kg=True`).  
- **numpy** — arrays and KG embedding construction.

---

This document, together with `project.md` and `RESEARCH_PLAN_TOP_JOURNAL.md`, gives a complete picture of the project for your mentor: **what** (task, dataset, ground truth), **which** (models and baselines), **how** (loss, training, evaluation, code layout), and **how to run** (single run and full pipeline).
