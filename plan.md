# CAPMeme: Research Plan for 


**Prepared from:** Full scan of `Memes_work` folder, `project.md`, `EMOFF_MEME.csv`, and `my_meme_data/my_meme_data` images.

---

## 1. What You Already Have

### 1.1 Dataset (EMOFF_MEME + Images)

| Item | Details |
|------|--------|
| **CSV** | `EMOFF_MEME.csv` — **7,416 samples** (after pandas load); **7,406 unique image names**; **7,376 images** present in `my_meme_data/my_meme_data` (**30 missing**, need to handle). |
| **Task** | Binary sarcasm detection: **Level1** (0 = non-sarcastic, 1 = sarcastic). |
| **Class balance** | **Imbalanced:** 5,632 non-sarcastic (0), 1,784 sarcastic (1) → ~76% / 24%. Important for metrics (F1, PR-AUC, balanced accuracy) and baselines. |
| **Rich annotations** | Level2; Level 3–5 **multi-label emotions** (18 emotion types, e.g. Joy, Rage, Envy, Neglect, Fear, Disappointment). |
| **Language** | **Hindi** meme text. |
| **Modalities** | **Image** (meme) + **Text** (caption/overlay). |

### 1.2 Codebase (Implemented)

- **dataset.py** — `MemeDataset`: CSV + images, CLIP processor, IndicBERT tokenizer, emotion vocab, multi-hot emotion targets, optional ConceptNet KG at load time.
- **model.py** — **CAPMeme**: CLIP Vision (`openai/clip-vit-base-patch32`) → E_v; IndicBERT (`ai4bharat/indic-bert`) → E_t; CAP (vis_affect, text_affect → A_v, A_t; F_contrast = |A_v − A_t|); KG module (ConceptNet or fallback from E_t); fusion → classifier → sarcasm.
- **loss.py** — Joint loss: BCE(sarcasm) + λ · MSE(A_v, emotion_target) + MSE(A_t, emotion_target).
- **kg_extractor.py** — ConceptNet API (Hindi), hash-based concept embeddings.
- **train.py** — Train/val split, AdamW, CosineAnnealingLR, F1/accuracy, gradient clipping, best model save by val F1.

---

---

## 3. Data Preparation (Must Do Before Experiments)

1. **Resolve 30 missing images**  
   - Either obtain the 30 missing PNGs so all CSV rows are usable, or **drop those rows** and report: "We use N = 7,386 samples after removing rows with missing images."  
   - Use a fixed seed and document the exact CSV/image list for reproducibility.

2. **Stratified splits**  
   - **Stratified train / validation / test** (e.g., 70–15–15 or 80–10–10) by **Level1** so that sarcasm ratio is preserved.  
   - Single fixed seed; report exact split sizes and, if possible, release split indices or script.

3. **Emotion vocabulary**  
   - Normalize emotion labels (e.g., lowercasing) so "Envy" and "envy" map to one token; remove "none" from the emotion vocab.  
   - Report final emotion set and multi-hot construction in the paper.

4. **Optional: cross-dataset**  
   - If you have or can obtain another **Hindi meme** (or Hindi multimodal sarcasm) dataset, even small, use it for **cross-dataset evaluation** or **domain adaptation** — strong plus for reviewers.

---

## 4. Models to Implement and Report (For the Paper)

### 4.1 Your Proposed Model (Main Contribution)

- **CAPMeme (full)**  
  - Vision: CLIP ViT-B/32 → E_v.  
  - Text: IndicBERT → E_t.  
  - CAP: A_v, A_t from vis_affect / text_affect; F_contrast = |A_v − A_t|; supervision from Level 3–5 multi-hot emotions.  
  - Optional KG: ConceptNet E_kg (or fallback from E_t).  
  - Fusion: [F_contrast; E_kg] → ReLU → classifier → sarcasm.  
  - Loss: L = λ₁·BCE + λ₂·(MSE(A_v, emotion) + MSE(A_t, emotion)).  
  - **Variants to report:** CAPMeme **w/ KG** and **w/o KG** (ablation).

### 4.2 Strong Baselines (Required for Top Venues)

Implement and compare against these; use **same data splits and seeds** for all.

| Model | Description | Purpose |
|-------|-------------|--------|
| **Text-only (IndicBERT)** | IndicBERT [CLS] → linear → sarcasm. | Text-only upper bound; shows value of image. |
| **Image-only (CLIP)** | CLIP vision pooler → linear → sarcasm. | Image-only upper bound; shows value of text. |
| **Late fusion (concat)** | Concat(E_v, E_t) → MLP → sarcasm. No CAP, no emotion. | Simple multimodal baseline. |
| **Early fusion / single encoder** | Only if you use a model that takes image+text together (e.g., some VLMs); otherwise skip. | Optional. |
| **CLIP dual encoder** | Use CLIP's image and text encoders; concat or contrastive-style fusion → classifier. | Strong multimodal baseline. |
| **IndicBERT + CLIP (concat, no CAP)** | Same encoders as CAPMeme, concat(E_v, E_t) → MLP → sarcasm, **no** affect module, **no** emotion supervision. | Direct ablation: value of CAP + emotion. |
| **Multilingual BERT / XLM-R** | Replace IndicBERT with `bert-base-multilingual` or `xlm-roberta-base`; same pipeline (concat with CLIP or with CAP). | Sensitivity to language-specific vs multilingual text encoder. |

### 4.3 SOTA-Oriented and Larger Models (If Compute Allows)

| Model | Description | Purpose |
|-------|-------------|--------|
| **ViLT / CLIP-ViL** | Vision–language pretrained model (if Hindi is supported or you use code-switched/transliterated input). | SOTA-style VLM baseline. |
| **BLIP / BLIP-2** | Same idea: VLMs that fuse image + text. | Strong comparison if applicable to Hindi. |
| **LLaVA / InstructBLIP (zero-shot)** | Use a VLM with a prompt: "Is this meme sarcastic? Yes/No." No fine-tuning. | Zero-shot baseline. |
| **Larger CLIP** | e.g., `openai/clip-vit-large-patch14` or `ViT-L/14`. | Ablation: impact of vision encoder size. |
| **Larger / better Hindi LM** | e.g., MuRIL, DevBERT, or newer Ai4Bharat models (if available). | Ablation: impact of text encoder. |

**Recommendation:** At minimum, report **CAPMeme (w/ and w/o KG)**, **IndicBERT-only**, **CLIP-only**, **Late fusion (concat)**, and **IndicBERT+CLIP concat (no CAP)**. Add 1–2 SOTA-style VLMs or larger encoders if feasible.

---

## 5. Experimental Protocol (What to Run and Report)

### 5.1 Main Results Table

- **Metrics:** Accuracy, **Macro F1**, **Weighted F1**, **Binary F1 (sarcasm class)**, Precision/Recall for class 1; optionally **PR-AUC** and **ROC-AUC** (good for imbalanced data).
- **Reporting:** Mean and std over **≥ 3 runs** (different seeds) for each model; same splits across all models.
- **Table:** Rows = models, columns = metrics; highlight best and second best; mark statistical significance (e.g., paired bootstrap or McNemar) if possible.

### 5.2 Ablations

- **CAPMeme:**  
  - w/ KG vs w/o KG.  
  - w/ emotion loss (λ₂ > 0) vs w/o (λ₂ = 0).  
  - F_contrast only vs concat(E_v, E_t) only (replace F_contrast in fusion) to show value of affective contrast.
- **Fusion:** Dimension of fusion hidden layer; kg_dim (e.g., 128 vs 256).
- **Encoders:** CLIP ViT-B/32 vs larger CLIP; IndicBERT vs MuRIL/multilingual BERT.

### 5.3 Analysis and Qualitative Results

- **Error analysis:** Where does CAPMeme fail? (e.g., subtle sarcasm, code-mixing, rare emotions.)
- **Emotion distribution:** Correlation between predicted A_v, A_t and gold Level 3–5; t-SNE or similar of A_v vs A_t.
- **Examples:** 2–3 success and 2–3 failure cases (image + text + prediction + gold).
- **KG impact:** When ConceptNet returns concepts vs fallback; performance on kg_valid=1 vs kg_valid=0 subsets.

### 5.4 Reproducibility

- **Code:** Public repo (anonymized for review); `requirements.txt`, exact versions.  
- **Data:** If you cannot release EMOFF_MEME, describe it precisely and offer to provide split indices or script.  
- **Seeds:** Fix and report (e.g., 42, 123, 456).  
- **Hyperparameters:** Batch size, epochs, learning rate, λ₁/λ₂, optimizer, scheduler — one table in paper or appendix.

---

---

- **Optional but recommended:** One multilingual BERT variant; one larger CLIP or one VLM (e.g., zero-shot) for "SOTA" comparison.

This plan positions your work for a **world-top NLP or multimodal journal/conference** with a clear contribution (contrastive affect + hierarchical emotion supervision + optional KG for Hindi meme sarcasm), reproducible setup, and strong comparative and ablation experiments.
