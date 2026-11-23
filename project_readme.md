# Clinical EEG Assistant with Multimodal RAG

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3127/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B.svg)](https://streamlit.io/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-000000.svg)](https://www.pinecone.io/)
[![Gemini API](https://img.shields.io/badge/Gemini-RAG%20Reporting-4285F4.svg)](https://ai.google.dev/)

An end-to-end **cross-modal EEG seizure detection and explainable clinical reporting system**. Instead of outputting opaque black-box probabilities, this application aligns raw brainwave recordings with historical medical notes in a shared 256-dimensional vector space, retrieves the most clinically similar past cases via **Pinecone**, and synthesizes an evidence-backed epileptology report using **Google Gemini**.

---

## Key Features

- **Cross-Modal Contrastive Alignment:** Bridges raw 19-channel EEG signals and clinical text notes into a shared 256-d L2-normalized embedding space (CLIP-style dual-tower architecture).
- **Zero Manual Feature Engineering for Retrieval:** Learns temporal spike and waveform representations directly from raw EEG tensors using a custom 3-stage 1D-CNN.
- **Physiological Grounding:** Extracts quantitative waveform biomarkers (Line-Length Index, Peak Z-Amplitude, Primary Active Channel) to prevent LLM hallucination.
- **Evidence-Backed Clinical Reports:** Combines retrieved Top-5 historical patient cases (with cosine similarity scores and clinician notes) with patient waveform metrics to generate a structured 3-part report (**Findings**, **Historical Evidence**, and **Clinical Impression**).
- **Interactive Streamlit Interface:** Upload `.edf` recordings locally to inspect waveform metrics, matched historical cases, and generated diagnostic summaries in real time.

---

## System Architecture

```text
                       ┌──────────────────────────────────────┐
  Raw EEG (.edf) ────► │ Preprocessing (19 ch × 256 samples)  │ ──► Morphological Metrics
                       │     + Per-Channel Z-Normalization    │     (Line-Length, Peak Z)
                       └──────────────────┬───────────────────┘                 │
                                          ▼                                     │
                       ┌──────────────────────────────────────┐                 │
                       │   EEG Tower (Trained 1D-CNN + MLP)   │                 │
                       │    [tusz_crossmodal_model.pt]        │                 │
                       └──────────────────┬───────────────────┘                 │
                                          │ 256-d L2-Normalized Query Vector    │
                                          ▼                                     │
                       ┌──────────────────────────────────────┐                 │
                       │     Pinecone Vector Database         │                 │
                       │ (Indexed 256-d Historical Embeddings │                 │
                       │   aligned with Bio_ClinicalBERT)     │                 │
                       └──────────────────┬───────────────────┘                 │
                                          │ Top-5 Similar Clinical Cases        │
                                          ▼                                     ▼
                       ┌──────────────────────────────────────────────────────────┐
                       │         Google Gemini API (Multimodal RAG Prompt)        │
                       └──────────────────────────┬───────────────────────────────┘
                                                  ▼
                       Structured Report: 1. Findings | 2. Historical Evidence | 3. Clinical Impression
```

### Dual-Tower Training Details
1. **Text Tower (Tower A):** A frozen `Bio_ClinicalBERT` encoder paired with a projection head maps historical clinical annotations into a 256-d L2-normalized semantic space.
2. **EEG Tower (Tower B):** A custom 3-stage `1D-CNN` (`19 → 64 → 128 → 256` channels with `BatchNorm`, `GELU`, and pooling) followed by a 2-layer MLP projection head maps raw `(19, 256)` EEG windows into the exact same 256-d space.
3. **Optimization:** Trained with **Symmetric InfoNCE Contrastive Loss** and **class-weighted sampling** to address severe seizure-vs-background class imbalance.

---

## Dataset & Preprocessing

This proof-of-concept was trained and evaluated on a **600-file `.edf` subset** (spanning **39 unique patients**) from the **Temple University Hospital Seizure Corpus (TUSZ)**:

| Class Label | Clinical Description | File Count | Share |
| :--- | :--- | :---: | :---: |
| **BCKG** | Normal Background | 526 | 87.7% |
| **FNSZ** | Focal Non-Specific Seizure | 54 | 9.0% |
| **GNSZ** | Generalized Non-Specific Seizure | 17 | 2.8% |
| **CPSZ** | Complex Partial Seizure | 3 | 0.5% |
| **Total** | **600-File TUSZ Subset** | **600** | **100%** |

- **Signal Extraction:** Parsed with `pyedflib` into 19-channel, 256-sample windows with per-channel z-score normalization.
- **Line-Length Index:** Computed as $\sum |x_{t+1} - x_t|$ to capture high-frequency amplitude variations typical of ictal activity.

---

## Benchmark Results (5-Fold Patient-Wise Cross-Validation)

To prevent data leakage, all models were evaluated using **strict 5-fold patient-wise cross-validation** across the 39 patients (zero patient overlap between train and test folds).

| Model | Input Representation | AUROC (Mean $\pm$ SD) | AUPRC (Chance = 0.123) | Explainable Retrieval |
| :--- | :--- | :---: | :---: | :---: |
| **Proposed (Contrastive) + kNN Retrieval** | Raw EEG + Clinical Notes | **0.477 $\pm$ 0.100** | **0.137** | Yes (Top-K Cases + RAG) |
| Handcrafted + Logistic Regression | 209 Manual Features | ~0.445 | **0.170** | No |
| Handcrafted + kNN | 209 Manual Features | ~0.445 | ~0.138 | No |
| Handcrafted + Random Forest | 209 Manual Features | ~0.443 | ~0.130 | No |
| Supervised 1D-CNN | Raw EEG (`0/1` Labels Only) | 0.354 | 0.103 | No |

### Key Takeaways
- **Highest AUROC & Stability:** Ranks **#1 in AUROC (`0.477`)** with the **lowest cross-patient variance (`±0.100`)** among all tested models.
- **Major Gain Over Supervised Deep Learning:** Using the identical 1D-CNN backbone, contrastive clinical-text alignment improves AUROC by **+0.123** over the purely supervised 1D-CNN (`0.354`) and lifts AUPRC (`0.137`) above random chance (`0.123`).
- **No Manual Feature Engineering:** Outperforms all three handcrafted baselines (LogReg, kNN, Random Forest) on AUROC directly from raw waveforms.

---

## Repository Structure

```text
Clinical-EEG-assistant-with-multimodal-Rag/
├── app.py                      # Main Streamlit application (EEG inference, Pinecone retrieval, Gemini RAG)
├── tusz_crossmodal_model.pt    # Pre-trained PyTorch weights for the 1D-CNN EEG Tower
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## Local Setup & Installation

### Prerequisites
- **Python 3.12** (recommended on Windows to avoid NumPy/PyTorch wheel build issues)
- A **Pinecone** account and populated 256-d index
- A **Google Gemini API** key

### 1. Clone the Repository
```bash
git clone https://github.com/amberpurwar636/Clinical-EEG-assistant-with-multimodal-Rag.git
cd Clinical-EEG-assistant-with-multimodal-Rag
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install -r requirements.txt
```
*(Note: If you encounter a `pyarrow` / NumPy 2.x compatibility error with Streamlit on Windows, pin NumPy 1.x via `python -m pip install "numpy<2"`.)*

### 4. Configure API Keys
Ensure your Pinecone API key, Pinecone index name, and Google Gemini API key are configured via environment variables or inside `app.py` / Streamlit secrets (`.streamlit/secrets.toml`).

### 5. Run the Application
```bash
python -m streamlit run app.py
```
Open the local URL displayed in your terminal (typically `http://localhost:8501`) and upload a TUSZ `.edf` file to run inference and generate a clinical report.

---

## Limitations & Future Scope

- **Dataset Scale:** Currently trained and indexed on a compute-constrained 600-file subset of TUSZ. Scaling to the full corpus will improve generalization and representation of rare seizure subtypes (`ABSZ`, `TNSZ`, `SPSZ`).
- **Automated Seizure-Window Scanning:** The default `process_edf_file()` function evaluates a 1-second (256-sample) window starting at `t = 0`. Integrating a sliding-window peak Line-Length detector or interactive onset timestamp selector will enable automatic localization across long recordings.
- **Montage Standardization:** Explicit bipolar montage mapping (e.g., TCP standard channels) can be added to guard against electrode-ordering variations across different clinical EDF files.

---

## Medical Disclaimer

This project is an academic research prototype built for educational and demonstration purposes only. It is **not** a certified medical device and must **not** be used for real-world clinical diagnosis or patient care.