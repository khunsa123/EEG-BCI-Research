# ⚡ Somatosensory Pain iEEG/ECoG Analysis (OpenNeuro ds002778)

This project analyzes the **ds002778** open‑access dataset from OpenNeuro, containing **intracranial EEG (iEEG/ECoG)** recordings during **somatosensory pain stimulation**.  
The dataset includes high‑resolution cortical signals recorded from subdural electrodes while participants received **painful and non‑painful somatosensory stimuli**.

This project focuses on:
- Preprocessing iEEG/ECoG signals  
- Extracting pain‑evoked cortical responses  
- Time–frequency analysis of nociceptive processing  
- Machine learning classification of pain vs non‑pain trials  
- Visualization of cortical activation patterns  

This dataset complements the **laser‑evoked EEG project (ds002338)** and strengthens the multimodal pain‑neurophysiology portfolio.

---

## 📌 Dataset Overview

**Dataset:** OpenNeuro *ds002778*  
**Modality:** iEEG / ECoG  
**Task:** Painful vs non‑painful somatosensory stimulation  
**Subjects:** Multiple neurosurgical patients  
**Format:** BIDS‑compliant  
**Size:** ~200–300 MB (ideal for Google Colab)

Each recording includes:
- Subdural grid/strip electrodes  
- High‑sampling‑rate iEEG  
- Pain vs non‑pain event markers  
- Anatomical electrode coordinates  

---

## 📂 Project Structure

Somatosensory_Pain_iEEG/
│
├── notebooks/
│   ├── 01_download_and_preprocess.ipynb
│   ├── 02_erp_and_timefreq_analysis.ipynb
│   ├── 03_ml_classification.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── ml_models.py
│   ├── visualization.py
│
├── data/                 # downloaded automatically (ignored by git)
├── README.md
└── requirements.txt


---

## 🧪 Methods

### **1. iEEG/ECoG Preprocessing**
- Load BIDS dataset with MNE‑Python  
- Bandpass filter (1–150 Hz)  
- Notch filter (50/60 Hz)  
- Re‑reference (common average or bipolar)  
- Epoch around pain stimuli (−0.2 to 0.8 s)  
- Artifact rejection  

### **2. Evoked Response Analysis**
- Extract early somatosensory evoked potentials (SEPs)  
- Compare pain vs non‑pain cortical responses  
- Plot:
  - Evoked potentials  
  - Butterfly plots  
  - Electrode‑wise responses  

### **3. Time–Frequency Analysis**
- Morlet wavelets  
- High‑gamma (70–150 Hz) pain‑related activity  
- Broadband gamma bursts  
- Pain‑specific oscillatory signatures  

### **4. Machine Learning Classification**
- Feature extraction:
  - Peak amplitudes  
  - Latencies  
  - High‑gamma power  
  - Broadband spectral features  
- Models:
  - Logistic Regression  
  - SVM  
  - Random Forest  
- Evaluation:
  - Accuracy  
  - ROC‑AUC  
  - Confusion matrix  

---

## ▶️ Quick Start (Google Colab)

### Install dependencies

```python
!pip install mne openneuro-py numpy scipy matplotlib scikit-learn

Download dataset (one subject only)

from openneuro import download

download(
    dataset="ds002778",
    target_dir="/content/ds002778",
    include=["sub-01"]
)

📊 Expected Results
Clear somatosensory evoked responses

Strong high‑gamma activation during painful stimuli

ML classifier accuracy typically 75–90%

Distinct cortical signatures of nociceptive processing

📝 Notes
iEEG/ECoG provides high‑spatial‑resolution pain biomarkers

Dataset is small enough for Colab

Perfect complement to laser‑evoked EEG (ds002338)

Strengthens multimodal pain‑neurophysiology research portfolio

👩‍🔬 Author
Khunsa Iftikhar  
Computational Neuroscience & AI Researcher
Multimodal Neurophysiology & Machine Learning
