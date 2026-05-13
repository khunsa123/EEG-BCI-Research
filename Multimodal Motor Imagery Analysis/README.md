# 🧠 EEG‑fMRI Neurofeedback (XP2 Dataset) — Multimodal Motor Imagery Analysis

This project analyzes the **XP2 simultaneous EEG‑fMRI neurofeedback dataset**, which includes:
- 64‑channel EEG recorded inside the MRI scanner  
- fMRI BOLD activity from motor cortex  
- Neurofeedback (NF) scores derived from EEG and fMRI  
- Motor imagery (MI) and neurofeedback (1D vs 2D) tasks  

The goal of this project is to build a **multimodal neurofeedback analysis pipeline**, including:
- EEG preprocessing  
- NF score extraction and analysis  
- fMRI activation analysis  
- EEG‑fMRI feature integration  
- Machine learning models to classify MI vs Rest and NF performance  

This project fits naturally into the **Multimodal‑NeuroPhysio‑Signal‑Research** repository and demonstrates advanced multimodal signal processing.

---

## 📌 Dataset Overview

**Dataset:** XP2 (OpenNeuro ds002336 + XP2 extension)  
**Modalities:** EEG (64‑channel), fMRI (EPI), NF scores  
**Subjects:** 16 participants  
**Tasks:**  
- Motor Imagery (MIpre, MIpost)  
- Neurofeedback (1D NF or 2D NF)  
- 20‑second block design (Rest ↔ Task)  

### EEG
- 64‑channel BrainProducts MR‑compatible EEG  
- Sampled at 5 kHz  
- Includes gradient artifact markers (R128)  
- Includes block markers (S99 = Rest, S2 = Task)  
- Preprocessed version available (200 Hz, 8–30 Hz filtered)

### fMRI
- 3T Siemens Verio  
- TR = 1 s  
- 16 slices (motor cortex coverage)  
- Preprocessed NF scores available (ROI‑based SMA/M1 activation)

### Neurofeedback Scores
- EEG NF: Laplacian C3 ERD (8–30 Hz)  
- fMRI NF: SMA/M1 activation difference  
- Provided as `.mat` files per subject  

---

## 📂 Project Structure

XP2_EEG_fMRI_Neurofeedback/

│

├── notebooks/

│   ├── 01_load_and_preprocess_EEG.ipynb

│   ├── 02_NF_score_analysis.ipynb

│   ├── 03_fMRI_activation_analysis.ipynb

│   ├── 04_multimodal_ML_classification.ipynb

│

├── src/

│   ├── eeg_preprocessing.py

│   ├── nf_features.py

│   ├── fmri_processing.py

│   ├── multimodal_ml.py

│   ├── visualization.py

│

├── data/                     # downloaded manually or via script

├── README.md

└── requirements.txt

---

## 🧪 Methods

### **1. EEG Preprocessing**
- Load BrainVision `.vhdr` files  
- Remove MRI gradient artifact (template subtraction)  
- Remove ballistocardiogram (BCG) artifact  
- Downsample to 200 Hz  
- Bandpass filter (1–40 Hz)  
- Epoch into Rest vs Task blocks  
- Extract ERD/ERS features (8–30 Hz)  
- Compute Laplacian C3 features (motor cortex)

### **2. Neurofeedback Score Analysis**
- Load EEG NF scores (`lapC3_ERD`)  
- Load fMRI NF scores (`NF_bold`)  
- Smooth and normalize NF time series  
- Compare 1D vs 2D NF performance  
- Correlate EEG NF with fMRI NF  
- Block‑wise NF learning curves  

### **3. fMRI Integration**
- Load preprocessed fMRI NF scores  
- Extract SMA/M1 activation  
- Compute ROI‑based features  
- Align fMRI NF with EEG NF in time  
- Build multimodal NF predictors  

### **4. Machine Learning Models**
- Classify Rest vs MI  
- Predict NF performance from EEG  
- Predict NF performance from EEG + fMRI  
- Models:
  - Logistic Regression  
  - Random Forest  
  - SVM  
  - Multimodal Fusion (early + late fusion)

---

## ▶️ Quick Start (Google Colab)

### Install dependencies

```python
!pip install mne scipy numpy matplotlib scikit-learn h5py
```
```bash
Upload XP2 data to Colab or mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```
---

## 📊 Expected Results

- Clear ERD/ERS patterns during MI

- Strong NF learning curves in 1D vs 2D groups

- EEG NF correlates with fMRI NF

- ML models classify MI vs Rest with 80–90% accuracy

- Multimodal EEG+fMRI improves NF prediction

---

## 📝 Notes

- XP2 is a multimodal dataset requiring careful synchronization

- Preprocessed EEG is recommended for ML tasks

- fMRI NF scores are already extracted (no heavy fMRI processing needed)

- This project demonstrates multimodal integration, ideal for neurofeedback research

---

## 👩‍🔬 Author

Khunsa Iftikhar  
Computational Neuroscience & AI Researcher
Multimodal Neurophysiology & Machine Learning
