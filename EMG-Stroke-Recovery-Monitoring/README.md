# EMG Stroke Recovery Monitoring

This repository contains a modular Python project for processing EMG data from healthy and post-stroke individuals, extracting features, training classification models, and visualizing model performance.

## Project Structure

- `main.py` - Entry point for running the full analysis pipeline.
- `emg_stroke_recovery_monitoring/`
  - `data.py` - EMG data loading, filtering, segmentation, and raw/feature preparation.
  - `features.py` - Time-domain and frequency-domain feature extraction.
  - `models.py` - SVM, MLP, and 1D-CNN model definitions and training helpers.
  - `utils.py` - Plotting and result visualization utilities.
  - `__init__.py` - Package exports.
- `requirements.txt` - Python package dependencies.

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the analysis from the repository root:

```bash
python main.py --dataset-root "EMG_Reaching_Healthy_Stroke" --output model_comparison.png
```

If your dataset is stored in a different location, set `--dataset-root` to the folder containing `Health_reaching` and `Stroke_reaching`.

## Notes

- The project assumes each subject folder contains `Target *.csv` files and that the EMG dataset structure matches the original project layout.
- The 1D-CNN model operates on raw segmented EMG windows, while the SVM and MLP use extracted features.
- The output plot is saved as `model_comparison.png` by default.

