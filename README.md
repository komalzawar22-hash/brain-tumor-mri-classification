# Brain Tumor MRI Classification

Deep-learning classifier that sorts brain MRI slices into four classes: **glioma, meningioma, pituitary tumor, and no tumor**.

> Educational / research project. It is **not** a medical device and must not be used for diagnosis.

## Approach
- **Model:** VGG16 pre-trained on ImageNet used as a feature extractor, plus a small dense classification head
- **Training:** two stages: (1) train the head with the base frozen, (2) fine-tune the last VGG16 block at a lower learning rate
- **Data handling:** augmentation (rotation, shifts, zoom, flip) on training data only; validation split taken from `Training/`; `Testing/` kept untouched for final evaluation
- **Evaluation:** classification report, confusion matrix, one-vs-rest ROC curves
- **Inference:** `predict_image()` helper gives the predicted class and confidence for one image

## Dataset
A public brain-tumor MRI dataset with `Training/` and `Testing/` folders and one sub-folder per class (`glioma`, `meningioma`, `notumor`, `pituitary`).
Dataset link: _add the Kaggle/source link here_

Expected layout:
```
brain_scans/
  Training/{glioma,meningioma,notumor,pituitary}/
  Testing/{glioma,meningioma,notumor,pituitary}/
```

## How to run
1. Download the dataset and put it somewhere reachable (local folder or Google Drive).
2. Open `brain_tumor_vgg16.ipynb` (Google Colab with a GPU runtime is the easiest way).
3. Set `DATA_DIR` in the configuration cell and run all cells.

Local setup: `pip install -r requirements.txt`

## Results
_Run the notebook to reproduce the metrics; results will be added here (test accuracy, per-class precision/recall, confusion matrix)._

## Limitations
- Trained on a single public dataset; behaviour on other scanners or hospitals is unknown.
- Depending on how the dataset was split, slices from the same patient may appear in both train and test sets, which can inflate scores.

## Stack
Python, TensorFlow/Keras, scikit-learn, NumPy, Matplotlib, Seaborn
