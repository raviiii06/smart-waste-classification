# Smart Waste Classification using Deep Learning

A deep learning project that classifies images of waste into categories (e.g. plastic,
paper, metal, glass, cardboard, organic, trash) using transfer learning, with a Flask
web app for live inference and disposal guidance.

## Project structure

```
smart-waste-classification/
├── data/
│   └── garbage-classification/   # dataset goes here (one subfolder per class)
├── src/
│   ├── config.py                 # all hyperparameters and paths
│   ├── preprocess.py             # dataset loading + augmentation
│   ├── model.py                  # transfer-learning model architecture
│   ├── train.py                  # training script (2-phase: head + fine-tune)
│   └── evaluate.py               # classification report + confusion matrix
├── app/
│   ├── app.py                    # Flask inference app
│   ├── templates/index.html
│   └── static/uploads/
├── models/                       # trained model + labels + plots land here
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the dataset

Download the Kaggle "Garbage Classification" dataset (12 classes):

```bash
pip install kaggle
# place your kaggle.json API token in ~/.kaggle/
kaggle datasets download -d mostafaabla/garbage-classification
unzip garbage-classification.zip -d data/garbage-classification
```

After unzipping you should have:
```
data/garbage-classification/
    battery/
    biological/
    cardboard/
    ...
```

If you use a different dataset (e.g. TrashNet, 6 classes), just point
`DATA_DIR` in `src/config.py` at it — the training script automatically
detects class names from the folder names, so no code changes are needed.

## 3. Train the model

```bash
python -m src.train
```

This runs:
- **Phase 1** — trains a new classification head on top of a frozen pretrained
  backbone (EfficientNetB0 by default — change in `config.py`)
- **Phase 2** — unfreezes the top ~30 layers and fine-tunes at a low learning rate

Outputs (in `models/`):
- `waste_classifier.h5` — the trained model
- `labels.json` — class index → class name mapping
- `training_history.png` — accuracy/loss curves

Typical training time: 15–40 min on a free Colab GPU, depending on dataset size.
**If you don't have a local GPU, run `train.py` in Google Colab** and download
the resulting `models/` folder back into this project before running the app.

## 4. Evaluate

```bash
python -m src.evaluate
```

Prints a precision/recall/F1 classification report and saves a confusion
matrix to `models/confusion_matrix.png` — good material for your report.

## 5. Run the web app

```bash
python app/app.py
```

Open `http://127.0.0.1:5000`, upload a photo of an item, and get:
- predicted category + confidence
- top-3 predictions
- a disposal tip for that category

## 6. Customizing

- **Change backbone**: edit `BASE_MODEL_NAME` in `src/config.py`
  (`EfficientNetB0`, `MobileNetV2`, or `ResNet50`) — useful if you want a
  "comparative study of architectures" section in your report.
- **Change classes**: just point at a differently-structured dataset folder;
  everything else adapts automatically.
- **Add disposal tips**: extend the `DISPOSAL_TIPS` dict in `app/app.py`.

## Suggested report sections

1. Introduction & motivation (waste management problem, manual sorting issues)
2. Literature review (prior CNN-based waste classification work)
3. Dataset description (source, classes, size, sample images)
4. Methodology (preprocessing, augmentation, transfer learning approach)
5. Model architecture diagram
6. Training process (2-phase fine-tuning, hyperparameters)
7. Results (accuracy/loss curves, confusion matrix, classification report)
8. Web app demo (screenshots of the Flask UI)
9. Conclusion & future work (e.g. object detection instead of classification,
   deployment as a mobile app, real-time camera feed)
