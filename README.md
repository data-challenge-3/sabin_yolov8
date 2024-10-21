# sabin_yolov8n

## 1. Project Overview
## 2. Prerequisites
## 3. Data Preprocessing
## 4. Data Augmentation
## 5. YOLOv8n Training
## 6. Predictions

## 1. Project Overview
The goal of this project is to use YOLOv8 (You Only Look Once version 8) to detect and count fish in images and videos. The project uses a pre-trained YOLOv8 model, and includes custom training with a fish dataset. Preprocessing and data augmentation techniques are applied to improve model generalization and performance.

## 2. Prerequisites
**Libraries**
- Python 3.x
- TensorFlow
- OpenCV
- ultralytics (for YOLOv8)
- Pandas
- Matplotlib
- TQDM (for progress bars)

- ## 3. Data Preprocessing
- The dataset consists of images and corresponding masks, which mark the fish locations. The dataset is loaded and visualized to verify the file structure.
```
import os
import pandas as pd

#Set directories
base_dir = 'C:/Users/Lenovo/PycharmProjects/DeepFish/DeepFish/Alternatives to given model'
images_dir = os.path.join(base_dir, 'images')
masks_dir = os.path.join(base_dir, 'masks')

#Load CSV
train_csv = os.path.join(base_dir, 'train.csv')
train_df = pd.read_csv(train_csv)
```

## 4. Data Augmentation
**TensorFlow Data Augmentation**
To improve model performance, TensorFlow's augmentation is applied. The data is augmented with random transformations (rotations, flips, etc.) to increase the diversity of the training set.

Path to augmented data: C:/Users/Lenovo/PycharmProjects/DeepFish/DeepFish/Alternatives to given model/augmented_training_images

## 5. YOLOv8n Training
```
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='path/to/data.yaml', 
    epochs=100,
    imgsz=360,
    batch=8,
    device='cpu'  # Or 'gpu' if available
)
```

## 6. Predictions
Making Predictions on Augmented Data
Using the trained YOLOv8 model, predictions are made on a set of test images (limited to 10 images for demonstration).
```
# Load trained model
model = YOLO('trained_model.pt')

# Loop through test images and make predictions
for filename in os.listdir(test_images_dir):
    predictions = model.predict(source=image_path, save=True, save_txt=True)
```
