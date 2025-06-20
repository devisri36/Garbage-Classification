Garbage Image Classification

This project classifies waste images into 6 categories:
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

We use a Convolutional Neural Network (CNN) with transfer learning (MobileNetV2) for this image classification task.


 Week 1 - Dataset & Preprocessing

### Tasks Completed:
- Downloaded and extracted the garbage classification dataset from Kaggle.
- Organized the dataset into 6 labeled folders: `cardboard`, `glass`, `metal`, `paper`, `plastic`, and `trash`.
- Set up a Jupyter notebook (`garbage_classifier.ipynb`) in VS Code.
- Used TensorFlow's `ImageDataGenerator` to:
  - Normalize pixel values.
  - Resize all images to 224x224.
  - Split dataset into training (80%) and validation (20%) sets.
- Visualized sample images with class labels to confirm correct loading.
