# Plant Disease Detection Using Deep Learning and Transfer Learning

## Overview
This project implements an automated plant disease detection system using EfficientNet-B0 with transfer learning. The model classifies 38 categories of plant diseases and healthy leaves across 14 crop species, achieving 99.65% validation accuracy on the PlantVillage dataset. GradCAM visualizations confirm the model focuses on disease-relevant leaf regions rather than background artifacts.

## Author
Nikhil Jaswaraj Karkera
Student ID: 111369531
University of Colorado Boulder

## Dataset
The PlantVillage dataset contains 54,306 RGB images across 38 classes spanning 14 crop species including apple, blueberry, cherry, corn, grape, orange, peach, pepper, potato, raspberry, soybean, squash, strawberry, and tomato. The dataset is publicly available on Kaggle: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

## Project Structure
```
├── Plant_Disease_Detection.ipynb    # Main notebook with all code
├── best_model.pth                   # Trained model weights
├── images/                          # Generated plots and figures
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── augmentation_examples.png
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   ├── misclassified_images.png
│   └── gradcam_results.png
├── Report.pdf                       # Final project report
├── Presentation.pptx                # Slide deck
└── README.md
```

## Requirements
- Python 3.8+
- PyTorch
- TorchVision
- Albumentations
- grad-cam
- Matplotlib
- scikit-learn
- Pillow

Install all dependencies:
```
pip install torch torchvision albumentations grad-cam matplotlib scikit-learn pillow
```

## How to Run
1. Open the notebook in Google Colab
2. Set runtime to T4 GPU (Runtime → Change runtime type → T4 GPU)
3. Upload the PlantVillage dataset zip to Google Drive
4. Update the ZIP_PATH variable in Cell 2 to point to your zip file
5. Run all cells sequentially

## Methodology
- **Model:** EfficientNet-B0 pretrained on ImageNet, fine-tuned end-to-end
- **Classification Head:** Dropout(0.2) → Linear(38)
- **Optimizer:** AdamW (lr=0.001, weight_decay=0.0001)
- **LR Schedule:** Cosine Annealing over 10 epochs
- **Loss:** CrossEntropyLoss
- **Augmentations:** Horizontal flip, rotation (±15°), color jitter
- **Input Size:** 224×224 RGB

## Results
- **Validation Accuracy:** 99.65% (38 misclassifications out of 10,862 images)
- **Weighted F1-Score:** 0.9965
- **Perfect Classes:** 22 out of 38 achieved 1.0 precision, recall, and F1
- **Parameters:** 4,056,226 (all trainable)


## Limitations
- Dataset images are from controlled lab conditions; real-world field performance may differ
- Class imbalance (15:1 ratio between largest and smallest classes) was not explicitly addressed
- No separate held-out test set; evaluation was on the validation split

## References
1. Mohanty et al. (2016). Using deep learning for image-based plant disease detection. Frontiers in Plant Science.
2. Tan & Le (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. ICML.
3. Selvaraju et al. (2017). Grad-CAM: Visual explanations from deep networks. ICCV.
4. Hughes & Salathé (2015). An open access repository of images on plant health. arXiv:1511.08060.

## License
This project is for academic purposes as part of coursework at the University of Colorado Boulder.
