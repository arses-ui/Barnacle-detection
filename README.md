# Barnacle-detection
## Description
This project implements an object detection model using YOLOv5 to identify and count barnacles in images. The model is trained on a custom dataset to enhance detection accuracy and generalization.
## Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/barnacle-detection.git
   cd barnacle-detectio
2. Set up a virtual environment:
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate  # On Windows
3. Install required packages
    pip install -r requirements.txt

## Usage Instructions 

1. Prepare the dataset in the required format:
     Organize images in images/train and images/val, with corresponding labels in labels/train and labels/val.
2. Adjust the configuration file:
     Edit barnacles.yaml to point to the correct folders.
3. Train the model:
      python train.py --img 640 --batch 16 --epochs 50 --data barnacles.yaml --weights yolov5s.pt --cache
4. Validate the model:
      python val.py --weights runs/train/exp/weights/best.pt --data barnacles.yaml --img 640
5. Perform inference:
      python detect.py --weights runs/train/exp/weights/best.pt --img 640 --source path/to/test/images (whatever the location is on your PC)
   ## Project Progress
1. Initial Setup and Dataset Creation : Collected images of barnacles and annotated them for training.Used images from the internet and ones 
   provided in the challenge. 12 images including 1000+ annotations used for training sample, 1 image with 200+ annotations used for validation test 
   and 3 randomly selected images from the internet used for inference 
2. Dataset Organization : Created a  dataset with separate folders for training and validation images, along with corresponding labels.
3. Training Process : Trained YOLOv5 using pre-trained weights.
4. Validation Results : Evaluated the model, observing precision, recall, and mAP metrics; results were disappointing and showed room for improvement.
5. Inference Results: Evaluated the detection of Barnacles in unseen images; results were disappointing and showed room for improvement
6. Next Steps : Investigating data augmentations and hyperparameter tuning based on validation performance.
## Validation Results and inference
1. Precision : 0 
2. Recall : 0 
3. mAP : 0
4. Detection: none (confidence threshold: 0.001 and 0.30)

## Improvement areas
Initial results indicated that the model struggled with detecting barnacles. Further room for improvement in several areas, including:

1. Dataset Quality and Size: Expanding the training dataset with more diverse examples can help the model learn to generalize better to different conditions.
2. Data Augmentation: Implementing various augmentation techniques can increase the diversity of the training set, helping the model become more robust.
3. Annotation Quality: Given that I manually annotated the images, ensuring consistent and accurate annotations is critical for effective training. Future projects may benefit from using automated annotation tools or crowd-sourced annotations to reduce annotation time while maintaining quality.
## Conclusion and learning Process

Based on my experience, while training YOLOv5 is a powerful method for object detection, the approach I initially took faced challenges, particularly in terms of model performance and efficiency. The manual annotations were tedious and time-consuming, while the subsequent model performance outcomes suggest that more robust training approaches, larger datasets, and higher-quality annotations may lead to better results. Further investigation is required to identify any underlying issues with the prototype implementation that could have contributed to the zero-detection results.
Despite all the challenges, I embraced the ecperience as this was my first time ever training a model using a dataset. This experience was enriching as I learned a little about utilizing object detection architectures like YOLOv5, as well as Roboflow for dataset management. Through this project, I also learned how to make a Github repository and properly communicate my code and project on it. Overall, I had a very fun learning experience, especially while creating the dataset for training and validation. 
