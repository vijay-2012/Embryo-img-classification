# Embryo Quality Classification
This project contains a convolutional neural network model for classifying embryo images as either good or poor quality.

### Overview
The goal is to develop an automated system for assessing embryo quality from images, to help standardize the in vitro fertilization (IVF) treatment process. The model was trained on a small dataset of 84 labeled embryo images.

## Getting Started

To train the model, please follow these steps:

1. Clone this repository.

    ```bash
   git clone https://github.com/vijay-2012/Embryo-img-classification.git
   ```

2. Run convert.py to convert the image dataset to TFrecords.

    ```bash
   python scripts/convert.py Images/train scripts/process/ 0
   ```

    Keep the percentage of validation images as 0 because we set 15% for validation inside the code

    It will save converted TFrecords in the "process" directory.
3. Navigate to scripts/slim/run and run the following command to train the model.

    ```bash
   ./load_inception_v1.sh
   ```
   The results and model checkpoints will be saved in the "result" directory.

4. To test the trained model, run the following command.

    ```bash
   python ../predict.py v1 ../../result/ ../../../Images/test ../output.txt 2
   ```

   It will save the output in a text file, also provides the evaluation metrics (Accuracy, Specificity, Sensitivity, Confusion matrix).
