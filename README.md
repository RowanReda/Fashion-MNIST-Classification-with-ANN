# Fashion MNIST Classification with Artificial Neural Network (ANN)

## Objective
This project aims to build an Artificial Neural Network (ANN) model using Keras to classify images from the Fashion MNIST dataset. The model will distinguish between 10 different categories of clothing and footwear, reaching an expected accuracy of around 89%.

## Dataset
The Fashion MNIST dataset consists of 70,000 grayscale images in 10 categories, with each image being 28x28 pixels. The categories include:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Project Structure
- `fashion_mnist_classification.ipynb`: Main Jupyter Notebook containing the code for data loading, preprocessing, model creation, training, evaluation, and visualization.
- `README.md`: This file, containing project overview, setup, and usage instructions.
- `requirements.txt`: List of dependencies required to run the notebook.

## Dependencies
To run this project, install the dependencies using the following command:

pip install -r requirements.txt
## Instructions for Running the Code
- Clone this repository:
git clone https://github.com/your-username/fashion-mnist-classification.git
- Navigate to the project directory:
cd fashion-mnist-classification
- Install the required libraries:
pip install -r requirements.txt
- Launch the Jupyter Notebook:
jupyter notebook fashion_mnist_classification.ipynb
- Run each cell in the notebook to train the model, evaluate its performance, and visualize the results.

## Model Architecture
The model consists of:
- Input layer: 784 nodes (flattened 28x28 image)
- Hidden layers: A series of Dense layers with ReLU activation, Batch Normalization, and Dropout for regularization
- Output layer: 10 nodes with softmax activation to classify the images into 10 categories

## Expected Results
The model is expected to achieve an accuracy of around 89% on the test dataset. You’ll also see visualizations of:
- Training and validation loss curves
- Confusion matrix
- Classification report (Precision, Recall, F1-Score) for each category
