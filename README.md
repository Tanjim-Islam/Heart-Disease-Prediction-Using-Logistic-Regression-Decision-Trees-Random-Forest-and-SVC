# Heart Disease Prediction Using Logistic Regression, Decision Trees, Random Forest, and SVC

## Project Overview

This project explores how well various machine learning models can predict the likelihood of heart disease based on health indicators. The goal is to experiment with different models to see how much accuracy can be achieved, recognizing that the results may vary. The project includes data preprocessing, model training, and evaluation using algorithms such as Logistic Regression, Decision Tree, Random Forest, and SVM. Each model attempts to classify whether a person's heart is healthy or potentially defective.

## Features

- **Binary Classification**: Predicts if the heart is healthy (0) or defective (1).
- **Training and Testing**: Uses multiple models with a train-test split strategy.
- **Multiple Machine Learning Models**: Decision Tree, Random Forest, SVM, and Logistic Regression.

## Dataset

- **Heart Disease Dataset** (`heart.csv`): The dataset contains various health parameters to predict heart disease.

## Requirements

- **Python 3.10**
- **Pandas**
- **NumPy**
- **Scikit-learn**

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/Heart-Disease-Prediction.git
    cd Heart-Disease-Prediction
    ```

2. Install the necessary dependencies:

    ```bash
    pip install pandas numpy scikit-learn
    ```

3. Make sure the `heart.csv` dataset is in the working directory.

## Code Structure

**1. Data Preprocessing:**
   - Handle missing values and inspect data types.
   - Separate the independent features (`X`) from the target (`y`).

**2. Model Training and Evaluation:**
   - Use **Logistic Regression** for the initial model.
   - Implement multiple models as homework:
     - Decision Tree
     - Random Forest
     - Support Vector Classifier (SVC)
     - Ensemble models
   - Evaluate models using:
     - **Accuracy Score** for both training and testing datasets.

**3. Custom Predictions:**
   - Accept user input for health indicators and predict whether the heart is healthy or defective.
   - Example input:
     ```python
     input_data = (37,1,2,130,250,0,1,187,0,3.5,0,0,2)
     ```

## Example Output

If the heart is predicted as defective:
Affected by Defective Heart Disease


If the heart is predicted as healthy:
Don't Worry, Healthy Heart


## Model Evaluation

- **Training Accuracy**: The model's performance on the training dataset.
- **Test Accuracy**: The model's performance on the unseen testing dataset.