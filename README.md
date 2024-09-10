# Credit Card Fraud Detection

This project is aimed at building a machine learning model to detect fraudulent credit card transactions using various classification algorithms like Logistic Regression, Decision Trees, and Random Forests. The dataset contains simulated credit card transactions from 1000 customers, covering legitimate and fraud transactions from January 1, 2019, to December 31, 2020.

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Description

Credit card fraud detection is crucial for financial institutions, and this project aims to build predictive models to classify transactions as legitimate or fraudulent. By applying machine learning techniques, we explore methods to detect anomalies and fraudulent patterns in credit card transaction data.

We experiment with three machine learning algorithms:
- Logistic Regression
- Decision Tree
- Random Forest

The primary goal is to compare the performance of these models and choose the best one for fraud detection based on evaluation metrics like accuracy, precision, recall, and F1-score.

## Features

- Fraud detection using machine learning models.
- Data preprocessing with feature engineering and encoding of categorical variables.
- Evaluation metrics include confusion matrix, accuracy, precision, recall, and F1-score.
- Scalable for integration into larger financial platforms.

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/fraud-detection.git
   cd fraud-detection
   
2. Install the required dependencies listed in the requirements.txt file:

```bash
pip install -r requirements.txt
```

3. Ensure that the dataset files fraudTrain.csv and fraudTest.csv are located in the root directory of the project. Update file paths in the code if necessary.

## Usage
Once you have the necessary files, run the following command to train and evaluate the models:

```bash
python main.py
```

## Steps:
1. The code will load both training (fraudTrain.csv) and testing (fraudTest.csv) datasets.
2. Perform data preprocessing:
   - Remove unnecessary columns.
   - Handle categorical features (e.g., gender, category) using one-hot encoding (pd.get_dummies).
   - Standardize the data using StandardScaler.
3. Train three models: Logistic Regression, Decision Tree, and Random Forest.
4. Evaluate the models with a confusion matrix, accuracy score, precision, recall, and F1-score.
   
## Dataset
The dataset includes:
- fraudTrain.csv: The training dataset containing labeled transaction data.
- fraudTest.csv: The test dataset to evaluate the models.
These datasets contain transactions from credit card customers across various merchants. The data includes categorical features like transaction category, gender, transaction time, amount, and a label indicating whether a transaction is fraudulent.

## Results
After running the models, the results include:
- Confusion Matrix: Shows the true positives, true negatives, false positives, and false negatives.
- Accuracy Score: Measures the overall performance of the model.
- Precision, Recall, F1-Score: Key metrics for evaluating classification performance, especially important for handling imbalanced datasets.

## Sample output:

- Confusion Matrix:[[235   4][ 13  58]]
- Accuracy: 0.956
- Precision: 0.935
- Recall: 0.871
- F1-score: 0.902
You can modify the evaluation process or save the model for future use.

## Contributing
We welcome contributions to improve this project! Follow the steps below to contribute:
Fork the repository:

```bash
git fork https://github.com/your-username/fraud-detection.git
```

Create a new branch for your feature or bugfix:
```bash
Copy code
git checkout -b feature-name
```
Make changes and commit your work:
```bash
git commit -am 'Add feature or fix
```
Push your branch to GitHub:

```bash
git push origin feature-name
```
Open a Pull Request, and describe the changes you’ve made. We'll review it as soon as possible.

## License
This project is licensed under the MIT License - see the LICENSE file for details.


