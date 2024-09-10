# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the datasets
train_data = pd.read_csv('c:/Users/HP/OneDrive/Documents/VS Code/Afame/fraudTrain.csv')
test_data = pd.read_csv('c:/Users/HP/OneDrive/Documents/VS Code/Afame/fraudTest.csv')

# Data preprocessing
def preprocess_data(data):
    # Drop unnecessary columns (example: transaction ID, customer ID, merchant name, etc.)
    columns_to_drop = ['trans_num', 'cc_num', 'first', 'last', 'street', 'city', 
                       'state', 'zip', 'lat', 'long', 'city_pop', 'job', 
                       'dob', 'trans_date_trans_time', 'unix_time', 'merchant']
    data = data.drop(columns=columns_to_drop, axis=1)
    
    # Convert categorical columns to dummy variables (e.g., 'category', 'gender')
    data = pd.get_dummies(data, drop_first=True)

    # Separate features and labels
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']
    
    return X, y

# Preprocess the training and testing data
X_train, y_train = preprocess_data(train_data)
X_test, y_test = preprocess_data(test_data)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Logistic Regression model
print("Logistic Regression:")
log_reg = LogisticRegression()
evaluate_model(log_reg, X_train_scaled, y_train, X_test_scaled, y_test)

# Decision Tree model
print("\nDecision Tree Classifier:")
tree = DecisionTreeClassifier()
evaluate_model(tree, X_train, y_train, X_test, y_test)

# Random Forest model
print("\nRandom Forest Classifier:")
forest = RandomForestClassifier(n_estimators=100)
evaluate_model(forest, X_train, y_train, X_test, y_test)