
# //!PART 1  Loading and Exploring the Data

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("creditcard.csv")

# Display the first 5 rows
print("First 5 rows of the data:")
print(df.head())

# Get a summary of the dataset
print("\nDataset Information:")
df.info()

# Check for any missing values
print("\nMissing values in each column:")
print(df.isnull().sum().max())

# this allows us ot see the Class imbalance
# Check the distribution of the 'Class' column
print("\nClass distribution:")
print(df['Class'].value_counts())

# Visualize the distribution
sns.countplot(x='Class', data=df)
plt.title('Class Distributions (0: No Fraud | 1: Fraud)')
plt.show()

# //! Part 2: Data Preprocessing

# //? Scale Amount and Time features to prevent them from disproportionately influencing the model due to their different scales. For line 39-51 :)


# Create a StandardScaler object
scaler = StandardScaler()

# Scale the 'Amount' and 'Time' columns
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop the original 'Time' and 'Amount' columns
df = df.drop(['Time', 'Amount'], axis=1)

# Display the new dataframe with scaled features
print("Data after scaling 'Amount' and 'Time':")
print(df.head())


# //! Part 3: Handling the Imbalanced Data

# //? Undersample the majority class in the training set to balance the data, ensuring the test set remains untouched to reflect real-world performance.


# Separate features (X) and target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data into training and testing sets (80% train, 20% test)
# We use 'stratify=y' to ensure the proportion of classes is the same in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ??Now, we perform undersampling on the X_train and y_train sets.

# Concatenate training data for easy undersampling
train_data = pd.concat([X_train, y_train], axis=1)

# Separate classes
fraud = train_data[train_data['Class'] == 1]
non_fraud = train_data[train_data['Class'] == 0]

# Perform random undersampling on the non-fraudulent class
non_fraud_undersampled = non_fraud.sample(n=len(fraud), random_state=42)

# Create the new balanced dataframe for training
balanced_train_df = pd.concat([non_fraud_undersampled, fraud])

# Shuffle the dataframe to mix the classes
balanced_train_df = balanced_train_df.sample(frac=1, random_state=42)

# Separate the balanced training features and target
X_train_balanced = balanced_train_df.drop('Class', axis=1)
y_train_balanced = balanced_train_df['Class']

# Check the new class distribution
print("\nBalanced training data class distribution:")
print(y_train_balanced.value_counts())

#!Part 4: Training the Machine Learning Model

# ? Time to start the Logic Regresstion


# Initialize the Logistic Regression model
model = LogisticRegression(random_state=42)

# Train the model on the balanced training data
model.fit(X_train_balanced, y_train_balanced)

#!!! Part 5: Evaluating the Model

# ?? Evaluate model performance by checking its Recall (% of actual frauds caught), Precision (% of alerts that were correct), and AUC score.

# Make predictions on the original test set
y_pred = model.predict(X_test)

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
            'No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred)
print(f"\nROC AUC Score: {roc_auc:.4f}")
