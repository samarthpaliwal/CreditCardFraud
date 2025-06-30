# train_and_save.py (Corrected for the final time)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

print("Starting training script...")

# --- Data Loading ---
df = pd.read_csv("creditcard.csv")

# --- Preprocessing ---
# Define the columns we are going to scale
cols_to_scale = ['Amount', 'Time']
scaler = StandardScaler()

# Fit the scaler on the two columns
scaler.fit(df[cols_to_scale])
joblib.dump(scaler, 'data_scaler.joblib')
print("Data scaler created and saved.")

# Transform the two columns and create the new scaled columns
df[['scaled_amount', 'scaled_time']] = scaler.transform(df[cols_to_scale])

# Drop the original, unscaled columns
df = df.drop(cols_to_scale, axis=1)


# --- Enforce Final Column Order ---
# This order is the single source of truth.
final_columns = ['scaled_amount', 'scaled_time'] + \
    [f'V{i}' for i in range(1, 29)]
df = df[final_columns]

print("\nColumns order for training:")
print(df.columns.tolist())

# --- Create Balanced Training Data ---
X = df
y = pd.read_csv("creditcard.csv")['Class']

train_data = pd.concat([X, y], axis=1)
fraud = train_data[train_data['Class'] == 1]
non_fraud = train_data[train_data['Class'] ==
                       0].sample(n=len(fraud), random_state=42)
balanced_df = pd.concat([non_fraud, fraud]).sample(frac=1, random_state=42)

X_train_balanced = balanced_df.drop('Class', axis=1)
y_train_balanced = balanced_df['Class']

# --- Train the Model ---
model = LogisticRegression(random_state=42)
model.fit(X_train_balanced, y_train_balanced)

joblib.dump(model, 'fraud_model.joblib')
print("\nModel training complete and saved.")
