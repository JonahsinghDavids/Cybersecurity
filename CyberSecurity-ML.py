import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
train_df = pd.read_csv(r"C:\Users\Lenovo\first_10000_rows.csv")
test_df = pd.read_csv(r"C:\Users\Lenovo\first_10000_rowsTest.csv")

# Initial Inspection
print(f"Train Dataset Shape: {train_df.shape}")
print(f"Test Dataset Shape: {test_df.shape}")
#print(f"Columns in Train Dataset: {train_df.columns}")
#print(f"Columns in Test Dataset: {test_df.columns}")

# EDA: Distribution of the target variable (IncidentGrade)
sns.countplot(x='IncidentGrade', data=train_df)
plt.title('Distribution of IncidentGrade')
plt.show()

# EDA: Correlation heatmap
plt.figure(figsize=(14, 12))
correlation_matrix = train_df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Handling Missing Data
columns_to_drop = ['Sha256', 'IpAddress', 'Url', 'AccountSid', 'AccountUpn', 'AccountObjectId', 
                    'NetworkMessageId', 'EmailClusterId', 'RegistryKey', 'RegistryValueName', 
                    'RegistryValueData', 'OAuthApplicationId']
train_df = train_df.drop(columns=columns_to_drop)
test_df = test_df.drop(columns=columns_to_drop)

# Imputation of missing data
imputer = SimpleImputer(strategy='most_frequent')
train_df = pd.DataFrame(imputer.fit_transform(train_df), columns=train_df.columns)
test_df = pd.DataFrame(imputer.transform(test_df), columns=test_df.columns)

# Feature Engineering: Creating new features from Timestamp
train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'], errors='coerce')
test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'], errors='coerce')

train_df['Hour'] = train_df['Timestamp'].dt.hour
train_df['DayOfWeek'] = train_df['Timestamp'].dt.dayofweek
train_df['Month'] = train_df['Timestamp'].dt.month
train_df['Year'] = train_df['Timestamp'].dt.year

test_df['Hour'] = test_df['Timestamp'].dt.hour
test_df['DayOfWeek'] = test_df['Timestamp'].dt.dayofweek
test_df['Month'] = test_df['Timestamp'].dt.month
test_df['Year'] = test_df['Timestamp'].dt.year

# Drop the original Timestamp column after feature extraction
train_df = train_df.drop(columns=['Timestamp'])
test_df = test_df.drop(columns=['Timestamp'])

# Ensure that the target variable and non-numeric columns are excluded from numeric transformations
non_numeric_columns = ['IncidentGrade', 'Id']
X = train_df.drop(columns=non_numeric_columns)
y = train_df['IncidentGrade']

# Encoding Categorical Variables
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    test_df[column] = le.transform(test_df[column])
    label_encoders[column] = le

# Save the label encoders for future use
joblib.dump(label_encoders, 'label_encoders.pkl')

# Data Splitting: Train-validation split with stratification
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Baseline Model: Logistic Regression
baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_val)

print("Baseline Model (Logistic Regression) Classification Report:")
print(classification_report(y_val, y_pred_baseline))

# Advanced Model: Random Forest with Cross-Validation
rf_model = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation score
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='f1_macro')
print(f"Cross-Validation F1 Macro Score: {cv_scores.mean():.4f}")

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(rf_model, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

# Retraining Random Forest with best hyperparameters
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train, y_train)

# Evaluate on Validation Set
y_pred_val = best_rf_model.predict(X_val)
print("Random Forest Model (Validation Set) Classification Report:")
print(classification_report(y_val, y_pred_val))

# Handling Class Imbalance: Using SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Retrain Random Forest on balanced data
best_rf_model.fit(X_train_balanced, y_train_balanced)

# Evaluate again on Validation Set after balancing
y_pred_val_balanced = best_rf_model.predict(X_val)
print("Random Forest Model with SMOTE (Validation Set) Classification Report:")
print(classification_report(y_val, y_pred_val_balanced))

# Feature Importance Analysis
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(12, 8))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()

# Error Analysis: Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_pred_val_balanced)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['TP', 'BP', 'FP'], yticklabels=['TP', 'BP', 'FP'])
plt.title('Confusion Matrix (Validation Set)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Final Evaluation on Test Set
X_test = test_df.drop(columns=['IncidentGrade', 'Id'])
y_test = test_df['IncidentGrade']

y_pred_test = best_rf_model.predict(X_test)
print("Random Forest Model (Test Set) Classification Report:")
print(classification_report(y_test, y_pred_test))

# Save the final model for future use
joblib.dump(best_rf_model, 'final_model.pkl')
