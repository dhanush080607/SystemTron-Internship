# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
df = pd.read_csv(
    r"C:\Users\hdhan\OneDrive\Documents\Internships\SystemTron\week3\bank-full.csv",
    sep=';'
)

# Dataset Information
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Variable Distribution:")
print(df['y'].value_counts())

# Convert categorical columns into numerical columns
df = pd.get_dummies(df, drop_first=True)

print("\nDataset Shape After Encoding:", df.shape)

# Features and Target
X = df.drop('y_yes', axis=1)
y = df['y_yes']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Decision Tree Model
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score:")
print(f"{accuracy * 100:.2f}%")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance.head(10))

# -----------------------------
# Figure 1: Feature Importance
# -----------------------------
top_features = importance.head(10).sort_values(by='Importance')

plt.figure(figsize=(10, 6))
plt.barh(
    top_features['Feature'],
    top_features['Importance']
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.show()

# -----------------------------
# Figure 2: Decision Tree
# -----------------------------
plt.figure(figsize=(25, 12))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Classifier - Bank Marketing Dataset")
plt.tight_layout()
plt.show()