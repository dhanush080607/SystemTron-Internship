# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
train = pd.read_csv(r"C:\Users\hdhan\OneDrive\Documents\Internships\SystemTron\week2\train.csv")

# Basic Information
print("First 5 Rows:")
print(train.head())

print("\nDataset Shape:")
print(train.shape)

print("\nDataset Info:")
print(train.info())

print("\nMissing Values:")
print(train.isnull().sum())

print("\nStatistical Summary:")
print(train.describe())

# Data Cleaning

# Fill missing Age values with median
train['Age'].fillna(train['Age'].median(), inplace=True)

# Fill missing Embarked values with mode
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

# Drop Cabin column because of too many missing values
train.drop('Cabin', axis=1, inplace=True)

print("\nMissing Values After Cleaning:")
print(train.isnull().sum())

# Exploratory Data Analysis (EDA)

sns.set_style("whitegrid")

# Survival Count

plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=train)
plt.title("Survival Count")
plt.show()

# Gender Distribution

plt.figure(figsize=(6,4))
sns.countplot(x='Sex', data=train)
plt.title("Gender Distribution")
plt.show()

# Gender vs Survival

plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=train)
plt.title("Gender vs Survival")
plt.show()

# Passenger Class Distribution

plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', data=train)
plt.title("Passenger Class Distribution")
plt.show()

# Passenger Class vs Survival

plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=train)
plt.title("Passenger Class vs Survival")
plt.show()

# Age Distribution

plt.figure(figsize=(8,5))
sns.histplot(train['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()


# Age vs Survival

plt.figure(figsize=(8,5))
sns.histplot(data=train,
             x='Age',
             hue='Survived',
             bins=30,
             kde=True)
plt.title("Age vs Survival")
plt.show()

# Fare Distribution

plt.figure(figsize=(8,5))
sns.histplot(train['Fare'], bins=30, kde=True)
plt.title("Fare Distribution")
plt.show()

# Embarked Distribution

plt.figure(figsize=(6,4))
sns.countplot(x='Embarked', data=train)
plt.title("Embarked Port Distribution")
plt.show()

# Embarked vs Survival

plt.figure(figsize=(6,4))
sns.countplot(x='Embarked', hue='Survived', data=train)
plt.title("Embarked Port vs Survival")
plt.show()

# Correlation Analysis

# Convert categorical columns to numeric

train_corr = train.copy()

train_corr['Sex'] = train_corr['Sex'].map({'male':0,'female':1})
train_corr['Embarked'] = train_corr['Embarked'].map({'S':0,'C':1,'Q':2})

corr_matrix = train_corr.corr(numeric_only=True)

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt='.2f')

plt.title("Correlation Matrix")
plt.show()

# Key Statistics

print("\nOverall Survival Rate:")
print(train['Survived'].mean()*100)

print("\nSurvival Rate by Gender:")
print(train.groupby('Sex')['Survived'].mean()*100)

print("\nSurvival Rate by Passenger Class:")
print(train.groupby('Pclass')['Survived'].mean()*100)

print("\nAverage Fare by Survival:")
print(train.groupby('Survived')['Fare'].mean())

print("\nAverage Age by Survival:")
print(train.groupby('Survived')['Age'].mean())

print("\nEDA Completed Successfully!")