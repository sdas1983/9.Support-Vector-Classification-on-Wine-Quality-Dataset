# Support Vector Classification on Wine Quality Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv(r"C:\Users\das.su\OneDrive - GEA\Documents\PDF\Machine Learning\BIT ML, AI and GenAI Course\winequality-red.csv")
df_x = pd.read_csv(r"C:\Users\das.su\OneDrive - GEA\Documents\PDF\Machine Learning\BIT ML, AI and GenAI Course\WineQT.csv")

# Display dataset information
df.info()

# Feature and target separation
X = df.drop(['quality'], axis=1)
y = df['quality']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Transform test data

# Logistic Regression model
regression = LogisticRegression()
regression.fit(X_train, y_train)
print('Test Score -', regression.score(X_test, y_test))
print('Train Score -', regression.score(X_train, y_train))

# Support Vector Classifier model
classifier = SVC()
classifier.fit(X_train, y_train)

# Predictions and evaluation
y_pred = classifier.predict(X_test)
print('Accuracy Score:', accuracy_score(y_test, y_pred))

# Data exploration and visualization
df['quality'].unique()
df['quality'].value_counts()
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.shape

# Correlation heatmap
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True)

# Count plots
ax = df.quality.value_counts().plot(kind='bar')
ax = sns.countplot(data=df, x='quality', order=df.quality.value_counts().index)
ax = sns.countplot(data=df, x='quality')
for bars in ax.containers:
    ax.bar_label(bars)

# Histograms
sns.histplot(df['fixed acidity'], kde=True)

# Categorical plots
sns.catplot(x='quality', y='alcohol', data=df, kind='box')

# Scatterplot
sns.scatterplot(x='alcohol', y='pH', hue='quality', data=df)

# Boxplot
plt.figure(figsize=(20,10))
sns.boxplot(data=df)

# Convert quality to binary classification
df['quality'] = df['quality'].apply(lambda x: 1 if x > 6.5 else 0)
X = df.drop(['quality'], axis=1)
y = df['quality']

# Handle imbalanced dataset using SMOTE
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)

# Visualization of sampling effect
sns.set_style('white')
plt.figure(figsize=(15,4))

plt.subplot(1,2,1)
sns.countplot(data=df, x='quality')
plt.xticks([0,1], ['Bad Wine', 'Good Wine'])
plt.xlabel('\nBefore Sampling')

plt.subplot(1,2,2)
sns.countplot(x=y_smote)
plt.xticks([0,1], ['Bad Wine', 'Good Wine'])
plt.xlabel('\nAfter Sampling')

plt.suptitle('Over Sampling \n\n\n')
plt.show()

# Pairplot
sns.pairplot(df, hue='quality', size=1.2, diag_kind='kde')
plt.show()

sns.pairplot(df, size=1.2, diag_kind='kde', kind='hist')
plt.show()

# Bar plots for quality vs sulphates and alcohol
sns.barplot(x='quality', y='sulphates', data=df)
sns.barplot(x='quality', y='alcohol', data=df)

# Re-split the dataset for SVM after balancing
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.25, random_state=42)

# Re-scale features for SVM
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM model
classifier = SVC()
classifier.fit(X_train, y_train)

# Predictions and evaluation
y_pred = classifier.predict(X_test)

# Confusion matrix and ROC curve
ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test)
RocCurveDisplay.from_estimator(classifier, X_test, y_test)

# Print classification report
print(classification_report(y_test, y_pred))
