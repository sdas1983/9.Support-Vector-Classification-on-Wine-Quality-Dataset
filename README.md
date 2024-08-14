# Support Vector Classification on Wine Quality Dataset

This project applies Support Vector Classification (SVC) to the Wine Quality Dataset. The goal is to classify the wine quality based on various chemical properties.

## Dataset

The dataset used in this project is the Wine Quality Dataset, which consists of red wine samples with various features such as acidity, sugar, pH levels, and alcohol content. The target variable is the wine quality, which is an ordinal integer value.

## Project Steps

### 1. Data Loading and Exploration

- Load the dataset using `pandas`.
- Explore the dataset with `info()`, `describe()`, and correlation analysis.
- Visualize the dataset using `Seaborn` and `Matplotlib` to understand the relationships between features.

### 2. Data Preprocessing

- Handle missing values and duplicates.
- Normalize the data using `StandardScaler` to improve model performance.
- Apply the Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance.

### 3. Model Training

- Split the dataset into training and testing sets.
- Train a Support Vector Classification model on the training data.
- Evaluate the model's performance on the test data.

### 4. Model Evaluation

- Calculate performance metrics such as accuracy, precision, F1-score, and confusion matrix.
- Visualize the performance using confusion matrix and ROC curves.

### 5. Data Visualization

- Create various plots to visualize the data and the results, including:
  - Pair plots
  - Heatmaps
  - Bar plots
  - Count plots
  - Scatter plots

## Dependencies

To run the code, you'll need the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imblearn`

You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imblearn
```

## Conclusion
This project demonstrates the application of Support Vector Classification on a real-world dataset. The code includes data preprocessing, model training, evaluation, and visualization, providing a comprehensive overview of the process.
