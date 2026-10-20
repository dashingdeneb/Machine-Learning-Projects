# Student Performance Analysis and Prediction

## 📌 Project Overview

This project explores student academic performance using **data analysis, probability, machine learning, and dimensionality reduction**. The objective is to understand factors associated with students' final scores and identify students who may require additional academic support based on their academic risk levels.

The project covers the complete data analysis workflow, from exploratory data analysis (EDA) and preprocessing to model training, evaluation, and visualization.

## 🎯 Objectives

- Analyze student performance patterns using exploratory data analysis and visualization.
- Calculate probabilities associated with passing outcomes and participation in extra classes.
- Predict students' final scores using regression models.
- Classify students according to their academic risk levels.
- Compare traditional machine learning models with a neural network.
- Analyze regression coefficients to understand associations between predictors and final scores.
- Apply dimensionality reduction techniques to visualize student groups.

## 🛠️ Technologies Used

- **Python** — programming and analysis
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical computations
- **Matplotlib and Seaborn** — data visualization
- **Scikit-learn** — preprocessing, machine learning, model evaluation, and dimensionality reduction
- **Jupyter Notebook** — interactive analysis and experimentation

## 📊 Project Workflow

### 1. Data Cleaning and Preprocessing

Prepared the dataset for analysis and model training by handling missing values, encoding categorical variables, and scaling numerical features.

Preprocessing was implemented using Scikit-learn pipelines and `ColumnTransformer` to maintain a consistent workflow and reduce data leakage during model training.

Categorical encoding strategies included:

- One-hot encoding for binary categorical variables and nominal categories.
- Ordinal encoding for `ParentEducation`, reflecting the order of educational levels.
- Standard scaling for numerical features.

### 2. Exploratory Data Analysis (EDA)

Explored the dataset to understand the distribution of student characteristics and academic outcomes.

Key activities included:

- Examining descriptive statistics and data distributions.
- Visualizing numerical and categorical variables.
- Investigating relationships between academic factors and `FinalScore`.
- Comparing student performance across relevant categories.

### 3. Probability Analysis

Calculated probabilities related to student academic outcomes, including:

- The probability of passing among students who took extra classes.
- The probability of passing among students who did not take extra classes.

This analysis helped compare passing outcomes between the two groups.

### 4. Regression — Final Score Prediction

Developed regression models to predict `FinalScore`.

**Models implemented:**

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- MLPRegressor (neural network)

**Evaluation metrics:**

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² score

The regression analysis also examined model coefficients to understand the direction and magnitude of associations between predictors and predicted final scores.

### 5. Classification — Academic Risk Prediction

Built classification models to predict `RiskLevel`, with the objective of identifying students who may need academic support.

**Models implemented:**

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

**Evaluation metrics:**

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Recall was given particular attention because failing to identify a student who may need support can be more concerning than incorrectly flagging a student.

### 6. Neural Network — MLPRegressor

Implemented an `MLPRegressor` using Scikit-learn and compared its performance with Linear Regression.

The neural network's architecture and training settings were adjusted experimentally, and its training loss curve was examined to understand model learning and convergence.

### 7. Dimensionality Reduction

Applied three dimensionality reduction techniques to explore patterns in the student data.

- **Principal Component Analysis (PCA):** Reduced the processed feature space to two components and visualized observations by `RiskLevel`.
- **Linear Discriminant Analysis (LDA):** Used `RiskLevel` as the supervised target to find a projection that separates the risk groups.
- **t-distributed Stochastic Neighbor Embedding (t-SNE):** Produced a two-dimensional representation to explore nonlinear patterns and local group structure.

The resulting visualizations were compared to assess how clearly the risk groups appeared separated.

## 📈 Key Findings

### Regression

Linear Regression achieved the following results:

| Metric | Result |
|---|---:|
| MAE | 3.2994 |
| MSE | 16.3839 |
| RMSE | 4.0477 |
| R² | 0.6908 |

Linear Regression outperformed the tested MLPRegressor configuration, which achieved an R² score of approximately 0.6223. This suggests that the neural network did not provide an improvement over the simpler linear model for this dataset.

The coefficient analysis showed positive associations between predicted final scores and variables such as study hours, previous GPA, extra classes, quiz average, and assignment score. `PassStatus` and `RiskLevel` were also included as predictors in the regression analysis, following the project instructions.

### Classification

The classification results were:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 87.14% | 65.96% | 93.94% | 77.50% |
| Decision Tree | 81.43% | 57.78% | 78.79% | 66.67% |
| Random Forest | 89.29% | 84.62% | 66.67% | 74.58% |

Logistic Regression achieved the highest recall for the Medium Risk class, identifying 31 of the 33 actual Medium Risk students in the test set. It missed only two students in this group.

Although Random Forest achieved the highest overall accuracy, Logistic Regression was the preferred model for the specific objective of identifying students who may need support, because it produced fewer false negatives.

### Dimensionality Reduction

The first two PCA components explained approximately **32.67% of the total variance**:

- PC1: 20.08%
- PC2: 12.59%

The PCA visualization showed some separation between Low Risk and Medium Risk students, but substantial overlap remained. LDA showed clearer separation in the resulting visualization, while t-SNE revealed additional nonlinear structure.

These are visual findings rather than evidence of perfect class separation or predictive performance.

## 💡 Key Learnings

Through this project, I gained practical experience in:

- Performing EDA and visualizing relationships in a real-world-style dataset.
- Applying probability concepts to compare academic outcomes.
- Building regression and classification pipelines using Scikit-learn.
- Selecting appropriate encoding strategies for categorical variables.
- Understanding data leakage and the importance of separating predictors from the target.
- Evaluating models using multiple metrics rather than accuracy alone.
- Comparing a neural network with traditional machine learning models.
- Applying and interpreting PCA, LDA, and t-SNE.
- Translating model results into meaningful findings and limitations.

