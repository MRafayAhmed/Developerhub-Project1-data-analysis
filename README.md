
## Project: Titanic Dataset - Data Cleaning & Visualization
## Project Steps (Explanation)

## 1. Load the Dataset
* We use `pandas` to load the Titanic dataset from a CSV file on your local machine.
* The dataset includes passenger details like age, sex, fare, survival status, etc.

## 2. Data Cleaning
a. Handle Missing Values
* Missing values in **numerical columns** are filled using the **median** (more robust to outliers).
* Missing values in **categorical columns** are filled using the **most frequent value (mode)**.
b. Remove Duplicates**
* Duplicate rows (if any) are removed using `drop_duplicates()` to avoid skewed analysis.
c. Manage Outliers**
* Outliers in numeric columns are removed using the **Interquartile Range (IQR)** method.
* This ensures extreme values don’t distort visualizations or statistical summaries.

## 3. Visualizations
a. Bar Charts (Categorical Variables)
* Visualize counts of categories (like `sex`, `embarked`, `pclass`, etc.) using bar charts.
b. Histograms (Numeric Variables)**
* Show the distribution and shape of numeric features (`age`, `fare`, etc.) using histograms with KDE curves.
c. Correlation Heatmap**
* Explore relationships between numeric features using a heatmap of their correlation matrix.

## 4. Summary of Insights**
* The final output includes written observations based on patterns discovered in the visualizations.


## How to Run the Script
1. Install Required Libraries (if not already installed):
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```
2. Place the CSV File:
* Make sure `tested.csv` is available at the path:
     ```
     C:/Users/Ansar-PC/Desktop/developerhub-data-analysis/tested.csv
     ```
3. Run the Script:
   * You can run the Python script using:
* **Command Line (CMD or PowerShell)**:
       ```bash
       python your_script_name.py
       ```
     * **Jupyter Notebook**: Paste the code into cells and run each step interactively.
     * **VS Code / PyCharm**: Open the script file and click Run.

## Observations (Insights from the Script)
1. Missing Data:
   * Several rows had missing values in `age`, `fare`, and possibly categorical fields like `embarked`.
   * These were successfully imputed to preserve dataset size and balance.
2. Duplicates:
   * Duplicate rows (if any) were removed to avoid bias in model training or analysis.

3. Outliers:
   * High fare or age outliers were removed to produce more reliable statistics and plots.
4. Bar Charts:
   * Most passengers were male.
   * `Pclass` and `embarked` had class imbalances worth noting.
5. Histograms:
   * `Age` was right-skewed, showing more younger passengers.
   * `Fare` had some extreme values, which justified outlier removal.
6. Correlation Heatmap:
   * Some positive correlations existed between features like `fare` and `pclass`.
   * Low correlations between features like `age` and `survived` suggest nonlinear or weak linear relationships.

# ---------- X -------------- X ---------------- X --------------- X 
# ---------- X -------------- X ---------------- X --------------- X 
# ---------- X -------------- X ---------------- X --------------- X 



# Project: Sentiment Analysis on IMDB Reviews
**Project Steps**
1.Text Preprocessing
   * Each review is:
     * Lowercased
     * Tokenized into words
     * Stripped of stopwords (common non-informative words like "the", "is")
     * Lemmatized (words are reduced to their root forms, e.g., “running” → “run”)
2. Feature Engineering
* Cleaned reviews are converted into numerical form using **TF-IDF vectorization**, which reflects the importance of each word across the dataset.
3. Model Training
   * The dataset is split into **training and testing sets** (80/20 split).
   * A **Logistic Regression** model is trained to classify sentiments.
4. Model Evaluation
   * Performance is measured using:

     * **Precision**
     * **Recall**
     * **F1-score**
   * Printed in a classification report.
5. Sentiment Prediction
* You can pass any new string to the `predict_sentiment()` function and get a sentiment classification ("Positive" or "Negative").

## How to Run the Script
1. Install Required Libraries
   Make sure you have the dependencies installed. Run this in terminal:

   ```bash
   pip install pandas nltk scikit-learn
   ```
2. Download NLTK Data
   The script automatically downloads:

   * Tokenizer data (`punkt`)
   * Stopword list
   * WordNet lemmatizer
3. Check Folder Path
   Ensure the following directory exists and has `pos/` and `neg/` subfolders:

   ```
   C:/Users/Ansar-PC/.cache/kagglehub/datasets/pawankumargunjan/imdb-review/versions/3/aclImdb/train
   ```
4. Run the Script
   Run from terminal or IDE:

   ```bash
   python Sentiment.py
   ```
# Observations
* Accuracy depends heavily on:
  * Quality of data
  * Balance between positive and negative samples
  * TF-IDF feature selection (e.g., number of `max_features`)
* **Lemmatization** improves performance over basic stemming by preserving word context.
* **Improvements you could try:**

  * Use **word embeddings** (e.g., Word2Vec or GloVe)
  * Use deep learning models (e.g., LSTM, BERT)
  * Use the **test folder** for final evaluation

# ---------- X -------------- X ---------------- X --------------- X 
# ---------- X -------------- X ---------------- X --------------- X 
# ---------- X -------------- X ---------------- X --------------- X


# Project: Credit Card Fraud Detection Using Random Forest

# **Overview**

This project detects fraudulent credit card transactions using a supervised machine learning model (Random Forest Classifier). The dataset used contains anonymized transaction details, including a binary class label (0 = Legit, 1 = Fraud).

# Step-by-Step Explanation
1. Import Libraries
You load essential libraries for:

* Data manipulation: `pandas`, `numpy`
* Visualization: `matplotlib`, `seaborn`
* Modeling: `sklearn`, `imblearn`
2. Load and Inspect the Dataset
```python
data = pd.read_csv('creditcard.csv')
```
You examine the first few rows and class distribution using:
```python
pd.value_counts(data['Class']).plot.bar()
```
Observation: The dataset is highly imbalanced (very few fraud cases).
3. Data Preprocessing
* Normalize the `Amount` column → `normAmount`
* Drop the `Time` and original `Amount` columns
* Split into features `X` and labels `y`
4. Train-Test Split and Oversampling
You split the dataset (70% training, 30% testing), then use **SMOTE** to balance the minority class in the training set:
```python
sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
```
Observation: Class distribution becomes 50-50 after SMOTE.
5. Train Random Forest Classifier
```python
classifier = RandomForestClassifier(...)
classifier.fit(Xtrain, Ytrain)
```
You fit the model using the training data and tune parameters like `n_estimators`, `max_depth`, etc.
6. Feature Importance Visualization
You use:
```python
sns.barplot(x='Feature importance', y='Feature', data=tmp)
```
To see which features influence predictions most.
7. Model Evaluation
You evaluate predictions using:
```python
confusion_matrix(Ytest, predictions)
classification_report(Ytest, predictions)
```
Key metrics shown: **Precision**, **Recall**, **F1-score**
8. Testing Interface (CLI)
You implement a simple command-line interface:
```python
predict_transaction()
```
It prompts the user for 29 input features and prints whether the transaction is likely **fraudulent** or **legit**.


# How to Run the Script
1. Place the CSV file at the specified path (or update the path).
2. Run the entire script in a Jupyter Notebook or Python environment.
3. After training is complete, the script calls:
   ```python
   predict_transaction()
   ```
   You’ll be asked to input 29 values (from a row in the dataset or manually).
4. The model predicts whether it’s **Fraud** or **Legit**.

# Observations

* **SMOTE** significantly improves the model's ability to detect fraud by balancing the classes.
* **Random Forest** performs well for imbalanced classification due to ensemble learning.
* **Feature importance** helps understand which anonymized features are critical for fraud detection.
* The **CLI test** is a simple but effective way to try out predictions manually.

# Optional Tip: Test with Real Data Row
Instead of entering 29 values manually, use:
```python
sample = Xtest.iloc[0].values.reshape(1, -1)
classifier.predict(sample)
```
# ---------- X -------------- X ---------------- X --------------- X 
# ---------- X -------------- X ---------------- X --------------- X 
# ---------- X -------------- X ---------------- X --------------- X

**Boston Housing Price Prediction project**

# Project Overview
This project implements a **machine learning regression task** to predict housing prices in Boston. Three models are developed **from scratch** without using libraries like `sklearn` for modeling:
1. **Linear Regression**
2. **Random Forest**
3. **XGBoost**
The models are evaluated using **RMSE** (Root Mean Squared Error) and **R² Score**, and feature importance is visualized for the Random Forest model.
# Project Steps
# 1. Load Dataset
* Loads the `BostonHousing.csv` file using `pandas`.
* Displays column names.
# 2. Preprocess Data
* Splits the dataset into:
  * Features (`X`)
  * Target variable (`y = medv`)
* Standardizes features (zero mean, unit variance).
* Splits the dataset into training and test sets using an 80/20 split.
# 3. Linear Regression (From Scratch)
* Implements the normal equation:
  $$
  \theta = (X^TX)^{-1}X^Ty
  $$
* Adds a bias term manually.
# 4. Random Forest Regressor (From Scratch)
* Implements:
  * A simple **decision tree regressor**
  * A **random forest** by bootstrapping training samples and averaging predictions from multiple trees
* Uses recursive splitting based on variance minimization.
# 5. XGBoost Regressor (From Scratch)
* Builds an ensemble of trees trained on residuals of previous predictions.
* Uses **gradient boosting** principles with a learning rate to improve predictions.
# 6. Evaluation Metrics
* **RMSE**: Measures average prediction error.
* **R² Score**: Measures how well the predictions approximate the actual values.
# 7. Train and Evaluate
* Trains each model and evaluates on the test set.
* Displays RMSE and R² for each model.
# 8. Feature Importance (Random Forest)
* Traverses each decision tree and counts how often each feature is used in splits.
* Visualizes the most influential features using a horizontal bar chart.

# How to Run the Script
1. Dependencies: Install the required Python libraries:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
2. CSV File:
   * Place `BostonHousing.csv` in the correct path as used in the script:

     ```
     C:/Users/Ansar-PC/Desktop/developerhub-data-analysis/BostonHousing.csv
     ```
   * Ensure the file contains the `medv` column.
3. Run the script:
   * You can run the script in:

     * Jupyter Notebook
     * Google Colab (update the path)
     * Any Python IDE (e.g., VS Code, PyCharm)

# Observations

# Model Performance (example output):
Model Performance Comparison:
Linear Regression: RMSE = 4.72, R² = 0.73
Random Forest:     RMSE = 3.45, R² = 0.85
XGBoost:           RMSE = 3.25, R² = 0.87
#  Key Insights:
* **XGBoost** performs best due to its ability to model complex patterns and correct errors iteratively.
* **Random Forest** also performs well due to its ensemble nature and variance reduction.
* **Linear Regression** is fast but limited due to its assumption of linear relationships.

# Feature Importance:
* The bar chart shows which features are most frequently used in Random Forest splits.
* This helps interpret the model and identify critical factors affecting housing prices, such as:

  * LSTAT (% lower status population)
  * RM (number of rooms)
  * PTRATIO (pupil-teacher ratio)

# Optional Enhancements
You can consider:

* Adding cross-validation
* Hyperparameter tuning (depth, estimators)
* Implementing regularized regression (Lasso/Ridge)
* Comparing with `scikit-learn`'s built-in models
# ---------- X -------------- X ---------------- X --------------- X 
# ---------- X -------------- X ---------------- X --------------- X 
# ---------- X -------------- X ---------------- X --------------- X
