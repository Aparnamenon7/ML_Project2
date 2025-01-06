# ML_Project2
Orinson Machine Learning Internship Task 2

**Objective**
This script performs a simple Linear Regression analysis using Python to predict Airbnb prices in Tokyo based on their distance from Tokyo Tower.The model that predicts Airbnb prices based on distance with reasonable accuracy. Visualization aids in assessing whether a linear relationship exists.

**Step By Step Loading**
1. Dataset Loading
The dataset is stored in an Excel file located at D:\Python projects\python\tokyo_airbnb_dataset.xlsx.
It is loaded into a Pandas DataFrame using pd.read_excel.
The dataset is expected to have at least two columns:
-Distance to Tokyo Tower (km): A numerical feature indicating the distance of an Airbnb listing from Tokyo Tower.
-AirBnB Price (USD): The target variable, representing the price of an Airbnb listing.

2. Feature and Target Extraction
The script extracts the independent variable (X) and dependent variable (y):
X: Distance to Tokyo Tower (reshaped as a 2D array for compatibility with scikit-learn).
y: Airbnb price in USD.

3. Data Splitting
The dataset is split into:
Training Set (80%): Used to train the model.
Testing Set (20%): Used to evaluate the model's performance.
This is done using train_test_split with random_state=42 for reproducibility.

4. Model Training
A Linear Regression model is created using LinearRegression from scikit-learn.
The model is trained on the training data (X_train, y_train) using .fit().


This script performs a simple Linear Regression analysis using Python to predict Airbnb prices in Tokyo based on their distance from Tokyo Tower. Here's a step-by-step explanation of the task:

1. Dataset Loading
The dataset is stored in an Excel file located at D:\Python projects\python\tokyo_airbnb_dataset.xlsx.
It is loaded into a Pandas DataFrame using pd.read_excel.
The dataset is expected to have at least two columns:
Distance to Tokyo Tower (km): A numerical feature indicating the distance of an Airbnb listing from Tokyo Tower.
AirBnB Price (USD): The target variable, representing the price of an Airbnb listing.

3. Feature and Target Extraction
The script extracts the independent variable (X) and dependent variable (y):
X: Distance to Tokyo Tower (reshaped as a 2D array for compatibility with scikit-learn).
y: Airbnb price in USD.

5. Data Splitting
The dataset is split into:
Training Set (80%): Used to train the model.
Testing Set (20%): Used to evaluate the model's performance.
This is done using train_test_split with random_state=42 for reproducibility.

6. Performance Metrics
Mean Squared Error (MSE): Measures the average squared difference between actual and predicted values. A lower MSE indicates better model performance.
R-squared (R2) Score: Represents the proportion of variance in the dependent variable explained by the independent variable. Ranges from 0 (no explanation) to 1 (perfect explanation).

7. Visualization
The script creates three plots to visualize the relationship between distance and price:-
-Scatter Plot:
Shows the data points (distance vs. Airbnb price) to understand the data distribution.
-Regression Line:
Plots the predicted regression line over the data points. This line represents the model's predictions based on the input distances.
-Combined Plot:
Combines the scatter plot of the data points with the regression line, providing a comprehensive view of the model's performance.

8. Model Training
A Linear Regression model is created using LinearRegression from scikit-learn.
The model is trained on the training data (X_train, y_train) using .fit().

9. Model Prediction
Predictions are made on the test data (X_test) using .predict().


