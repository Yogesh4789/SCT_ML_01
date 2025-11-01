# -------------------------------------------
# LINEAR REGRESSION MODEL FOR HOUSE PRICE PREDICTION
# -------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\part 01\\SCT\\SCT_ML_01\\train.csv")

# Select relevant features
features = [
    'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr',
    'OverallQual', 'GarageArea', 'YearBuilt', 'TotalBsmtSF',
    'YrSold', 'SalePrice'
]
df = data[features].dropna()


# FEATURE ENGINEERING
# -----------------------------

# Combine full and half bathrooms into a single feature
df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath']

# Create an "Age" feature (how old the house is at sale)
df['Age'] = df['YrSold'] - df['YearBuilt']

# Apply log transformation to handle skewness
df['SalePrice'] = np.log(df['SalePrice'])
df['GrLivArea'] = np.log(df['GrLivArea'] + 1)
df['TotalBsmtSF'] = np.log(df['TotalBsmtSF'] + 1)
df['GarageArea'] = np.log(df['GarageArea'] + 1)


# FEATURE SELECTION
# -----------------------------
X = df[['GrLivArea', 'TotalBath', 'BedroomAbvGr', 'OverallQual',
        'GarageArea', 'TotalBsmtSF', 'Age']]
y = df['SalePrice']


# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# FEATURE SCALING
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# MODEL TRAINING
# -----------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)


# PREDICTION & EVALUATION
# -----------------------------
y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Linear Regression Model Performance:")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


# VIEW COEFFICIENTS
# -----------------------------
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\nFeature Importance (Linear Coefficients):")
print(coefficients)
