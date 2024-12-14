import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import glob
import os
from sklearn.model_selection import KFold

# Load data
data_files = []
for i in range(10):
    data_files.extend(glob.glob(f'data_processed/sym{i}_data*_am.csv'))
    data_files.extend(glob.glob(f'data_processed/sym{i}_data*_pm.csv'))

file_dir = './data_processed'
i = 0
dates = list(range(12))
df = []
for date in dates:
    if (date & 1):
        file_name = f"snapshot_sym{i}_date{date//2}_am.csv"
    else:
        file_name = f"snapshot_sym{i}_date{date//2}_pm.csv"
    df.append(pd.read_csv(os.path.join(file_dir, file_name)))
    
# Concatenate all sampled data
data = pd.concat(df, ignore_index=True)
# print(data.T)

# Prepare features and labels
X = data.drop(columns=['date','time', 'sym', 'label_5', 'label_10', 'label_20', 'label_40', 'label_60'])
y = data['label_5']

# Split data into training and testing sets
# Ensure data is sorted by date and time to maintain the time series order
data = data.sort_values(by=['date', 'time'])

# Split data into training and testing sets without shuffling to prevent data leakage
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

X_train = train_data.drop(columns=['date', 'time', 'sym', 'label_5', 'label_10', 'label_20', 'label_40', 'label_60'])
y_train = train_data['label_5']
X_test = test_data.drop(columns=['date', 'time', 'sym', 'label_5', 'label_10', 'label_20', 'label_40', 'label_60'])
y_test = test_data['label_5']

# Train XGBoost model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Feature importance
importance = model.feature_importances_
importance_normalized = importance / importance.max()
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importance_normalized
}).sort_values(by='importance', ascending=False)

print(feature_importance)

# Prepare features and labels
X = data.drop(columns=['date', 'time', 'sym', 'label_5', 'label_10', 'label_20', 'label_40', 'label_60'])
y = data['label_5']

# Initialize parameters for rolling window cross-validation
window_size = int(len(data) * 0.2)
step_size = int(len(data) * 0.1)

mse_list = []
start = 0

while start + window_size < len(data):
    end = start + window_size
    X_train, X_test = X.iloc[start:end], X.iloc[end:end + step_size]
    y_train, y_test = y.iloc[start:end], y.iloc[end:end + step_size]

    # Train XGBoost model
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)

    # Feature importance
    importance = model.feature_importances_
    importance_normalized = importance / importance.max()
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importance_normalized
    }).sort_values(by='importance', ascending=False)

    # print(feature_importance)

    start += step_size

# Print average Mean Squared Error
print(f'Average Mean Squared Error: {sum(mse_list) / len(mse_list)}')