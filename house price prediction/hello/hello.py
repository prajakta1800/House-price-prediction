import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading dataset...")
df = pd.read_csv('housing_price_dataset.csv')
print(df.head())

Neighborhood_map = {'Rural': 0, 'Suburb': 1, 'Urban': 2}
df['Neighborhood'] = df['Neighborhood'].map(Neighborhood_map)
print(df)
print("Preprocessing data...")
# df.dropna(inplace=True)
# scaler = StandardScaler()
# df[[ 'Bedrooms', 'Bathrooms']] = scaler.fit_transform(df[[ 'Bedrooms', 'Bathrooms']])
print(df.head())

print("Splitting data...")
X = df.drop('Price', axis=1)
y = df['Price']
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("Training models...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("Linear Regression model trained")

rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
print("Random Forest model trained")


print("Making predictions...")
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

print("Evaluating models...")

lr_mse = mean_squared_error(y_test, lr_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
print(f"Linear Regression MSE: {lr_mse}")
print(f"Random Forest MSE: {rf_mse}")

plt.figure(figsize=(10, 6))
sns.barplot(x='SquareFeet', y='Price', data=df)
plt.title('Average House Price by Number of SquareFeet')
plt.xlabel('Number of SquareFeet')
plt.ylabel('Average House Price')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(X='SquareFeet', y='Price', data=df)
plt.title('  SquareFeet vs. Price')
plt.xlabel(' SquareFeet')
plt.ylabel('House Price ($)')
plt.show()