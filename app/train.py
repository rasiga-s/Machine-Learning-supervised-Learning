import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("data/house_prices.csv")

# Separate input (X) and target (y)
X = data[["area"]]
y = data["price"]

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict the price of a 2000 sq ft house
predicted_price = model.predict(pd.DataFrame({"area": [2000]}))

print(f"Model coefficient: {model.coef_[0]}")
print(f"Model intercept: {model.intercept_}")
print(f"Predicted price: ₹{predicted_price[0]:.2f} lakh")

# Save the trained model
import joblib

joblib.dump(model, "house_price_model.pkl")
print("Model saved successfully!")