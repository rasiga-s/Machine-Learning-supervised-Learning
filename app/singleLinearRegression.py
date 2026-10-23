
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# 1. Load the dataset
data = pd.read_csv("data/house_prices.csv")

# 2. Separate input (X) and target (y)
X = data[["area"]]
y = data["price"]

# 3. Train the model
model = LinearRegression()
model.fit(X, y)

print("House Price Prediction Model is ready!")

# 4. Get input from the user
while True:
    user_input = input(
        "\nEnter house area in sq ft (or type 'exit' to quit): "
    )

    if user_input.lower() == "exit":
        print("Thank you for using the model!")
        break

    try:
        area = float(user_input)

        if area <= 0:
            print("Please enter an area greater than zero.")
            continue

        # 5. Predict the price
        input_data = pd.DataFrame({"area": [area]})
        predicted_price = model.predict(input_data)

        # 6. Display the result
        print(f"House area: {area:.0f} sq ft")
        print(f"Predicted house price: ₹{predicted_price[0]:.2f} lakh")

    except ValueError:
        print("Invalid input. Please enter a number or 'exit'.")

# 7. Save the trained model
joblib.dump(model, "house_price_model.pkl")
print("Model saved successfully!")
