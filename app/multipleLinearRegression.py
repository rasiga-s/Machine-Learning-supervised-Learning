
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Load the dataset
data = pd.read_csv("data/house_prices.csv")

# 2. Separate input (X) and target (y)
X = data[["area", "bedrooms", "age"]]
y = data["price"]

# 3. Train the model
model = LinearRegression()
model.fit(X, y)

# Find the supported area range from training data
min_area = data["area"].min()
max_area = data["area"].max()

print("\n=== House Price Prediction ===")

# 4. Get user input
try:
    area = float(input("Enter house area (sq ft): "))
    bedrooms = int(input("Enter number of bedrooms: "))
    age = float(input("Enter house age (years): "))

    # 5. Validate user input
    if area <= 0 or bedrooms <= 0 or age < 0:
        print("Please enter a valid area, bedroom count, and age.")

    elif area < min_area or area > max_area:
        print(
            f"Area must be between {min_area:.0f} "
            f"and {max_area:.0f} sq ft for this model."
        )

    else:
        # 6. Prepare input for prediction
        new_house = pd.DataFrame({
            "area": [area],
            "bedrooms": [bedrooms],
            "age": [age]
        })

        # 7. Predict price
        predicted_price = model.predict(new_house)[0]

        print("\n=== Prediction Result ===")
        print(f"Area: {area:.0f} sq ft")
        print(f"Bedrooms: {bedrooms}")
        print(f"House age: {age:g} years")

        if predicted_price < 0:
            print(
                f"Warning: The model predicted an unrealistic "
                f"price of ₹{predicted_price:.2f} lakh."
            )
        else:
            print(f"Predicted price: ₹{predicted_price:.2f} lakh")

except ValueError:
    print("Invalid input. Please enter numeric values.")
