
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Training data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([3, 5, 8, 13, 20])

# Convert input into polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train the model
model = LinearRegression()
model.fit(X_poly, y)

# Get input from the user
experience = float(input("Enter years of experience: "))

# Transform the user input
user_input = np.array([[experience]])
user_input_poly = poly.transform(user_input)

# Predict salary
predicted_salary = model.predict(user_input_poly)

print(
    f"Predicted salary for {experience} years of experience: "
    f"{predicted_salary[0]:.2f} lakhs"
)

# Visualize the model and user prediction
X_curve = np.linspace(1, 10, 100).reshape(-1, 1)
X_curve_poly = poly.transform(X_curve)
y_curve = model.predict(X_curve_poly)

plt.scatter(X, y, label="Training data")
plt.plot(X_curve, y_curve, label="Polynomial prediction")
plt.scatter(
    [experience],
    predicted_salary,
    color="red",
    label="Your prediction"
)

plt.xlabel("Years of experience")
plt.ylabel("Salary (lakhs)")
plt.title("Polynomial Regression")
plt.legend()
plt.show()
