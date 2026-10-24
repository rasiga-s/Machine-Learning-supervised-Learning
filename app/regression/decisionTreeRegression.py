
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# 1. Load CSV data
df = pd.read_csv("data/salary.csv")

# 2. Select input and output
X = df[["experience"]]
y = df["salary"]

# 3. Create the model
model = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

# 4. Train the model
model.fit(X, y)

# 5. Get input from the user
experience = float(input("Enter years of experience: "))

# 6. Predict salary
user_input = pd.DataFrame({
    "experience": [experience]
})

predicted_salary = model.predict(user_input)

print(
    f"Predicted salary: {predicted_salary[0]:.2f} lakhs"
)

# 7. Visualize the model's predictions
X_curve = pd.DataFrame({
    "experience": [
        value / 10 for value in range(10, 101)
    ]
})

y_curve = model.predict(X_curve)

plt.scatter(X, y, label="Training data")
plt.plot(
    X_curve["experience"],
    y_curve,
    label="Decision Tree prediction"
)

plt.scatter(
    [experience],
    predicted_salary,
    color="red",
    label="Your prediction"
)

plt.xlabel("Years of experience")
plt.ylabel("Salary (lakhs)")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
