
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree

# 1. Load the dataset
df = pd.read_csv("data/salary.csv")

# 2. Separate input and output
X = df[["experience"]]
y = df["salary"]

# 3. Create the Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# 4. Train the model
model.fit(X, y)

# 5. Get input from the user
experience = float(input("Enter years of experience: "))

user_input = pd.DataFrame({
    "experience": [experience]
})

# 6. Predict salary
predicted_salary = model.predict(user_input)

print(f"Predicted salary: {predicted_salary[0]:.2f} LPA")

# 7. Generate predictions for the graph# Get the first tree from the Random Forest
one_tree = model.estimators_[0]

# Visualize that tree
# plt.figure(figsize=(20, 10))

# plot_tree(
#     one_tree,
#     feature_names=["experience"],
#     filled=True,
#     rounded=True
# )

# plt.title("One Decision Tree Inside Random Forest")
# plt.show()

X_curve = pd.DataFrame({
    "experience": [
        value / 10 for value in range(10, 101)
    ]
})

y_curve = model.predict(X_curve)

# 8. Plot the results
plt.scatter(
    X["experience"],
    y,
    label="Training data"
)

plt.plot(
    X_curve["experience"],
    y_curve,
    label="Random Forest prediction"
)

plt.scatter(
    [experience],
    predicted_salary,
    color="red",
    label="Your prediction"
)

plt.xlabel("Years of experience")
plt.ylabel("Salary (LPA)")
plt.title("Random Forest Regression")
plt.legend()
plt.show()
