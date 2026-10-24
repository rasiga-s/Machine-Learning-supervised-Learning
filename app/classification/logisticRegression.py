
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Training data
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "result": [
        "Fail", "Fail", "Fail", "Fail", "Pass",
        "Pass", "Pass", "Pass", "Pass", "Pass"
    ]
}

df = pd.DataFrame(data)

# 2. Input feature and target label
X = df[["study_hours"]]
y = df["result"]

# 3. Create the Logistic Regression model
model = LogisticRegression()

# 4. Train the model
model.fit(X, y)

# 5. Get input from the user
hours = float(input("Enter study hours: "))

user_input = pd.DataFrame({
    "study_hours": [hours]
})

# 6. Predict the category
prediction = model.predict(user_input)

# 7. Display the result
print("Predicted result:", prediction[0])

# 8. Display the probability of each category
probabilities = model.predict_proba(user_input)[0]

for label, probability in zip(model.classes_, probabilities):
    print(f"{label}: {probability:.2%}")
