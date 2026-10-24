
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# 1. Historical loan application data
data = {
    "person_name": [
        "Arun", "Priya", "Rahul", "Divya",
        "Karthik", "Meena", "Vijay", "Sneha"
    ],
    "income": [
        25000, 30000, 45000, 60000,
        80000, 35000, 70000, 20000
    ],
    "credit_score": [
        550, 600, 650, 720,
        780, 580, 750, 500
    ],
    "existing_loans": [
        3, 2, 1, 0, 0, 3, 1, 4
    ],
    "loan_status": [
        "Rejected", "Rejected", "Approved", "Approved",
        "Approved", "Rejected", "Approved", "Rejected"
    ]
}

df = pd.DataFrame(data)

print("Historical loan records:")
print(df.to_string(index=False))

# 2. Features (X) - information used for prediction
X = df[["income", "credit_score", "existing_loans"]]

# 3. Target (y) - known outcome
y = df["loan_status"]

# 4. Create and train the Decision Tree
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X, y)

print("\nModel training completed!")

# 5. Enter a new applicant
person_name = input("\nEnter applicant name: ")
income = float(input("Enter monthly income (₹): "))
credit_score = int(input("Enter credit score (300-900): "))
existing_loans = int(input("Enter existing loans: "))

# 6. Validate inputs
if income < 0:
    raise ValueError("Income cannot be negative.")

if not 300 <= credit_score <= 900:
    raise ValueError("Credit score must be between 300 and 900.")

if existing_loans < 0:
    raise ValueError("Existing loans cannot be negative.")

# 7. Prepare the new applicant's features
applicant = pd.DataFrame({
    "income": [income],
    "credit_score": [credit_score],
    "existing_loans": [existing_loans]
})

# 8. Predict
prediction = model.predict(applicant)[0]

print("\n--- Loan Prediction ---")
print("Applicant:", person_name)
print("Predicted loan status:", prediction)
