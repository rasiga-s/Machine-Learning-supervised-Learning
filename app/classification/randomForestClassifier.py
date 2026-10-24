
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Historical loan application data
data = {
    "person_name": [
        "Arun", "Priya", "Rahul", "Divya",
        "Karthik", "Meena", "Vijay", "Sneha",
        "Anita", "Ramesh", "Kavya", "Mohan"
    ],
    "income": [
        25000, 30000, 45000, 60000,
        80000, 35000, 70000, 20000,
        55000, 40000, 90000, 50000
    ],
    "credit_score": [
        550, 600, 650, 720,
        780, 580, 750, 500,
        700, 620, 810, 690
    ],
    "existing_loans": [
        3, 2, 1, 0, 0, 3, 1, 4, 1, 2, 0, 1
    ],
    "loan_status": [
        "Rejected", "Rejected", "Approved", "Approved",
        "Approved", "Rejected", "Approved", "Rejected",
        "Approved", "Rejected", "Approved", "Approved"
    ]
}

df = pd.DataFrame(data)

print("Historical loan records:")
print(df.to_string(index=False))

# 2. Select features and target
X = df[["income", "credit_score", "existing_loans"]]
y = df["loan_status"]

# 3. Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42
)

# 4. Train the model
model.fit(X, y)

print("\nRandom Forest training completed!")

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

# 7. Prepare applicant features
applicant = pd.DataFrame({
    "income": [income],
    "credit_score": [credit_score],
    "existing_loans": [existing_loans]
})

# 8. Predict loan status
prediction = model.predict(applicant)[0]

# 9. Display results
print("\n--- Loan Prediction ---")
print("Applicant:", person_name)
print("Predicted loan status:", prediction)
