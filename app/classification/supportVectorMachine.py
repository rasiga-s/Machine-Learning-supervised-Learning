
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# 1. Training data
data = {
    "monthly_income": [
        20000, 25000, 30000, 35000,
        45000, 50000, 60000, 70000,
        22000, 28000, 48000, 65000
    ],
    "website_visits": [
        2, 3, 4, 3, 7, 8, 9, 10,
        1, 2, 6, 9
    ],
    "purchased": [
        "No", "No", "No", "No",
        "Yes", "Yes", "Yes", "Yes",
        "No", "No", "Yes", "Yes"
    ]
}

df = pd.DataFrame(data)

# 2. Separate features and target
X = df[["monthly_income", "website_visits"]]
y = df["purchased"]

# 3. Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. Scale features and train the SVM model
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear")
)

model.fit(X_train, y_train)

# 5. Evaluate the model
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

# 6. Predict for a new customer
customer = pd.DataFrame({
    "monthly_income": [40000],
    "website_visits": [5]
})

result = model.predict(customer)[0]

print("\n--- Purchase Prediction ---")
print("Monthly income: ₹40,000")
print("Website visits: 5")
print("Predicted purchase:", result)
