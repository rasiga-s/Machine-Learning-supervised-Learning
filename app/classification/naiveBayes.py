
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = [
    "win money now",
    "claim your free prize",
    "free offer click now",
    "win a free gift",
    "meeting scheduled for tomorrow",
    "please send the project report",
    "let us discuss the project",
    "your appointment is confirmed",
    "claim your prize now",
    "team meeting starts tomorrow",
    "free money offer",
    "please review the report"
]

labels = [
    "Spam", "Spam", "Spam", "Spam",
    "Not Spam", "Not Spam", "Not Spam", "Not Spam",
    "Spam", "Not Spam", "Spam", "Not Spam"
]

# Train the model
model = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)

model.fit(emails, labels)

print("Naive Bayes model training completed!")

# Get input from user
user_email = input("\nEnter your email message: ")

# Predict
prediction = model.predict([user_email])[0]
probabilities = model.predict_proba([user_email])[0]

print("\n--- Email Prediction ---")
print("Email:", user_email)
print("Predicted class:", prediction)

print("\nClass probabilities:")
for label, probability in zip(model.classes_, probabilities):
    print(f"{label}: {probability:.2%}")
