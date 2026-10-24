
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# 1. Product dataset
data = {
    "product": [
        "Headphones A",
        "Headphones B",
        "Earbuds C",
        "Speaker D",
        "Speaker E"
    ],
    "price": [2000, 2200, 1800, 5000, 4800],
    "rating": [4.5, 4.3, 4.4, 4.7, 4.6]
}

df = pd.DataFrame(data)

# 2. Select features
X = df[["price", "rating"]]

# 3. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Give price more importance than rating
price_weight = 2.0
X_scaled[:, 0] *= price_weight

# 5. Create KNN model
model = NearestNeighbors(
    n_neighbors=4,
    metric="euclidean"
)
model.fit(X_scaled)

# 6. Get user input
price = float(input("Enter product price: "))
rating = float(input("Enter product rating (0-5): "))

# 7. Validate input
if price < 0:
    raise ValueError("Price cannot be negative.")

if not 0 <= rating <= 5:
    raise ValueError("Rating must be between 0 and 5.")

# 8. Prepare user input
user_input = pd.DataFrame({
    "price": [price],
    "rating": [rating]
})

# 9. Scale using the same scaler and feature weighting
user_scaled = scaler.transform(user_input)
user_scaled[:, 0] *= price_weight

# 10. Find nearest products
distances, indices = model.kneighbors(user_scaled)

# 11. Display recommendations
print("\nRecommended similar products:")

for rank, (index, distance) in enumerate(
    zip(indices[0], distances[0]),
    start=1
):
    product = df.iloc[index]

    print(
        f"{rank}. {product['product']} | "
        f"Price: ₹{product['price']} | "
        f"Rating: {product['rating']} | "
        f"Distance: {distance:.3f}"
    )
