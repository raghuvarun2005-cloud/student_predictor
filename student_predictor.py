import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------
# Step 1: Create Dataset
# -------------------------------
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Score': [35, 45, 48, 52, 60, 68, 75, 82, 88, 95]
}

df = pd.DataFrame(data)

# -------------------------------
# Step 2: Explore Data
# -------------------------------
print("===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA STATISTICS =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# -------------------------------
# Step 3: Visualize Data
# -------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(df['Hours'], df['Score'])
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Hours Studied vs Exam Score")
plt.grid(True)
plt.show()

# -------------------------------
# Step 4: Prepare Data
# -------------------------------
X = df[['Hours']]
y = df['Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 5: Train Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Step 6: Predictions
# -------------------------------
y_pred = model.predict(X_test)

print("\n===== ACTUAL VS PREDICTED =====")

for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual} | Predicted: {predicted:.2f}")

# -------------------------------
# Step 7: Evaluate Model
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# -------------------------------
# Step 8: Predict New Student Score
# -------------------------------
hours = float(input("\nEnter study hours: "))

predicted_score = model.predict([[hours]])

print(
    f"Predicted Exam Score for {hours} hours of study: "
    f"{predicted_score[0]:.2f}"
)

# -------------------------------
# Step 9: Regression Line
# -------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Actual Data")
plt.plot(X, model.predict(X), label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Student Performance Predictor")
plt.legend()
plt.grid(True)
plt.show()