import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("train.csv")

# Keep useful columns
df = df[["Pclass","Sex","Age","Fare","Embarked","Survived"]]

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode categorical columns
df["Sex"] = df["Sex"].map({"male":1,"female":0})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Convert bool columns to int
for c in df.columns:
    if df[c].dtype == bool:
        df[c] = df[c].astype(int)

# Split features and target
X = df.drop("Survived", axis=1).to_numpy(dtype=float)
y = df["Survived"].to_numpy(dtype=float)

# Shuffle data
np.random.seed(42)
idx = np.random.permutation(len(X))
X = X[idx]
y = y[idx]

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Standardize using training data
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize parameters
weights = np.zeros(X_train.shape[1])
bias = 0.0
lr = 0.01
epochs = 1000
loss_history = []

# Train model
for epoch in range(epochs):
    linear = np.dot(X_train, weights) + bias
    pred = sigmoid(linear)

    loss = -np.mean(
        y_train*np.log(pred + 1e-10) +
        (1-y_train)*np.log(1-pred + 1e-10)
    )
    loss_history.append(loss)

    dw = np.dot(X_train.T, (pred - y_train)) / len(y_train)
    db = np.mean(pred - y_train)

    weights -= lr * dw
    bias -= lr * db

# Predict
test_prob = sigmoid(np.dot(X_test, weights) + bias)
test_pred = (test_prob >= 0.5).astype(int)

# Accuracy
accuracy = np.mean(test_pred == y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Confusion matrix
tp = np.sum((test_pred==1)&(y_test==1))
tn = np.sum((test_pred==0)&(y_test==0))
fp = np.sum((test_pred==1)&(y_test==0))
fn = np.sum((test_pred==0)&(y_test==1))

print("\nConfusion Matrix")
print([[tn, fp],[fn, tp]])

# Plot loss
plt.figure(figsize=(6,4))
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Predict new passenger
new = np.array([[3,1,22,7.25,0,1]], dtype=float)
new = (new - mean) / std
prob = sigmoid(np.dot(new, weights) + bias)[0]

print("\nNew Passenger Probability:", round(prob,4))
print("Prediction:", "Survived" if prob>=0.5 else "Not Survived")
