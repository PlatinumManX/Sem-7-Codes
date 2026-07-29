import numpy as np
import matplotlib.pyplot as plt

# Objective Function
def objective(x1,x2):
    return (-x1**3+6*(x2**2))

# Partial derivatives
def gradient(x1,x2):
    dx1 = -3 * (x1 **2)
    dx2 = 12 * x2
    return x1,x2

# initial points
x1=1
x2=1

# History to store previous values
history = []

learning_rate=0.01
iterations=50

print("Iteration\t x1\t\t x2\t\t f(x)")

for i in range(iterations):

    dx1, dx2 = gradient(x1, x2)

    x1 = x1 - learning_rate * dx1
    x2 = x2 - learning_rate * dx2

    history.append(objective(x1, x2))

    print(f"{i+1}\t\t{x1:.4f}\t\t{x2:.4f}\t\t{objective(x1,x2):.4f}")

# Plotting the values
plt.plot(history)
plt.xlabel("Iteration")
plt.ylabel("Objective Function")
plt.title("Gradient Descent")
plt.grid(True)
plt.show()
