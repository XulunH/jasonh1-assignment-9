import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function

        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01  # (2,3)
        self.b1 = np.zeros((1, hidden_dim))  # (1,3)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01  # (3,1)
        self.b2 = np.zeros((1, output_dim))  # (1,1)

    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation function")

    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        else:
            raise ValueError("Unsupported activation function")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.X = X  # Store input for use in backward pass
        self.Z1 = np.dot(X, self.W1) + self.b1  # (N,3)
        self.A1 = self.activation(self.Z1)      # (N,3)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # (N,1)
        self.A2 = self.sigmoid(self.Z2)  # Output activation (sigmoid for binary classification)
        return self.A2  # (N,1)

    def backward(self, X, y):
        m = y.shape[0]  # Number of examples

        # Compute dZ2 = A2 - y
        dZ2 = (self.A2 - y) / m  # (N,1)

        # Compute gradients for W2 and b2
        dW2 = np.dot(self.A1.T, dZ2)  # (3,N) x (N,1) = (3,1)
        db2 = np.sum(dZ2, axis=0, keepdims=True)  # (1,1)

        # Backpropagate to hidden layer
        dA1 = np.dot(dZ2, self.W2.T)  # (N,1) x (1,3) = (N,3)
        dZ1 = dA1 * self.activation_derivative(self.Z1)  # (N,3) * (N,3)

        # Compute gradients for W1 and b1
        dW1 = np.dot(X.T, dZ1)  # (2,N) x (N,3) = (2,3)
        db1 = np.sum(dZ1, axis=0, keepdims=True)  # (1,3)

        # Update weights and biases
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # Store gradients for visualization
        self.dW1 = dW1
        self.db1 = db1
        self.dW2 = dW2
        self.db2 = db2

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden features
    hidden_features = mlp.A1  # (N,3)
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                      c=y.ravel(), cmap='bwr', alpha=0.7)

    # Hyperplane visualization in the hidden space
    W2 = mlp.W2.flatten()
    b2 = mlp.b2.flatten()
    x_range = np.linspace(np.min(hidden_features[:, 0]), np.max(hidden_features[:, 0]), 10)
    y_range = np.linspace(np.min(hidden_features[:, 1]), np.max(hidden_features[:, 1]), 10)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    # Avoid division by zero
    if W2[2] != 0:
        Z_grid = (-W2[0]*X_grid - W2[1]*Y_grid - b2[0]) / W2[2]
        ax_hidden.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.3)
    ax_hidden.set_xlabel('Hidden Unit 1')
    ax_hidden.set_ylabel('Hidden Unit 2')
    ax_hidden.set_zlabel('Hidden Unit 3')
    ax_hidden.set_title('Hidden Layer Feature Space')

    # Plot input layer decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # Forward pass for the grid
    Z1_grid = np.dot(grid, mlp.W1) + mlp.b1
    A1_grid = mlp.activation(Z1_grid)
    Z2_grid = np.dot(A1_grid, mlp.W2) + mlp.b2
    A2_grid = mlp.sigmoid(Z2_grid)
    Z = A2_grid.reshape(xx.shape)
    ax_input.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_xlabel('X1')
    ax_input.set_ylabel('X2')
    ax_input.set_title('Decision Boundary in Input Space')

    # Visualize features and gradients as circles and edges
    ax_gradient.axis('off')
    ax_gradient.set_title('Network Gradients')

    # Positions of nodes
    input_layer_positions = [(0, i) for i in range(2)]
    hidden_layer_positions = [(1, i) for i in range(3)]
    output_layer_position = (2, 1)

    # Plot nodes
    for pos in input_layer_positions:
        circle = Circle(pos, radius=0.1, fill=True, color='lightgray', ec='k')
        ax_gradient.add_patch(circle)
    for pos in hidden_layer_positions:
        circle = Circle(pos, radius=0.1, fill=True, color='lightgray', ec='k')
        ax_gradient.add_patch(circle)
    circle = Circle(output_layer_position, radius=0.1, fill=True, color='lightgray', ec='k')
    ax_gradient.add_patch(circle)

    # Plot edges from input to hidden layer
    max_grad_w1 = np.max(np.abs(mlp.dW1))
    for i, input_pos in enumerate(input_layer_positions):
        for j, hidden_pos in enumerate(hidden_layer_positions):
            grad = abs(mlp.dW1[i, j]) / (max_grad_w1 + 1e-8)
            linewidth = grad * 5  # Scale for visualization
            ax_gradient.plot([input_pos[0], hidden_pos[0]], [input_pos[1], hidden_pos[1]], 'k-', linewidth=linewidth)

    # Plot edges from hidden to output layer
    max_grad_w2 = np.max(np.abs(mlp.dW2))
    for i, hidden_pos in enumerate(hidden_layer_positions):
        grad = abs(mlp.dW2[i, 0]) / (max_grad_w2 + 1e-8)
        linewidth = grad * 5  # Scale for visualization
        ax_gradient.plot([hidden_pos[0], output_layer_position[0]], [hidden_pos[1], output_layer_position[1]], 'k-', linewidth=linewidth)

    ax_gradient.set_xlim(-0.5, 2.5)
    ax_gradient.set_ylim(-0.5, max(len(input_layer_positions), len(hidden_layer_positions)) + 0.5)

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden,
                                     ax_gradient=ax_gradient, X=X, y=y),
                        frames=step_num // 10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "relu"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
