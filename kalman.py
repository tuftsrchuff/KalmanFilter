#Specifying imports
print("Starting filter program...")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Specify envs here

#Animation to plot incrementally for each time step
def animate(i):
    #Clear previous plot
    ax.clear()

    #Plot to desired time step
    plt.plot(x[:i+1], true_states[0][:i+1], label='True State', markersize=8, color='blue')
    plt.plot(x[:i+1], measurements[:i+1], 'rx', label='Measurements', markersize=8, color='red')
    plt.plot(x[:i+1], filtered_states[0][:i+1], label='Filtered State', color='green')
    plt.title('Kalman Filter Position')
    plt.xlabel('Time Steps')
    plt.ylabel('Position')
    plt.legend()


# Kalman filter parameters
dt = 1.0  # Time step
A = np.array([[1, dt], [0, 1]])  # State transition matrix
H = np.array([[1, 0]])  # Measurement matrix
Q = 0.1 * np.eye(2)  # Process noise covariance
R = 1.0  # Measurement noise covariance

# Initial state estimate
x_hat = np.array([[0], [0]])

# Initial covariance estimate
P = np.eye(2)

# Number of time steps
num_steps = 50
x = np.arange(num_steps)

# True state (simulated)
true_states = np.linspace(0, 2 * np.pi, num_steps)
true_positions = np.cos(true_states)
true_states = np.vstack((true_states, true_positions))

# Measurements with noise
measurements = true_states[0, :] + np.sqrt(R) * np.random.randn(num_steps)

# Kalman filter loop
filtered_states = np.zeros((2, num_steps))

for k in range(num_steps):
    # Prediction step
    x_hat_minus = A.dot(x_hat)
    P_minus = A.dot(P).dot(A.T) + Q

    # Update step
    K = P_minus.dot(H.T).dot(np.linalg.inv(H.dot(P_minus).dot(H.T) + R))
    x_hat = x_hat_minus + K.dot(measurements[k] - H.dot(x_hat_minus))
    P = (np.eye(2) - K.dot(H)).dot(P_minus)

    filtered_states[:, k] = x_hat.flatten()


# Create a figure and axis
fig, ax = plt.subplots()

#Save animation as gif
animation = FuncAnimation(fig, animate, frames=num_steps, interval=400, repeat=False)
animation.save('kalman_filter_animation.gif', writer='pillow', fps=5)
plt.show()
