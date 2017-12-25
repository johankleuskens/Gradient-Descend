import numpy as np
import matplotlib
import matplotlib.pyplot as plt


X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100,1)
X_b = np.c_[np.ones((100 ,1)), X]

#Calculate Normal Equation
theta_norm = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

#Make predictions
X_new_norm = np.array([[0], [2]])
X_new_norm_b = np.c_[np.ones((2,1)), X_new_norm]
y_pred_norm = X_new_norm_b.dot(theta_norm)

# Do a gradient descent
eta =0.1
n_iterations = 1000
m = 100

theta_grad = np.random.rand(2,1)

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_grad) - y)
    theta_grad = theta_grad - eta * gradients

# Make prediction     
X_new_grad = np.array([[0], [2]])
X_new_grad_b = np.c_[np.ones((2,1)), X_new_norm]
y_pred_grad = X_new_grad_b.dot(theta_grad)
    

# show things
fig = plt.figure()
fig.subplots_adjust(hspace=0.5)

ax1 = fig.add_subplot(211)
ax1.set_title('Normal Equation')
ax1.plot(X,y,"b.")
ax1.plot(X_new_norm, y_pred_norm, "r-")
ax1.axis([0,2,0,15])


ax2 = fig.add_subplot(212)
ax2.set_title('Gradient Descent')
ax2.plot(X,y,"b.")
ax2.plot(X_new_grad, y_pred_grad, "y-")
ax2.axis([0,2,0,15])
plt.show()
