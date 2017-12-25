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

# Do a stochastic Gradient Descent
n_epochs = 50
t0, t1 = 5, 50

theta_sgd = np.random.randn(2,1)

def learning_schedule(t):
    return t0/(t + t1)
    
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta_sgd) - yi)
        eta = learning_schedule(i + m* epoch)
        theta_sgd = theta_sgd - eta * gradients

# Make prediction     
X_new_sgd = np.array([[0], [2]])
X_new_sgd_b = np.c_[np.ones((2,1)), X_new_norm]
y_pred_sgd = X_new_sgd_b.dot(theta_sgd)
                                 
# show things
fig = plt.figure()
fig.subplots_adjust(hspace=0.5)

ax1 = fig.add_subplot(311)
ax1.set_title('Normal Equation')
ax1.plot(X,y,"b.")
ax1.plot(X_new_norm, y_pred_norm, "r-")
ax1.axis([0,2,0,15])


ax2 = fig.add_subplot(312)
ax2.set_title('Gradient Descent')
ax2.plot(X,y,"b.")
ax2.plot(X_new_grad, y_pred_grad, "y-")
ax2.axis([0,2,0,15])

ax3 = fig.add_subplot(313)
ax3.set_title('Stochastic Gradient Descent')
ax3.plot(X,y,"b.")
ax3.plot(X_new_grad, y_pred_grad, "p-")
ax3.axis([0,2,0,15])

plt.show()
