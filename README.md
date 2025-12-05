#Gradient Descent Visualizer
A simple yet powerful visualization of Gradient Descent implemented in Python using Matplotlib.
This project animates how Gradient Descent moves step by step on the parabola function 
𝑦 = 𝑥^2

#Features:
Smooth animation of the descent path (ball rolling down the slope).
Highlights the converged point (minimum).
Clear plot with function curve, descent path, and moving dot.

#How It Works
Start with a random initial value of 𝑥 (default: 4.0).
At each iteration:
Compute the gradient f'(x) = 2x
Update x using x_new = x - lr * g
Plot the function y = x^2
Animate the movement of the point along the descent path until convergence.

#Requirments:
Along with python, install matplotlib and NumPy.

Note: Please check the OldVersion branch, main branch is still under working. Thank You!
