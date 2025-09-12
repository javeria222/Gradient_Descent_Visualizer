import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

#A hardcoded function with its gradient
def func(x):
    return x ** 2

def grad_f(x):
    return 2 * x

#gradient steps
def grad_step(x, lr):
    g = grad_f(x)
    x_new = x - lr * g
    return x_new, g, func(x_new)

def main():
    lr = 0.1
    x = 4.0
    steps = 20

    history = []

    for i in range(steps):
        x, g, f_val = grad_step(x, lr)
        history.append((i, x, f_val, g))

    x_val = [j[1] for j in history]

    y_val = [j[2] for j in history]

    #parabola points
    x_range = np.linspace(-5, 5, 100)
    y_range = func(x_range)

    fig, axis = plt.subplots()
    axis.plot(x_range, y_range, label="y=x^2")

    descent_path, = axis.plot([], [], "o-", color = "red", ms = 5, label="Descent Path")

    converged_point, = axis.plot([], [], marker = 'o', color = "red", ms = 10)

    total_frames = (len(x_val) + 1) * steps

    def update(frames):
        stair = frames // steps
        t = (frames % steps) / steps #to get range from 0 to 1

        if stair < len(x_val) - 1:
            # interpolation formula!!
            x_curr = (1 - t) * x_val[stair] + t * x_val[stair + 1]
            y_curr = func(x_curr)
        else:
            x_curr, y_curr = x_val[-1], y_val[-1]

        xs = list(x_val[:stair+1]) + [x_curr]
        ys = list(y_val[:stair+1]) + [y_curr]
        descent_path.set_data(xs, ys)

        converged_point.set_data([x_curr], [y_curr])

        if frames >= total_frames -1:
            converged_point.set_markeredgecolor("blue")

        return descent_path, converged_point


    animation = FuncAnimation(
        fig = fig,
        func = update,
        frames = total_frames,
        interval = steps,
        repeat=False
    )

    animation.save("GDV.gif")

    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()

