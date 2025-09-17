import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

pause = False


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


def slider(fig, position, label, valmin, valmax, valinit):
    ax_slider = fig.add_axes(position)
    slider = Slider(ax_slider, label, valmin, valmax, valinit=valinit)

    return slider


def toggle_pause(event):
    global pause
    pause = not pause  # flip state
    if pause:
        animation.event_source.stop()
        btn_pause.label.set_text("Resume")
    else:
        animation.event_source.start()
        btn_pause.label.set_text("Pause")
    fig.canvas.draw_idle()


#========Main Plot=============
fig = plt.figure(figsize=(12, 6))
fig.patch.set_facecolor("#e6e6e6")
main_plot = fig.add_axes([0.1, 0.2, 0.55, 0.7], facecolor="#f5f2e9") # [left, bottom, width, height]
main_plot.set_title("Gradient Descent Visualizer")
plt.grid()

#========Control Panel=========
control_panel = fig.add_axes([0.7, 0.2, 0.25, 0.7], facecolor="#f5f2e9")  # [left, bottom, width, height]
control_panel.set_xticks([])
control_panel.set_yticks([])
control_panel.text(0.05, 0.95, "Controls", fontsize=12, fontweight='bold', va="top")
functext = control_panel.text(0.05, 0.38, "x= 4.0,  f(x)= 16", fontsize=11)
steptext = control_panel.text(0.05, 0.3, "Steps= 0", fontsize=11)

fig._lrslider = slider(fig, [0.75, 0.75, 0.17, 0.04], "lr=     ", 0.1, 1.0, 0.5)
fig._stepSlider = slider(fig, [0.75, 0.65, 0.17, 0.04], "steps=", 5, 60, 20)

ax_ent = fig.add_axes([0.73, 0.55, 0.08, 0.05]) # [left, bottom, width, height]
btn_enter = Button(ax_ent, "Enter")
ax_pause = fig.add_axes([0.85, 0.55, 0.08, 0.05])
btn_pause = Button(ax_pause, "Pause")

btn_pause.on_clicked(toggle_pause)

  #  fig._enter = enter(fig)


lr = 0.1
x = 4.0
steps = 20

history = []

for i in range(steps):
    x, g, f_val = grad_step(x, lr)
    history.append((i, x, f_val, g))

x_val = [j[1] for j in history]

y_val = [j[2] for j in history]

    # parabola points
x_range = np.linspace(-5, 5, 100)
y_range = func(x_range)

main_plot.plot(x_range, y_range, label="y=x^2")
descent_path, = main_plot.plot([], [], "o-", color = "red", ms = 5, label="Descent Path")
converged_point, = main_plot.plot([], [], marker = 'o', color = "red", ms = 10)



total_frames = (len(x_val) + 1) * steps #420

def update(frames):
    stair = frames // steps
    t = (frames % steps) / steps #to get range from 0 to 1
    steptext.set_text(f"Steps = {stair}")

    if stair < len(x_val) - 1:
        # interpolation formula!!
        x_curr = (1 - t) * x_val[stair] + t * x_val[stair + 1]
        y_curr = func(x_curr)
        functext.set_text(f"x={x_curr:.2f}, f(x)={y_curr:.2f}")
    else:
        x_curr, y_curr = x_val[-1], y_val[-1]

    xs = list(x_val[:stair+1]) + [x_curr]
    ys = list(y_val[:stair+1]) + [y_curr]
    descent_path.set_data(xs, ys)

    converged_point.set_data([x_curr], [y_curr])

    if frames >= total_frames -1:
        converged_point.set_markeredgecolor("blue")
        control_panel.text(0.05, 0.05, "Converged!", fontsize=11, fontweight="bold", color="red")

    return descent_path, converged_point


animation = FuncAnimation(
    fig = fig,
    func = update,
    frames = total_frames,
    interval = steps,
    repeat=False
)

  #  fig.legend(loc='upper left', bbox_to_anchor=(1.05, 0.95), borderaxespad=0)

leg = fig.legend(labelspacing=1)
leg.set_bbox_to_anchor((0.83, 0.4))
#leg.set(labelspacing=1.5)
leg.get_frame().set_facecolor('none')
leg.get_frame().set_edgecolor('none')
plt.show()
plt.close()


#if __name__ == '__main__':
 #   main()

