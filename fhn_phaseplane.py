# Imports for plotting, animation, math, and system exit
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import numpy as np
import sys

# Generates a gradient of RGB colors from c1 to c2 based on normalized array values
def colgen(c1, c2, arr):
    mi = min(arr)
    ma = max(arr)
    res = []
    lerp = lambda a, b, t: (b - a) * t + a  # Linear interpolation
    for i in arr:
        t = (i - mi) / (ma - mi)  # Normalize value between 0 and 1
        c = (lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t))
        res.append(c)
    return res

# Create two vertically stacked subplots: phase plane and voltage-time graph
fig, ax = plt.subplots(2)

# FitzHugh–Nagumo model parameters
b = 1
a = 0
eps = 0.01
I = 0  # External input current

# Define the system's differential equations (v' and w') and nullclines
xderiv = lambda x,y: -x*(x-a)*(x-1)-y+I  # dv/dt
yderiv = lambda x,y: eps*(x-b*y)         # dw/dt
zerox = lambda x: -x*(x-a)*(x-1)+I       # v-nullcline: where dv/dt = 0
zeroy = lambda x: x/b                    # w-nullcline: where dw/dt = 0

# Define plot limits for phase plane
xrange = (-0.3, 1.1)
yrange = (-0.1, 0.3)

# Plot placeholders for nullclines
vnull, = ax[0].plot([], [], label='v nullcline')
wnull, = ax[0].plot([], [], label='w nullcline')

# Generate a grid of points for vector field visualization
n = 20
xvec, yvec = np.meshgrid(np.linspace(xrange[0], xrange[1], n), np.linspace(yrange[0], yrange[1], n))
xvec = xvec.flatten()
yvec = yvec.flatten()
dummy = np.empty(len(xvec))  # Placeholder for initial zero-length vectors

# Plot the initial empty vector field
vecfield = ax[0].quiver(xvec, yvec, dummy, dummy)

# Draw x=0 and y=0 dashed lines
ax[0].plot([0, 0], [yrange[0], yrange[1]], 'b--')
ax[0].plot([xrange[0], xrange[1]], [0, 0], 'b--')

# Set axis limits and labels
ax[0].set_xlim(xrange[0] - 0.1, xrange[1] + 0.1)
ax[0].set_ylim(yrange[0] - 0.1, yrange[1] + 0.1)
ax[0].set_xlabel('v')
ax[0].set_ylabel('w')
ax[0].legend()

# Updates the phase plane: recomputes nullclines and vector field
def update_phaseplane():
    xplot = np.arange(xrange[0], xrange[1], 0.01)
    vnull.set_data(xplot, zerox(xplot))
    wnull.set_data(xplot, zeroy(xplot))
    u = xderiv(xvec, yvec)
    v = yderiv(xvec, yvec)
    vecfield.set_UVC(u, v)
    vecfield.set_facecolor(colgen((1, 0, 0), (0, 1, 0), [u[i] for i in range(len(u))]))

# Initial update of the vector field and nullclines
update_phaseplane()

# Initialize time and simulation state
t = 0
dt = 0.05
cur = (0.02, 0)  # Initial (v, w) point

# Setup for phase plane dot (moving point)
box = ax[0].get_window_extent()
val_width = xrange[1] - xrange[0]
val_height = yrange[1] - yrange[0]
circ_size = 7
dot = Ellipse((cur[0], cur[1]), (circ_size / box.width) * val_width, (circ_size / box.height) * val_height, color='red')
ax[0].add_patch(dot)

# Setup for trail (trajectory) in phase plane
streak, = ax[0].plot([], [], color='black')
streak_x = [cur[0]]
streak_y = [cur[1]]
max_streak_len = 1000  # Maximum number of trail points to keep

# Setup for voltage over time graph (bottom subplot)
curve, = ax[1].plot([], [])
ax[1].set_xlim(-30, 1000)
ax[1].set_ylim(-0.6, 1.5)
vx = [t]
vy = [cur[0]]

# Controls how many steps happen per animation frame
updates_per_frame = 10
update_graph = False  # Flag to trigger recomputation of vector field
stim_cur = (False, 0)  # (stimulate?, strength)

# Animation update function: advances the simulation state
def update(frame):
    global t, cur, streak_x, streak_y, update_graph, stim_cur
    for _ in range(updates_per_frame):
        t += dt
        curv = cur[0]
        curw = cur[1]

        # Apply one-time stimulation if requested
        if stim_cur[0]:
            curv += stim_cur[1]
            stim_cur = (False, 0)

        # Update state using Euler method
        newv = curv + xderiv(curv, curw) * dt
        neww = curw + yderiv(curv, curw) * dt

        # Update voltage-time trace
        vx.append(t)
        vy.append(newv)

        # Update current state
        cur = (newv, neww)

        # Update red dot's position and size
        box = ax[0].get_window_extent()
        dot.set_center((newv, neww))
        dot.set_width((circ_size / box.width) * val_width)
        dot.set_height((circ_size / box.height) * val_height)

        # Add point to trajectory trail
        streak_x.append(newv)
        streak_y.append(neww)

        # Limit trail length
        streak_len = len(streak_x)
        if streak_len > max_streak_len:
            streak_x = streak_x[streak_len - max_streak_len:]
            streak_y = streak_y[streak_len - max_streak_len:]

    # Update trajectory line and voltage curve
    streak.set_data(streak_x, streak_y)
    curve.set_data(vx, vy)

    # Scroll x-axis of voltage graph if needed
    if vx[-1] > 1000:
        ax[1].set_xlim(vx[-1] - 1030, vx[-1])

    # Update vector field and nullclines if needed
    if update_graph:
        update_phaseplane()
        update_graph = False

    return dot, streak, curve, vnull, wnull, vecfield

# Exit program if window is closed
def on_close(event):
    sys.exit()

# Start animation
func = FuncAnimation(fig, update, interval=1, blit=True)
fig.canvas.mpl_connect('close_event', on_close)
plt.show(block=False)

# Print spacing to separate GUI from terminal input
print('\n'*3)

# Interactive loop to allow user to stimulate system or modify parameters
while True:
    command = input(
'''
Enter a value to give an instant stimulation,
or type a line of code (e.g. I=1.0 or b=2)
to update fhn parameters:\n
'''
    )
    stim = None
    try:
        stim = float(command)  # Try to parse as numeric stimulation
    except:
        pass
    if stim is None:
        # Try to execute the user-provided code (e.g., modifying 'a', 'b', 'I', etc.)
        try:
            exec(command)
        except:
            print('invalid command')
        update_graph = True  # Trigger a redraw of vector field and nullclines
    else:
        stim_cur = (True, stim)  # Schedule an instantaneous voltage bump

