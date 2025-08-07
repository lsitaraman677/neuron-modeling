import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
from classes import Neuron, Oscillator
import random

# match a section of a curve to a different curve
# Parameters:
# match_sec - the section to match as an array of y-values
# match_vals - the curve to match the section to, also as an array of y-values but likely much longer
# start_idx - the index in match_vals from which to start checking for a match
# tol - tolerance that the mean squared error should be under to be considered for a match
# count_num - the number of increments that are worse than the best match before the best match is returned
# inc - the index increment between match checks
# Returns: The index in the second curve where the start of the first curve should be for the best match
def match_function(match_sec, match_vals, start_idx, tol=1e-2, count_num=20, inc=1):
    # Max iterations before curve section exceeds the other curve's length
    max_iters = len(match_vals) - len(match_sec)
    # Store best match as (score, shift index)
    best = (0, 0)
    # Count how many consecutive times the match gets worse
    count = 0
    reached = False
    # Slide match_sec along match_vals starting from start_idx
    for shift in range(start_idx, max_iters, inc):
        s = 0
        # Compute mean squared error between match_sec and slice of match_vals
        for i in range(len(match_sec)):
            s += (match_sec[i] - match_vals[i + shift])**2
        s /= len(match_sec)
        if reached:
            # If we already found a match below tolerance, look for improvements
            if s < best[0]:
                best = (s, shift)
                count = 0  # Reset counter if improved
            else:
                count += 1
                # If no improvement after count_num steps, return best match
                if count > count_num:
                    return best[1]
        else:
            # If first time a good match is found, start tracking improvements
            if s < tol:
                best = (s, shift)
                reached = True
    # return the location of the best match found (lowest mean squared area)
    return best[1]

# Finds two adjacent peaks in a curve
# Parameters:
# vals - a list of y-values describing the curve
# start - the index from which to start looking for peaks
# thresh - the index distance from a maximum value after which a new maximum is considered separate peak
# inc - the index increment when checking for peaks
# mindiff - the min dist between y-vals of two peaks for the second to be considered valid (curve should be periodic)
# Returns: the indices of the two adjacent peaks in the data passed in
def find_adjacent_peaks(vals, start, thresh=500, inc=1, mindiff=0.1):
    # Initialize peak trackers
    maxpoint1 = (vals[start], start)
    maxpoint2 = None
    idx = start
    try:
        while True:
            idx += inc  # Move forward by 'inc' steps
            if idx - maxpoint1[1] < thresh:
                # Still within threshold window from first peak, look for a higher peak
                if vals[idx] > maxpoint1[0]:
                    maxpoint1 = (vals[idx], idx)
            elif abs(vals[idx] - maxpoint1[0]) < mindiff:
                # If value is close to first peak height (within mindiff), treat as candidate for second peak
                if maxpoint2 is None:
                    maxpoint2 = (vals[idx], idx)
                else:
                    if idx - maxpoint2[1] < thresh:
                        # Still within threshold from previous second peak candidate, look for higher
                        if vals[idx] > maxpoint2[0]:
                            maxpoint2 = (vals[idx], idx)
                    else:
                        # Threshold exceeded for second peak, return both peak indices
                        return maxpoint1[1], maxpoint2[1]
    except:
        # If indexing goes out of bounds or another error occurs, plot the values and peaks for debugging
        plt.plot(list(range(len(vals))), vals)
        plt.plot([maxpoint1[1], maxpoint1[1]], [-2, 2])  # Vertical line at first peak
        plt.plot([0, len(vals)], [maxpoint1[0], maxpoint1[0]])  # Horizontal line at first peak height
        plt.show()

# Finds the maximum value of a curve within an interval
# Parameters:
# vals - the y-values in a list
# start, end - the start and end indices to look for the maximum in
# inc - the index increment
# Returns: the index of the maximum value in vals
def find_peak(vals, start, end, inc=1):
    best = (vals[start], start)
    for i in range(start+1, end, inc):
        if vals[i] > best[0]:
            best = (vals[i], i)
    return best[1]

# Finds all peaks of a curve
# Parameters:
# vals - the y-values of the curve
# thresh - the range in which a maximum must be the highest value to be considered a peak,
#          i.e. index i is a peak if vals[i] == max(vals[i-thresh:i+thresh])
# Returns: a list of indices of all peaks found in vals
def find_all_peaks(vals, thresh=int(50/Neuron.dt)):
    peaks = []  # List to store peak indices
    idx = thresh  # Start scanning from index = thresh to avoid boundary issues
    while True:
        cont = False
        if idx >= len(vals):
            break  # Exit if we've reached the end of the data
        # Look backward to see if any of the previous `thresh` values are higher
        for i in range(idx - thresh, idx):
            if vals[i] > vals[idx]:
                idx += 1     # Not a peak, shift forward
                cont = True  # Skip further checks for this idx
                break
        if cont:
            continue  # Try next index if current one is disqualified
        broken = False
        i = idx + 1
        # Look forward `thresh` steps to see if a higher value exists
        while i < idx + thresh:
            if i >= len(vals):
                broken = True  # Reached end of list during lookahead
                break
            if vals[i] > vals[idx]:
                idx = i  # Found a higher value — move idx forward
            i += 1
        if broken:
            break  # Can't continue search — exit
        peaks.append(idx)  # Current idx is a peak
        idx += thresh  # Skip ahead to avoid detecting same peak again
    return peaks  # Return list of all detected peak indices

# Finds the intrinsic period (time for one cycle with no external input) of an oscillator
def get_period(I=None):
    dummy_oscil = None
    if I is None:
        dummy_oscil = Oscillator()
    else:
        dummy_oscil = Oscillator(const_stim=I)
    for i in range(int(400 / Neuron.dt)):
        dummy_oscil.tick()
    dummy_oscil.start_plot()
    for i in range(int(2000 / Neuron.dt)):
        dummy_oscil.tick()
    dat = dummy_oscil.end_plot()
    peaks = find_all_peaks(dat[1])
    T = (peaks[len(peaks) // 2 + 1] - peaks[len(peaks) // 2]) * Neuron.dt
    return T

# Plots a neuron stimulated at a random excitation time with no constant input (I = 0)
def rand_stim_plot():
    n=Neuron()
    n.set_v(0.1)
    n.start_plot()
    excite_time=random.randint(1,500)
    for i in range(int(500 / Neuron.dt)):
        if int(i*Neuron.dt)==excite_time:
            n.set_v(0.1)
        n.tick()
    n.end_plot()
    n.plot_data(show=False)
    plt.plot([excite_time,excite_time],[-0.4,1],linestyle="--")
    plt.show()

# Draws a neuron connection diagram given vals, 
# where osc1.neurons[vals[0]-1] -> osc2.neurons[vals[1]-1] and osc2.neurons[vals[2]-1] -> osc1.neurons[vals[3]-1]
def draw_arrangement(plot, vals):
    for i in [0, 6]:
        for j in [0, 6]:
            plot.add_patch(Circle((i, j), 1, fill=False))
        plot.add_patch(Rectangle((-2, i - 2), 10, 4, fill=False))
    p1 = ((vals[0]-1) * 6 - 0.25, 6)
    p2 = ((vals[1]-1) * 6 - 0.25, 0)
    p3 = ((vals[2]-1) * 6 + 0.25, 0)
    p4 = ((vals[3]-1) * 6 + 0.25, 6)
    x = [p1[0], p3[0]]
    y = [p1[1], p3[1]]
    u = [p2[0] - p1[0], p4[0] - p3[0]]
    v = [p2[1] - p1[1], p4[1] - p3[1]]
    plot.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1.0)
    plot.text(3, 0, 'Osc2', ha='center', va='center', fontsize=12)
    plot.text(3, 6, 'Osc1', ha='center', va='center', fontsize=12)
    plot.set_aspect('equal')

# Plots the phase difference between two oscillator data arrays by finding peaks and interpolating
# Can plot both phase difference and both phases for osc1 and osc2 depending on inputs
def plot_phase_convergence(osc1, osc2, ax1=None, ax2=None, show=True, phase_plot=True, diff_plot=True):
    if ((ax1 is None) and phase_plot) or ((ax2 is None) and diff_plot):
        fig, ax = plt.subplots(2, 1)
        ax1 = ax[0]
        ax2 = ax[1]
    peaks1 = [osc1[0][i] for i in find_all_peaks(osc1[1])]
    peaks2 = [osc2[0][i] for i in find_all_peaks(osc2[1])]
    diff = np.empty(len(osc1[0]))
    idx1 = 0
    idx2 = 0
    for i in range(len(osc1[0])):
        t = osc1[0][i]
        if t > peaks1[idx1]:
            if idx1 < len(peaks1) - 2:
                idx1 += 1
        if t > peaks2[idx2]:
            if idx2 < len(peaks2) - 2:
                idx2 += 1
        p1 = (t - peaks1[idx1]) / (peaks1[idx1 + 1] - peaks1[idx1]) + idx1
        p2 = (t - peaks2[idx2]) / (peaks2[idx2 + 1] - peaks2[idx2]) + idx2
        diff[i] = p2 - p1
        if i > 0:
            prev_gap = round(diff[i] - diff[i-1])
            diff[i] -= prev_gap
        else:
            diff[i] = (10 + diff[i]) % 1
    y1 = list(range(len(peaks1)))
    y2 = list(range(len(peaks2)))
    if phase_plot:
        ax1.plot(peaks1, y1)
        ax1.plot(peaks2, y2)
        ax1.set_ylabel('phase')
        ax1.grid(True, which='both', linestyle='-', color='gray', alpha=0.3)
    if diff_plot:
        ax2.plot(osc1[0], diff)
        ax2.set_ylim(-1, 1)
        ax2.set_ylabel('phase difference')
        ax2.set_xlabel('time')
        ax2.grid(True, which='both', linestyle='-', color='gray', alpha=0.3)
    if show:
        plt.show()

# Plots phase differences as a funcion of time as well as oscillator value plots for each connection topology
# Phase differences are plotted from 4 different starting conditions - 0, 0.25, 0.5, 0.75, where 1.0 is one cycle
# Starting conditions are offset slightly from exact value to avoid unstable steady states
def plot_phasediff_combs():
    T = get_period()
    print(f'period: {T}')
    combs = [(1, 1, 1, 1), (1, 2, 1, 1), (1, 2, 1, 2), (1, 2, 2, 1)]
    fig, ax = plt.subplots(len(combs), 3)
    start_configs = [10, T*0.25 + 10, T*0.5 + 10, T*0.75 + 10]
    for plt_idx in range(len(combs)):
        y2 = []
        for offset_idx in range(len(start_configs)):
            offset = start_configs[offset_idx]
            oscil1 = Oscillator()
            oscil2 = Oscillator()
            for i in range(int(400/Neuron.dt)):
                oscil1.tick()
                oscil2.tick()
            for i in range(int(offset/Neuron.dt)):
                oscil2.tick()
            oscil1.reset()
            oscil2.reset()
            oscil1.start_plot()
            oscil2.start_plot()
            oscil1.connect(oscil2, combs[plt_idx][0], combs[plt_idx][1], -0.003)
            oscil2.connect(oscil1, combs[plt_idx][2], combs[plt_idx][3], -0.003)
            for i in range(int(20000/Neuron.dt)):
                oscil1.tick()
                oscil2.tick()
            data1 = oscil1.end_plot()
            data2 = oscil2.end_plot()
            if offset_idx == 1:
                ax[plt_idx, 2].plot(data1[0], data1[1])
                ax[plt_idx, 2].plot(data2[0], data2[1])
            plot_phase_convergence(data1, data2, phase_plot=False, diff_plot=True, ax2=ax[plt_idx, 1], show=False)
            print(f'finished start config {offset_idx}')
        draw_arrangement(ax[plt_idx, 0], combs[plt_idx])
        print(f'~~~~~~~~~~comb {plt_idx} finished~~~~~~~~~~')

    plt.show()

# Numerically calculates the shift in phase of a neuron given a stimulation time, length, and strength
def get_phase_kick(stim_t, stim_len, stim_strength, const_stim=0.1, plot=True):
    if plot:
        fig, ax = plt.subplots(2, 1)
    n3 = Neuron()
    I = 0
    n3.start_lambda_stim(lambda: I)
    n3.start_plot()
    for i in range(int(stim_t*2/Neuron.dt)):
        if (n3.t > stim_t) and (n3.t < stim_t + stim_len):
            I = const_stim + stim_strength
        else:
            I = const_stim
        n3.tick()
    data = n3.end_plot()
    if plot:
        ax[0].plot(data[0], data[1])
        ax[0].plot([stim_t, stim_t], [-1, 1.5], 'b--')
        ax[0].plot([stim_t + stim_len, stim_t + stim_len], [-1, 1.5], 'g--')
        ax[0].set_xlim(data[0][0] - 100, data[0][-1] + 100)
    peaks = find_all_peaks(data[1])
    phase = []
    for i in range(len(peaks)):
        xpeak = peaks[i] * Neuron.dt
        phase.append(xpeak)
        if plot:
            ax[0].plot([xpeak, xpeak], [-1, 1.5], 'r--')
    y_vals = list(range(len(peaks)))
    idx = 0
    while phase[idx] < stim_t:
        idx += 1
    idx -= 1
    period = 0
    for i in range(2, idx):
        period += peaks[i+1] - peaks[i]
    period = (period / (idx - 2)) * Neuron.dt
    stim_phase = (stim_t - phase[idx]) / period
    c_before = np.polyfit(phase[:idx+1], y_vals[:idx+1], 1)
    phase_diff = (y_vals[idx + 3] - (c_before[0] * phase[idx + 3] + c_before[1])) % 1
    if phase_diff > 0.5:
        phase_diff -= 1.0
    if plot:
        ax[1].plot(phase, y_vals)
        ax[1].axline((0, c_before[1]), (1, c_before[1] + c_before[0]), color='g', linestyle='--')
        ax[1].set_xlim(data[0][0] - 100, data[0][-1] + 100)
        plt.show()
    
    return stim_phase, phase_diff

# Creates and plots a phase response curve (PRC) for one neuron
def plot_prc():
    p = []
    for time in np.arange(1000, 1175, 1):
        point = get_phase_kick(time, 60, 0.01, plot=False)
        p.append(point)
    p.sort()
    pert_locs = [i[0] for i in p]
    plt.plot(pert_locs, [i[1] for i in p])
    plt.xlabel('perturbation location')
    plt.ylabel('phase difference')
    plt.grid()
    plt.show()

    plt.plot(pert_locs, [i[1] for i in p])
    plt.plot(pert_locs, [(np.sin(i * 2 * np.pi)*-0.04) for i in pert_locs], 'r--') # sinusoidal H func used in Skinner model
    plt.xlabel('perturbation location')
    plt.ylabel('phase difference')
    plt.grid()
    plt.show()
    return p

# analytically calculates plots dφ/dt where φ is the difference in phases of two oscillators and ranges from 0 to 1
# Uses PRC as H function and finds dφ/dt using dθ2/dt - dθ1/dt, which can both be expressed using H(some func of θ2-θ1)
def plot_dphi_dt_combs(prc):
    pert_locs = [i[0] for i in prc]
    H = lambda phi: prc[np.searchsorted(pert_locs, np.mod(phi, 1))][1]
    x = np.arange(0, 1.005, 0.01)
    fig, ax = plt.subplots(4, 2)
    combs = [(1, 1, 1, 1), (1, 2, 1, 1), (1, 2, 1, 2), (1, 2, 2, 1)]
    for plt_idx in range(len(combs)):
        comb = combs[plt_idx]
        asc = 0.5 * (comb[1] - 1) - 0.5 * (comb[0] - 1)
        dsc = 0.5 * (comb[3] - 1) - 0.5 * (comb[2] - 1)
        y = []
        ma = -1e15
        mi = 1e15
        for i in x:
            val = H(dsc - i) - H(asc + i)
            y.append(val)
            if val > ma:
                ma = val
            if val < mi:
                mi = val
        pl = ax[plt_idx][1]
        pl.plot(x, y)
        pl.plot([x[0], x[-1]], [0, 0], 'r--')
        for xmark in [0, 0.25, 0.5, 0.75, 1]:
            pl.plot([xmark, xmark], [mi, ma], 'b--')
        if plt_idx == 0:
            pl.set_xlabel(f'phase difference (φ)')
            pl.set_ylabel(f'dφ/dt')
        draw_arrangement(ax[plt_idx][0], comb)
    plt.show()

# Plots two oscillators as a function of time with a random starting offset in phase connected with the config specified
def plot_oscillators_rand_start(config): 
    osc1 = Oscillator()
    osc2 = Oscillator()
    randt = random.random() * get_period()
    for i in range(int((400 + randt) / Neuron.dt)):
        osc2.tick()
    for i in range(int(400 / Neuron.dt)):
        osc1.tick()
    osc1.connect(osc2, config[0], config[1], -0.007)
    osc2.connect(osc1, config[2], config[3], -0.007)
    osc1.reset()
    osc2.reset()
    osc1.start_plot()
    osc2.start_plot()
    for i in range(int(10000 / Neuron.dt)):
        osc1.tick()
        osc2.tick()
    dat1 = np.array(osc1.end_plot())
    dat2 = np.array(osc2.end_plot())
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(dat1[0], dat1[1], label='osc1')
    ax[0].plot(dat2[0], dat2[1], label='osc2')
    ax[0].legend()
    ax[0].set_ylabel('oscillator states')
    plot_phase_convergence(dat1, dat2, ax1=ax[1], ax2=ax[2])

# Functions called here to show various plots and obtain results
if __name__ == '__main__':
    # testing match function
    dx = 0.001
    x = np.arange(0, 4*np.pi, dx)
    r = [int(0.2*np.pi/dx), int(0.6*np.pi/dx)]
    section = np.sin(x)[r[0]: r[1]]
    y = np.sin(x-0.47*np.pi)
    plt.plot(x, y)
    plt.plot(x[r[0]: r[1]], section)
    sec_start = match_function(section, y, r[0])
    new_r = [sec_start, len(section) + sec_start]
    plt.plot(x[new_r[0]: new_r[1]], section)
    plt.plot([x[r[0]], x[new_r[0]]], [section[0], section[0]], 'r--')
    plt.plot([x[r[1]], x[new_r[1]]], [section[-1], section[-1]], 'r--')
    plt.show()
     
    # plotting other stuff
    rand_stim_plot()
    plot_phasediff_combs()
    period = get_period()
    print(f'intrinsic period of an oscillator (I=0.1): {period}')
    kick_phase, kick = get_phase_kick((10 + random.random())*period, 0.25*period, 0.2, plot=True)
    print(f'shift in phase for a kick of strength 0.2 at {kick_phase*100}% of a cycle lasting for 25%: {kick*100}%')
    prc = plot_prc()
    plot_dphi_dt_combs(prc)
    plot_oscillators_rand_start((1, 1, 1, 1))
    plot_oscillators_rand_start((1, 2, 2, 1))
    plot_oscillators_rand_start((1, 2, 1, 1))




