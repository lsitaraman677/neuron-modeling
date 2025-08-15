# Neuron Modeling for Neurolocomotion

This repository contains Python tools and simulations for studying **neurolocomotion** — how nervous systems generate rhythmic, coordinated motion. 
It was created as part of the COSMOS 2025 program at UC Davis, in Cluster 9: Mathematical Modeling of Biological Systems.
We use the **FitzHugh–Nagumo (FHN) equations** to model neurons and oscillators, and explore how their interactions can produce locomotor patterns such as the **0.25 phase shift** between crayfish swimmerets.

---

## File Overview

### **`classes.py`**
Implements the core simulation framework:
- **`Neuron`** – models a single neuron with two-variable FHN dynamics.  
  Key methods:  
  - `tick()` — advance the neuron’s state by one time step  
  - `connect(other, strength, activation)` — connect this neuron to another with specified strength and activation function  
  - `break_all_connections()` — remove all incoming connections  
  - `start_plot()` / `end_plot()` — begin and stop recording voltage data  
  - `reset()` — clear recorded data and time  
  - `start_lambda_stim(func)` / `end_stim()` — enable/disable custom external stimulation  
  - `plot_data()` — visualize recorded voltage over time  

- **`Oscillator`** – models a pair of reciprocally connected neurons as a simple oscillator.  
  Key methods:  
  - `tick()` — advance both neurons  
  - `connect(other, from_n, to_n, strength)` — connect one neuron in this oscillator to one in another  
  - `start_plot()` / `end_plot()` — record voltage difference between neurons  
  - `plot_data()` — plot voltage differences or raw voltages  
  - `reset()` — clear oscillator state and data  

---

### **`make_plots.py`**
Provides tools for simulating and analyzing oscillator networks.  
Includes functions for:
- Measuring oscillator periods  
- Plotting neuron responses and phase differences  
- Generating Phase Response Curves (PRCs)  
- Visualizing how network connectivity affects phase relationships  

When run directly, this file executes a variety of simulations and plots to showcase oscillator behavior.

---

### **`skinner_model.py`**
A minimal simulation of **four connected oscillators** using the **Skinner model**, a reduced form of the FHN coupling dynamics.  
- Demonstrates a stable state with **0.25 phase shifts** between oscillators — the same pattern found in crayfish swimmerets.  
- Uses cosine-based coupling functions to reproduce biological coordination.

---

### **`fhn_phaseplane.py`**
An **interactive** FHN phase-plane simulator:
- Displays nullclines, a vector field, and the trajectory of a single neuron’s `(v, w)` state over time.  
- Bottom subplot shows `v` (voltage) vs. time.  
- In real time, you can:
  - Enter a number to apply an instantaneous voltage bump to the neuron  
  - Enter Python expressions (e.g., `I=1.0`, `b=2`) to change model parameters (`a`, `b`, `eps`, `I`, etc.) and instantly update the vector field and nullclines  
- Great for building intuition about parameter effects on neuron dynamics.

---

## Acknowledgements
Special thanks to **Dr. Guy**, **Dr. Lewis**, **Dr. Schreiber**, and the
**COSMOS 2025** program for their guidance and support during this project.
