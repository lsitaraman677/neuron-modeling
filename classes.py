import numpy as np
from matplotlib import pyplot as plt

# Represents a single neuron with dynamics based on a two-variable differential equation model
class Neuron:

    dt = 0.1  # Time step
    vthresh = 0.3  # Voltage threshold for activation function (determining whether neuron is on or off)

    def __init__(self, alpha=0, beta=1.0, gamma=0.0, epsilon=0.005):
        # State variables
        self.v = 0  # Membrane potential
        self.w = 0  # Recovery variable
        self.t = 0  # Time
        # Model parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = epsilon
        # External stimulation
        self.func = None
        self.func_active = False
        # Plotting
        self.vals = [[], []]
        self.plot = False
        # List of incoming connection functions
        self.connections = []

    def set_v(self, v):
        # Manually set the neuron's voltage
        self.v = v

    def tick(self):
        # Advance the neuron's state by one time step using the model equations
        funcTerm = 0
        if self.func_active and (not (self.func is None)):
            funcTerm = self.func()
        connectTerm = sum(i() for i in self.connections)  # Sum of inputs from connected neurons
        # Update dynamics
        self.v += (-self.v * (self.v-self.alpha) * (self.v-1) - self.w + funcTerm + connectTerm) * self.dt
        self.w += self.eps * (self.v - self.beta*self.w - self.gamma) * self.dt
        self.t += self.dt
        # Log values if plotting
        if self.plot:
            self.vals[0].append(self.t)
            self.vals[1].append(self.v)

    def connect(self, other, strength, activation=lambda v, vthresh: (0 if (v < vthresh) else 1)):
        # Connect this neuron to another neuron with a specified strength and activation function
        connect_func = lambda: activation(self.v, self.vthresh) * strength
        other.recieve_connection(connect_func)

    def recieve_connection(self, connect_func):
        # Accept a connection from another neuron
        self.connections.append(connect_func)

    def break_all_connections(self):
        # Remove all incoming connections
        self.connections = []

    def start_plot(self):
        # Enable voltage logging for plotting
        self.plot = True

    def end_plot(self):
        # Stop logging and return collected voltage data
        self.plot = False
        return self.vals

    def reset(self):
        # Reset time and recorded data (does not reset v/w)
        self.t = 0
        self.vals = [[], []]

    def stimulate(self, v):
        # Add a voltage input directly
        self.v += v

    def start_lambda_stim(self, func=None):
        # Begin external stimulation using a lambda function
        self.func_active = True
        if not (func is None):
            self.func = func

    def end_stim(self):
        # Stop external stimulation
        self.func_active = False

    def plot_data(self, show=True):
        # Plot the recorded voltage over time
        plt.plot(self.vals[0], self.vals[1])
        if show:
            plt.show()


# Represents a pair of reciprocally connected neurons forming a simple oscillator
class Oscillator:

    sigmoid = staticmethod(lambda x, xt: 1 / (1 + np.exp(-(x-xt)*10)))  # Smooth activation function

    def __init__(self, n1start=0.2, n2start=0, const_stim=0.1):
        # Create two neurons
        self.n1 = Neuron()
        self.n2 = Neuron()
        # Apply constant stimulation to both neurons
        stim_lambda = lambda: const_stim
        self.n1.start_lambda_stim(stim_lambda)
        self.n2.start_lambda_stim(stim_lambda)
        # Set initial voltages
        self.n1.set_v(n1start)
        self.n2.set_v(n2start)
        # Connect neurons with inhibitory connections
        self.n1.connect(self.n2, -0.02, self.sigmoid)
        self.n2.connect(self.n1, -0.02, self.sigmoid)
        # For plotting difference between neurons
        self.data = [[], []]
        self.save = False

    def tick(self):
        # Advance both neurons by one time step
        self.n1.tick()
        self.n2.tick()
        # Record voltage difference if enabled
        if self.save:
            self.data[0].append(self.n1.t)
            self.data[1].append(self.n1.v - self.n2.v)

    def set_v(self, v, n2_set=False):
        # Set voltage for one of the neurons
        if not n2_set:
            self.n1.set_v(v)
        else:
            self.n2.set_v(v)

    def start_plot(self):
        # Begin recording data for plotting
        self.save = True
        self.n1.start_plot()
        self.n2.start_plot()

    def end_plot(self):
        # Stop recording and return voltage difference data
        self.save = False
        self.n1.end_plot()
        self.n2.end_plot()
        return self.data

    def reset(self):
        # Reset internal state and plot data
        self.n1.reset()
        self.n2.reset()
        self.data = [[], []]

    def plot_data(self, neuron_plot=False, diff_plot=True, show=True):
        # Plot difference between neurons and/or individual neuron voltages
        if diff_plot:
            plt.plot(self.data[0], self.data[1])
        if neuron_plot:
            self.n1.plot_data(show=False)
            self.n2.plot_data(show=False)
        if show:
            plt.show()

    def connect(self, other, from_n, to_n, strength):
        # Connect one neuron from this oscillator to one in another oscillator
        if from_n == 1:
            if to_n == 1:
                self.n1.connect(other.n1, strength, self.sigmoid)
            elif to_n == 2:
                self.n1.connect(other.n2, strength, self.sigmoid)
            else:
                raise Exception('Invalid connection option specified, use either 1 or 2')
        elif from_n == 2:
            if to_n == 1:
                self.n2.connect(other.n1, strength, self.sigmoid)
            elif to_n == 2:
                self.n2.connect(other.n2, strength, self.sigmoid)
            else:
                raise Exception('Invalid connection option specified, use either 1 or 2')
        else:
            raise Exception('Invalid connection option specified, use either 1 or 2')

