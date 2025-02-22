import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from typing import cast
import json


# Simulation setup
def setup_simulation(config):
    # Access configuration values
    planet_name = config['planet_info']['name']
    color_at_perihelion = config['planet_info']['perihelion_color']
    color_at_aphelion = config['planet_info']['aphelion_color']
    initial_position = np.array(config['initial_conditions']['position_at_perihelion']) * 1e9  # m
    initial_velocity = np.array(config['initial_conditions']['velocity_at_perihelion']) * 1e3  # m/s
    time_step = config['time_settings']['time_step']  # sec
    max_time = config['time_settings']['simulation_time'] * 24 * 3600  # sec
    method_integration = config['numerical_integration']['method']

    # Time array for numerical solution
    t = np.arange(0, max_time, time_step)

    # Initialize arrays to store positions and velocities at all time steps
    r = np.empty(shape=(len(t), 2))
    v = np.empty(shape=(len(t), 2))

    # Set initial conditions for position and velocity
    r[0], v[0] = np.array([initial_position, 0]), np.array([0, -initial_velocity])

    return planet_name, color_at_perihelion, color_at_aphelion, r, v, t, time_step, method_integration


def read_json_config(file_path):
    """
    Read a JSON configuration file and return the configuration data.

    Parameters:
    - file_path (str): The path to the JSON configuration file.

    Returns:
    - dict: A dictionary containing the configuration data.
    """
    # Load the JSON file for Configuration
    with open(file_path, 'r') as file:
        config_data = json.load(file)
    return config_data


def numerical_integration(g, m_sun, r, v, dt, method):
    """
    This will perform numerical integration based on the method
    :param g:  Universal Gravitational Constant.
    :param m_sun: Mass of the Sun in kilograms.
    :param r: Array representing the position vector at each time step.
    :param v: Array representing the velocity vector at each time step.
    :param dt: Time step for the simulation.
    :param method: Integration method, either "euler" or "rk4".
    :return:
    """
    if method.lower() == 'euler':
        euler_method(g, m_sun, r, v, dt)
    elif method.lower() == 'rk4':
        rk4_method(g, m_sun, r, v, dt)
    else:
        raise Exception(f"You can choose Euler or RK4.  Your current input method is: {method}")


# Define the Acceleration function
def accel(g, m_sun, r):
    return (-g * m_sun / np.linalg.norm(r) ** 3) * r


# Euler Integration
def euler_method(g, m_sun, r, v, dt):
    """
    Equations for Euler Method
    ODE for Position
    --> dr/dt = v
    --> r_new = r_old + v_old*dt

    ODE for Velocity
    --> dv/dt = a
    --> v_new = v_old + a(r_old)*dt

    :param g:
    :param m_sun:
    :param r: empty array for position of size t
    :param v: empty array for velocity of size t
    :param dt: time step
    :return:
    """
    for i in range(1, len(r)):
        r[i] = r[i - 1] + v[i - 1] * dt
        v[i] = v[i - 1] + accel(g, m_sun, r[i - 1]) * dt


# RK4 Method
def rk4_method(g, m_sun, r, v, dt):
    """
     Equations for RK4
     ODE for Position
     --> dr/dt = v
     --> r_new = r_old + dt/6(k1r + 2*k2r + 3*k3r + k4r)

     ODE for Velocity
     --> dv/dt = a
     --> v_new = v_old + dt/6(k1v + 2*k2v + 3*k3v + k4v)

     Methods to calculate steps:
     Step 1: @0
     k1v = accel(r[i-1])
     k1r = v[i-1]

     Steps 2: dt/2 using k1
     k2v = accel(r[i-1] + k1r * dt/2)
     k2r = v[i-1] + k1v * dt/2

     Step 3: dt/2 using k2
     k3v = accel(r[i-1] + k2r * dt/2)
     k3r = v[i-1] + k2v * dt/2

     Step 4: dt using k3
     k4v = accel(r[i-1] + k3r * dt)
     k4r = v[i-1] + k3v * dt

     :param g:
     :param m_sun:
     :param r: empty array for position of size t
     :param v: empty array for velocity of size t
     :param dt: time step
     :return:
     """
    for i in range(1, len(r)):
        # Step 1: @0
        k1v = accel(g, m_sun, r[i - 1])
        k1r = v[i - 1]

        # Steps 2: dt / 2 using k1
        k2v = accel(g, m_sun, r[i - 1] + k1r * dt / 2)
        k2r = v[i - 1] + k1v * dt / 2

        # Step 3: dt / 2 using k2
        k3v = accel(g, m_sun, r[i - 1] + k2r * dt / 2)
        k3r = v[i - 1] + k2v * dt / 2

        #Step 4: dt using k3
        k4v = accel(g, m_sun, r[i - 1] + k3r * dt)
        k4r = v[i - 1] + k3v * dt

        # Update r and v
        r[i] = r[i - 1] + dt / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
        v[i] = v[i - 1] + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)


def at_aphelion(r, v):
    sizes = np.array([np.linalg.norm(position) for position in r])
    pos_aphelion = np.max(sizes)
    arg_aphelion = np.argmax(sizes)
    vel_aphelion = np.linalg.norm(v[arg_aphelion])
    return arg_aphelion, vel_aphelion, pos_aphelion

def plot_simulated_data(r, method, arg_aphelion,
                        vel_aphelion, pos_aphelion, name_planet, color_peri, color_ap):
    # Setup figure
    plt.style.use("dark_background")
    plt.figure(figsize=(7, 12))
    plt.subplot(projection="3d")

    # Add subtitle
    sub_str = "RK4" if method.lower() == "rk4" else "Euler"
    plt.suptitle(f"{sub_str} Method", fontsize="18", color="r", weight="bold")

    # Add title
    title_str = f'At Aphelion, the {name_planet} is {round(pos_aphelion / 1e9, 1)} million km away from the Sun\nMoving at the speed of {round(vel_aphelion / 1e3, 1)} km/s.'
    plt.title(title_str, fontsize=14, color='orange')

    # Plot the Orbit, Sun, Earth at Perihelion and Aphelion
    plt.plot(r[:, 0], r[:, 1], color='tab:pink', lw=2, label='Orbit')
    plt.scatter(0, 0, color='yellow', s=1000, label='Sun')
    plt.scatter(r[0, 0], r[0, 1], s=200, label=f'{name_planet} at its Perihelion', color=color_peri)
    plt.scatter(r[arg_aphelion, 0], r[arg_aphelion, 1], s=200,
                label=f'{name_planet} at its Aphelion', color=color_ap)

    # Add legend and use type casting to modify legend size
    legend = plt.legend(loc="lower right", frameon=False)
    for i, size in [(1, 150), (2, 80), (3, 80)]:
        handle = cast(PathCollection, legend.legend_handles[i])
        handle.set_sizes([size])

    # Turn off axis and display
    plt.axis("off")
    plt.show()
