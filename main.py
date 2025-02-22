from utils import setup_simulation, read_json_config, numerical_integration, at_aphelion, plot_simulated_data

# Constants
G = 6.6743e-11  # gravitational constant
M_SUN = 1.989e30  # mass of sun (kg)

def orbit_sim():
    # Read config.json
    config = read_json_config("config.json")

    # Setup simulation
    planet_name, color_at_perihelion, color_at_aphelion, r, v, t, time_step, method = setup_simulation(config)

    # Call numerical integration
    numerical_integration(G, M_SUN, r, v, time_step, method)

    # Get data of Earth at its Aphelion
    arg_aphelion, vel_aphelion, pos_aphelion = at_aphelion(r, v)

    # Plot the simulated data
    plot_simulated_data(r, method, arg_aphelion, vel_aphelion, pos_aphelion,
                        planet_name, color_at_perihelion, color_at_aphelion)


if __name__ == "__main__":
    orbit_sim()