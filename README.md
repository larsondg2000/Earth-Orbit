## Earth Orbit Simulation

<img src="orbit2.png" alt="Orbit of Earth around the Sun" width="75%">

### Problem Statement

* Create a simulation to track the orbit of the Earth around the Sun for a period of 1 year.
* Use Euler and Runge - Kutta method of 4th order (RK4) for this task.
* Find the distance from Earth to Sun at Apogee using Euler and RK4 method and compare it with the original.

   #### Given Equations

   * Accn of Earth due to Gravity of the Sun                    
       → $a = -\frac{GM}{|r|^3}\times\vec{r}$
   
   * ODE for Position                               
       → $\frac{dr}{dt} = v$ 
   
   * ODE for Velocity                  
      → $\frac{dv}{dt} = a$
   
   #### Initial Condition
   * Earth is at its Perihelion (closest to Sun)
   
   #### Sample Output

    <img src="orbit.png" alt="Orbit of Earth around the Sun" width="50%">
  
---

### Setup: Simulate any Planet's Orbit in our Solar System
**1.** Clone the Repository:- `https://github.com/larsondg2000/Earth-Orbit`             
**2.** Install `requirements.txt`:- pip install -r requirements.txt                 
**3.** <a href="#useage">Configure</a> the Simulation Parameters using `config.json`        
**4.** Run the `main.py` file                                                                   
**5.** Simply, change the Simulation Parameters to simulate different Planets in our Solar System      

---

### Understanding `config.json` for Simulation Parameters</h1>
Change the Simulation Parameters using [config.json](https://github.com/SpartificialUdemy/PSA/blob/main/M5%20-%20Earth's%20Orbit%20around%20the%20Sun/config.json):- 

   **a)** Planet Info:                                              
      → `name`: The name of the Planet to Display on the Plot.                      
      → `perihelion_color`: The color to give to the Planet at its Perihelion.                   
      → `aphelion_color`: The color to give to the Planet at its Aphelion.                  
                                 
   **b)** Initial Conditions:                                                                             
      → `position_at_perihelion`: The closest distance between the Sun and the Planet `(in million km)`.                                                                                       
      → `velocity_at_perihelion`: The value of speed at the Planet's Perihelion `(in km/s)`.                 
                          
   **c)** Time Settings:                                                                             
      → `time_step`: The steps in the Simulation for updating position and velocity `(in seconds)`.      
      → `simulation_time`: The maximum time of the simulation `(in days \rarr 1 day = 24 hours)`.      
                               
   **d)** Numerical Integration:                                                              
      → `method`: The method to choose for Numerical Integration `(either "RK4" or "Euler")`.      

---

### Simulate any planet of your choice
* Here is the [Planetary Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/
) that you can refer.