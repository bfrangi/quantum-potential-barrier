# NORMALIZED VALUES OF THE PARAMETERS
sigma_0 = 7                             #Spatial width of the wavefunction
k_0 = 1                                 #Constant that is proportional to the velocity of the packet
x_0 = -10                               #Value around which the initial wave function is centered

x_min = -60                             #Min value of x (Distance is normalized with: x = x_real/a_0)
x_max = 60                              #Max value of x                 
Dx = 0.05                               #Space interval
M = int(1 + (x_max - x_min) / Dx)       #Number of x values
l =  2                                  #Width of the potential barrier (positive)
V_0 = 2                                 #Height of the potential barrier (Potential is normalized as V = V_real/E_h)
m = 1                                   #Mass of the particle (Mass is normalized with: m = m_real/m_e) ---> m = 1 is fixed!!

N = 2000                                #Total values of time (Time is normalized with: t = t_real · E_h/h_bar)
Dt = 0.005                              #Time interval 
t_tot = (N - 1) * Dt                    #Total time 

#Note also that energy is normalized as E = E_real/E_h

# For the last exercise, calculate transmission probability for:
k_0_list = [0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8]
