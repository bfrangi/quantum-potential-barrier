import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# CONSTANTS
e = np.exp(1) #Constant e
I = 1j #Complex unit

m_e = 9.1093837015*10**(-31) #Mass of the electron
epsilon_0 = 8.85*10**(-12) #Vacuum permitivity
h_bar = 1.054571*10**(-34) #Reduced Plank's constant
q_e = 1.6*10**(-19) #Elementary charge
a_0 = 4 * pi * epsilon_0 * h_bar**2 / ( m_e * q_e**2 ) #Bohr's Radius
E_h = h_bar**2/ ( m_e * a_0**2 )

N = 1000 #Number of x values
sigma_0 = 1 #Spatial width of the wavefunction
k_0 = 1
x_0 = -15 #Value around which the inital wave function is centered


# NORMALIZED VALUES OF THE PARAMETERS
x_min = -20 #Min value of x (Distance is normalized with: x = x_real/a_0)
x_max = 20 #Max value of x                 
Dx = (x_max - x_min) / float(N - 1) #Space interval
a = 2 #Width of the potential barrier (positive)
V_0 = 0.1 #Height of the potential barrier (Potential is normalized as V = V_real/E_h)
m = 1 #Mass of the particle (Mass is normalized with: m = m_real/m_e) ---> m = 1 is fixed!!

Dt = 0.05 #Time interval (Time is normalized with: t = t_real · E_h/h_bar)

#Note also that energy is normalized as E = E_real/E_h




# POTENTIAL FUNCTION
def V(x):
    if 0 < x < a:
        return V_0
    else:
        return 0

# MESH GERNERATION
def generate_mesh(N, Dt, x_min):
    mesh = []
    for k in range(1, N + 1):
        time = {
            't': (k-1)*Dt,
            'x': [],
            'phi': [],
            'modulus_squared_phi': []
        }
        for j in range(1, N + 1):
            x = x_min + (j - 1) * Dx
            time['x'].append(x)
        mesh.append(time)
    return mesh

def print_mesh(mesh):
    for i in mesh:
        print("t =", round(i['t'], 4), end='; x = [')
        for x_val in i['x']:
            print(round(x_val, 4), end=' ')
        print("]; Phi = [", end='')
        for phi_val in i['phi']:
            print(np.around(phi_val, 4), end=' ')
        print("]")
        #print(i['modulus_squared_phi'])

def insert_phi(mesh):
    for k in range(1, N + 1):
        for j in range(1, N + 1):
            mesh[k - 1]['phi'].append(10**len(str(j))*k + j)
        
# INITIAL WAVE FUNCTION
def phi_0(x):
    #return ( pi * sigma_0**2 )**( -1/4 ) * e**( I * k_0 * x * a_0) * e**( - (x - x_0)**2 * a_0**2 / (2*sigma_0**2) )
    return ( pi * sigma_0**2 )**( -1/4 ) * e**( I * k_0 * x) * e**( - (x - x_0)**2 / (2*sigma_0**2) )

def get_initial_phi(x_list):
    initial_phi_list = []
    for x in x_list:
        initial_phi_list.append(phi_0(x))#initial wave function formula here
    return initial_phi_list

# INTEGRATION METHOD
def next_phi(k, mesh):
    phi_k_list = mesh[k - 1]['phi']
    phi_k_plus_1_list = []
    for j in range(1, N + 1):
        phi_j_k = phi_k_list[j-1]
        if j < N:
            phi_j_plus_1_k = phi_k_list[j]
        else:
            phi_j_plus_1_k = 0
        if j > 1:
            phi_j_minus_1_k = phi_k_list[j-2]
        else:
            phi_j_minus_1_k = 0
        x = mesh[k - 1]['x'][j - 1]
        phi_k_plus_1 = phi_j_k + I * Dt/Dx**2 * (phi_j_plus_1_k - 2*phi_j_k + phi_j_minus_1_k) - 2*I*Dt*phi_j_k*V(x)
        phi_k_plus_1_list.append(phi_k_plus_1)
    return phi_k_plus_1_list

def modulus_squared(k, mesh):
    phi_k_list = mesh[k - 1]['phi']
    phi_k_moduli_list = []
    for phi_k in phi_k_list:
        phi_k_moduli_list.append(np.absolute(phi_k)**2)
    return phi_k_moduli_list

# PLOT ANIMATION
def animate(i):
    x = msh[i]['x']
    y = msh[i]['modulus_squared_phi']

    ax.clear()
    ax.plot(x, y)
    ax.set_xlim([-20,20])
    ax.set_ylim([0,2])

# MAIN FUNCTION
if __name__=="__main__":
    msh = generate_mesh(N, Dt, x_min)
    msh[0]['phi'] = get_initial_phi(msh[0]['x'])
    msh[0]['modulus_squared_phi'] = modulus_squared(1, msh)
    for k in range(1, N):
        msh[k]['phi'] = next_phi(k, msh)
        msh[k]['modulus_squared_phi'] = modulus_squared(k + 1, msh)

    x = []
    y = []
    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, animate, frames=100, interval=10, repeat=True)

    plt.show()


    
