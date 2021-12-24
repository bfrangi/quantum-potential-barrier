import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from progressbar import AnimatedMarker, Bar, ETA, FileTransferSpeed, Percentage, ProgressBar, RotatingMarker
widgets = ['Computing:', Percentage(), ' ', AnimatedMarker(markers='-\|/'),' ', Bar('█'), ' ', ETA(), ' ', FileTransferSpeed()]
pbar1 = ProgressBar(widgets=widgets, maxval=10000000)
pbar2 = ProgressBar(widgets=widgets, maxval=10000000)


# CONSTANTS
e = np.exp(1) #Constant e
I = 1j #Complex unit

m_e = 9.1093837015*10**(-31) #Mass of the electron
epsilon_0 = 8.85*10**(-12) #Vacuum permitivity
h_bar = 1.054571*10**(-34) #Reduced Plank's constant
q_e = 1.6*10**(-19) #Elementary charge
a_0 = 4 * pi * epsilon_0 * h_bar**2 / ( m_e * q_e**2 ) #Bohr's Radius
E_h = h_bar**2/ ( m_e * a_0**2 )

M = 2401 #Number of x values
sigma_0 = 3 #Spatial width of the wavefunction
k_0 = 1
x_0 = -10 #Value around which the inital wave function is centered


# NORMALIZED VALUES OF THE PARAMETERS
x_min = -60 #Min value of x (Distance is normalized with: x = x_real/a_0)
x_max = 60 #Max value of x                 
Dx = (x_max - x_min) / float(M - 1) #Space interval
l = 2 #Width of the potential barrier (positive)
V_0 = 2 #Height of the potential barrier (Potential is normalized as V = V_real/E_h)
m = 1 #Mass of the particle (Mass is normalized with: m = m_real/m_e) ---> m = 1 is fixed!!

N = 1000 #Total values of time (Time is normalized with: t = t_real · E_h/h_bar)
Dt = 0.1 #Time interval 
t = N * Dt #Total time 

#Note also that energy is normalized as E = E_real/E_h


# POTENTIAL FUNCTION
def V(x):
    if 0 < x < l:
        return V_0
    else:
        return 0

# SPACE AND TIME VECTORS
def generate_x_and_t(M, Dx, x_min, N, Dt):
    t = []
    x = []
    for k in range(N):
        t.append( k*Dt )
    for k in range(M):
        x.append( x_min + k*Dx )
    return np.array(x), np.array(t)

# INITIAL WAVE FUNCTION

def initial_wf(x_vector):
    wf_0 = np.zeros(len(x_vector), dtype='complex')
    for i in range(len(x_vector)):
        wf_0[i] = phi_0(x_vector[i])
    return wf_0

def phi_0(x):
    return ( pi * sigma_0**2 )**( -1/4 ) * e**( I * k_0 * x) * e**( - (x - x_0)**2 / (2*sigma_0**2) )

# MODULUS SQUARED OF THE WAVE FUNCTION

def modulus_squared(mat):
    print("Modulus Squared of the Wave Function")
    mod_sq = np.zeros([M, N])
    for row in pbar2(range(M)):
        for col in range(N):
            mod_sq[row, col] = np.absolute(mat[row, col])**2
    return mod_sq

# PLOT ANIMATION
def animate(i):
    y = wf_modulus_squared[:,i]
    ax.clear()
    ax.axvspan(0, l, alpha=0.5, color='black')
    ax.plot(x, y, color='blue', alpha=0.5)
    ax.fill_between(x, y, alpha=0.5, color='blue')
    ax.set_xlim([-60,60])
    ax.set_ylim([0,0.5])

# MAIN FUNCTION
if __name__=="__main__":
    # CREATE SPACE AND TIME VECTORS AND WAVE FUNCTION MATRIX
    x, t = generate_x_and_t(M, Dx, x_min, N, Dt)#create the space and time vectors
    wf = np.zeros([M, N], dtype='complex')# Create a matrix to store the wave function accross space (rows) and time (columns)
    wf[:,0] = initial_wf(x)# Set the initial wave function

    # FIND MATRICES L AND R
    alpha = Dt / ( 4 * Dx**2 )
    beta = 1 +  2 * I * alpha

    L = np.zeros([M, M], dtype='complex')# Create a matrix to store the values of L
    for i in range(M):# diagonal elements
        L[i, i] = beta + I * Dt * V( x[i] ) / 2
    for i in range(1, M):# other elements
        L[i - 1, i] = - I * alpha
        L[i, i - 1] = - I * alpha

    R = np.conjugate(L)

    # SOLVING TRIDIAGONAL MATRIX PROBLEM (L·X = S)
    print("Matrix of the Wave Function")
    a = L[0,1]
    d = lambda i: L[i,i]
    s = lambda i: S[i]

    a_prime = np.zeros([M,1], dtype='complex')
    for i in range(M - 1):
        if i == 0:
            a_prime[0] = a / d(0)
        else:
            a_prime[i] = a / ( d(i) - a * a_prime[i - 1] )

    S = np.matmul(R, wf[:,0])
    s_prime = np.zeros([M,1], dtype='complex')
    for i in range(M):
        if i == 0:
            s_prime[0] = s(0) / d(0)
        else:
            s_prime[i] = ( s(i) - a * s_prime[i - 1] ) / ( d(i) - a * a_prime[i - 1] )

    s_prime = np.zeros([M,1], dtype='complex')
    for t in pbar1(range(N - 1)):# iterate over time
        S = np.matmul(R, wf[:,t])
        for i in range(M):
            if i == 0:
                s_prime[0] = s(0) / d(0)
            else:
                s_prime[i] = ( s(i) - a * s_prime[i - 1] ) / ( d(i) - a * a_prime[i - 1] )
        for i in range(M - 1, -1, -1):
            if i == M - 1:
                wf[M - 1, t + 1] = s_prime[M - 1]
            else:
                wf[i, t + 1] = s_prime[i] - a_prime[i] * wf[i + 1, t + 1]

    # CALCULATE THE MODULUS
    wf_modulus_squared = modulus_squared(wf)

    # PLOT THE FIGURE
    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, animate, frames=N, interval=1, repeat=False)

    plt.show()

    
