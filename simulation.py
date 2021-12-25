import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from quantumpython import *
from initial_parameters import *

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

# INTEGRAL OF THE MODULUS SQUARED OF THE WAVEFUNCTION
def integral_modulus_squared(mod_sq):
    print("Integral of the Modulus Squared of the Wave Function")
    integral_list = []
    for col in pbar3(range(N)):
        integral = 0
        for row in range(M):
            if row == 0 or row == M - 1:
                integral += 0.5 * mod_sq[row, col] * Dx
            else:
                integral += mod_sq[row, col] * Dx
        integral_list.append(integral)
    return integral_list

def export_integral(wf_modulus_squared):
    f = open("integral_of_the_modulus_squared.txt", 'w')
    f.writelines([ str(integral) + "\n" for integral in integral_modulus_squared(wf_modulus_squared)])
    f.close()

# PLOT ANIMATION
def animate(i):
    y = wf_modulus_squared[:,10*i]
    ax.clear()
    ax.axvspan(0, l, alpha=0.5, color='black')
    ax.plot(x, y, color='blue', alpha=0.5)
    ax.fill_between(x, y, alpha=0.5, color='blue')
    ax.set_xlim([-60,60])
    ax.set_ylim([0,0.5])

# MAIN FUNCTIONS
def main_from_scratch():
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

    # STORE THE MATRICES IN A FILE
    filename = "wavefunction.txt"
    print("\nExporting Wave Function to File", filename)
    export_mat(wf, filename=filename)
    filename = "wavefunction_modulus_squared.txt"
    print("Exporting Wave Function Modulus Squared to File", filename)
    export_mat(wf_modulus_squared, filename=filename)

    # INTEGRATE
    print("")
    export_integral(wf_modulus_squared)

    return wf_modulus_squared

def main_from_files():
    # IMPORT WAVEFUNCTION MATRIX FROM .TXT
    #wf = import_wavefunction_mat()

    # IMPORT MATRIX OF THE MODULUS SQUARED FROM .TXT
    wf_modulus_squared = import_wavefunction_modulus_mat()

    # INTEGRATE
    export_integral(wf_modulus_squared)

    return wf_modulus_squared



# MAIN FUNCTION
if __name__=="__main__":
    print("1. Calculate the wave function from scratch using the initial parameters from file initial_parameters.py")
    print("2. Plot the wave function from the file wavefunction.txt")
    choice = input("Enter your choice: ")
    if choice == "1":
        wf_modulus_squared = main_from_scratch()
    elif choice == "2":
        wf_modulus_squared = main_from_files()
    else:
        print("Invalid Choice")
        exit()

    # PLOT THE FIGURE
    x = [ x_min + k*Dx for k in range(M) ]
    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, animate, frames=int(N/10), interval=Dt/10, repeat=False)
    plt.show()