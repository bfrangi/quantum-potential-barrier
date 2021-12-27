import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from quantumpython import *
from initial_parameters import *
import os
from itertools import cycle


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
    progressbar = ProgressBar(widgets=widgets, maxval=10000000)
    for row in progressbar(range(M)):
        for col in range(N):
            mod_sq[row, col] = np.absolute(mat[row, col])**2
    return mod_sq

# INTEGRAL OF THE MODULUS SQUARED OF THE WAVEFUNCTION
def integral_modulus_squared(mod_sq):
    print("Integral of the Modulus Squared of the Wave Function")
    integral_list = []
    progressbar = ProgressBar(widgets=widgets, maxval=10000000)
    for col in progressbar(range(N)):
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
    plt.xlabel("Position")
    plt.ylabel("Squared Wavefunction Modulus")

# MAIN FUNCTIONS
def main_from_scratch(export_and_integrate=True):
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
    progressbar = ProgressBar(widgets=widgets, maxval=10000000)
    for t in progressbar(range(N - 1)):# iterate over time
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
    if export_and_integrate:    
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

# PLOT ONLY CERTAIN TIMES TOGETHER IN ONE FIGURE
def plot_times(wf_mod_sq, times):
    x = [ x_min + k*Dx for k in range(M) ]
    fig, ax = plt.subplots()
    handles = []

    cycol = cycle('bgrcmk')
    for t in times:
        y = wf_mod_sq[:,t]
        ax.plot(x, y, c=next(cycol), alpha=0.5)
        handles.append(str(t) + "Δt")

    ax.axvspan(0, l, alpha=0.5, color='black')
    ax.axvspan(0, l, alpha=0.5, color='black')
    ax.set_xlim([-25,10])
    ax.set_ylim([0,0.5])
    ax.legend(handles)
    plt.xlabel("Position")
    plt.ylabel("Squared Wavefunction Modulus")

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    plt.savefig('figure.pdf', bbox_inches='tight')
    f = os.path.dirname(os.path.realpath(__file__)) + "/figure.pdf"
    print("Saved animation to", f)

    plt.show()

# CALCULATE PROBABILITY OF TRANSMISSION
def transmission_prob(wf_prob):
    print("Probability of Transmission")
    integral_list = []
    i = int((l-x_min)/Dx)
    progressbar = ProgressBar(widgets=widgets, maxval=10000000)
    for time in progressbar(range(N)):
        integral = 0
        for position in range(i, M):
            if position == i or position == M - 1:
                integral += 0.5 * wf_prob[position, time] * Dx
            else:
                integral += wf_prob[position, time] * Dx
        integral_list.append(integral)
    return integral_list


# MAIN FUNCTION
if __name__=="__main__":
    print(f"{bcolors.BOLD}1. Compute the wave function from scratch using the initial parameters from file initial_parameters.py")
    print("2. Import the wave function from the file wavefunction.txt")
    print(f"3. Skip {bcolors.WARNING}(by skipping this step, some of the functions of the program are made unavailable){bcolors.ENDC}")
    choice = input(f"Enter your choice:{bcolors.OKGREEN} ")
    print(f"{bcolors.ENDC}")
    skipped = False
    if choice == "1":
        wf_modulus_squared = main_from_scratch()
    elif choice == "2":
        wf_modulus_squared = main_from_files()
    elif choice == "3":
        skipped = True
    else:
        print(f"{bcolors.FAIL}Invalid Choice{bcolors.ENDC}")
        exit()

    if skipped:
        print("Choose an option:")
        print(f"{bcolors.WARNING}WARNING: You have skipped computing/importing the wave function, so options 1, 2, and 3 are not available.{bcolors.ENDC}")
    else:
        print(f"\n{bcolors.OKCYAN}Finished Computations.{bcolors.ENDC} Choose an option:")

    print(f"{bcolors.BOLD}1. Create animation of the evolution of the wave function in time")
    print("2. Plot the wave function for 0Δt, 500Δt, 1000Δt, 1500Δt and 2000Δt ")
    print("3. Compute probability of finding the electron beyond the barrier as a function of time")
    print(f"4. Compute probability of finding the electron beyond the barrier for different values of", f"k0".translate(SUB) + f"{bcolors.ENDC}")
    choice = input(f"Enter choice:{bcolors.OKGREEN} ")
    print(f"{bcolors.ENDC}")
    if not skipped and choice == "1":
        # PLOT THE FIGURE
        x = [ x_min + k*Dx for k in range(M) ]
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, animate, frames=int(N/10), interval=Dt/10, repeat=False)
        
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        plt.show()

        choice = input("Do you want to save the animation as an .mp4? [Y/n] ")
        if choice.lower() == "y" or choice.lower() == "yes":       
            f = os.path.dirname(os.path.realpath(__file__)) + "/animation.mp4"
            print("Saving animation to", f)
            writervideo = FFMpegWriter(fps=60, bitrate=-1) 
            ani.save(f, writer=writervideo)
            print("Done")
    elif not skipped and choice == "2":
        plot_times(wf_modulus_squared, [0, 500, 1000, 1500, 2000])
    elif not skipped and choice == "3":
        integral_list = transmission_prob(wf_modulus_squared)

        t = [ k*Dt for k in range(N) ]
        fig, ax = plt.subplots()
        plt.axhline(y=integral_list[-1], color='black', linestyle='dashed', label=f"T = {integral_list[-1]}")
        ax.plot(t, integral_list, c='blue', alpha=0.5)
        ax.legend([f"T = {round(integral_list[-1], 4)}"])
        ax.set_xlim([0,Dt * N])
        ax.set_ylim([0,None])
        plt.xlabel("Time")
        plt.ylabel("Transmission Probability")

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        plt.savefig('probability_of_transmission.pdf', bbox_inches='tight')
        f = os.path.dirname(os.path.realpath(__file__)) + "/probability_of_transmission.pdf"
        print("Saved figure to", f)

        plt.show()
    elif choice == "4":
        k_0_list = [0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8]
        transmission_probability_functions = []
        for i in k_0_list:
            k_0 = i
            print(f"{bcolors.UNDERLINE}Making Calculations for:{bcolors.ENDC}{bcolors.OKBLUE}", "K0 =".translate(SUB), str(k_0) + f"{bcolors.ENDC}")
            wf_modulus_squared = main_from_scratch(export_and_integrate=False)
            transmission_probability_functions.append(transmission_prob(wf_modulus_squared))
            print("")

        t = [ k*Dt for k in range(N) ]
        fig, ax = plt.subplots()
        handlers = []

        cycol = cycle('bgrcmk')
        for i in range(len(transmission_probability_functions)):
            ax.plot(t, transmission_probability_functions[i], c=next(cycol), alpha=0.5)
            handlers.append("k0 = ".translate(SUB) + str(k_0_list[i]))            

        
        ax.legend(handlers)
        
        ax.set_xlim([0,Dt * N])
        ax.set_ylim([10e-6,None])
        plt.xlabel("Time")
        plt.ylabel("Transmission Probability")
        plt.yscale("log")

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        plt.savefig('probability_of_transmission_comparison.pdf', bbox_inches='tight')
        f = os.path.dirname(os.path.realpath(__file__)) + "/probability_of_transmission_comparison.pdf"
        print("Saved figure to", f)

        plt.show()
        choice = input("Do you want to study the dependence of the (asymptotic) transmission probability on the energy E0 choice? [Y/n]".translate(SUB) + f" {bcolors.OKGREEN}")
        print(f"{bcolors.ENDC}")
        if choice.lower() in ["yes", "y"]:
            pass
        else:
            print(f"{bcolors.FAIL}Invalid Choice{bcolors.ENDC}")
    else:
        print(f"{bcolors.FAIL}Invalid Choice{bcolors.ENDC}")
