import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
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

# INITIAL WAVE FUNCTION
initial_wf = lambda x: ( pi * sigma_0**2 )**( -1/4 ) * e**( I * k_0 * x) * e**( - (x - x_0)**2 / (2*sigma_0**2) )

# MODULUS SQUARED OF THE WAVE FUNCTION
def modulus_squared(mat):
    return np.absolute(mat)**2

# INTEGRAL OF THE MODULUS SQUARED OF THE WAVEFUNCTION
def integral_modulus_squared(mod_sq):
    print("Computing Integral of the Modulus Squared of the Wave Function...")
    integral_list = []
    for col in range(N):
        integral = ( np.sum(mod_sq[:,col]) - 0.5 * ( mod_sq[0, col] + mod_sq[M - 1, col] ) ) * Dx
        integral_list.append(integral)
    print(f"{bcolors.OKGREEN}done{bcolors.ENDC}")
    return integral_list

def export_integral(wf_modulus_squared):
    f = open("integral_of_the_modulus_squared.txt", 'w')
    integral_list = integral_modulus_squared(wf_modulus_squared)
    f.writelines([ str(integral) + "\n" for integral in integral_list])
    f.close()
    return integral_list

# PLOT ANIMATION
def animate(i):
    y = wf_modulus_squared[:,10*i]
    #update main axes, where the modulus squared of the wave function is plotted
    main_axes.clear()
    main_axes.axvspan(0, l, alpha=0.5, color='black')
    main_axes.plot(x, y, color='blue', alpha=0.5)
    main_axes.fill_between(x, y, alpha=0.5, color='blue')
    main_axes.set_xlim([-60,60])
    main_axes.set_ylim([0,1])
    main_axes.set_xlabel("Position")
    main_axes.set_ylabel("Squared Wavefunction Modulus")
    #update T_prob axes, where the transmission probability is plotted
    if integral_list:
        main_axes.axis["right"].set_visible(False)
        main_axes.axis["top"].set_visible(False)
        T_prob.clear()
        T_prob.axis["right"].set_visible(True)
        T_prob.axis["right"].major_ticklabels.set_visible(True)
        T_prob.axis["right"].label.set_visible(True)
        T_prob.axis["top"].set_visible(True)
        T_prob.axis["top"].major_ticklabels.set_visible(True)
        T_prob.axis["top"].label.set_visible(True)
        T_prob.set_xlabel("Time")
        T_prob.set_ylabel("Transmission Probability")
        T_prob.set_yscale("log")
        T_prob.set_ylim([1e-7, 2])
        T_prob.set_xlim([0, N*Dt])
        T_prob.plot(t[0:i*10], integral_list[0:i*10], color='black', alpha=0.5)

# MAIN FUNCTIONS
def main_from_scratch(export_and_integrate=True):
    # CREATE SPACE AND TIME VECTORS AND WAVE FUNCTION MATRIX
    x = np.linspace(x_min, x_min + Dx * (M - 1), M)
    t = np.linspace(0, Dt * (N - 1), N)
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

    # SOLVING TRIDIAGONAL MATRIX PROBLEM (L·X = S) <---------------------------------------------------- OPTIMIZE
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
        filename = "wavefunction.npy"
        print("\nExporting Wave Function to File", filename, "...")
        export_matrix(wf, filename)
        print(f"{bcolors.OKGREEN}done{bcolors.ENDC}")
        filename = "wavefunction_modulus_squared.npy"
        print("Exporting Wave Function Modulus Squared to File", filename, "...")
        export_matrix(wf_modulus_squared, filename)
        print(f"{bcolors.OKGREEN}done{bcolors.ENDC}")

        # INTEGRATE
        integral_list = export_integral(wf_modulus_squared)
        return wf_modulus_squared, integral_list
    else:
        return wf_modulus_squared

def main_from_files():
    # IMPORT WAVEFUNCTION MATRIX FROM .NPY
    # filename = "wavefunction.npy"
    # wf = import_matrix(filename)

    # IMPORT MATRIX OF THE MODULUS SQUARED FROM .NPY
    filename = "wavefunction_modulus_squared.npy"
    print("Importing Wave Function Modulus Squared from File", filename, "...")
    wf_modulus_squared = import_matrix(filename)
    print(f"{bcolors.OKGREEN}done{bcolors.ENDC}")

    # INTEGRATE
    integral_list = export_integral(wf_modulus_squared)

    return wf_modulus_squared, integral_list

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

    plt.savefig('Output-Media/wavefunction_for_given_times.pdf', bbox_inches='tight')
    f = os.path.dirname(os.path.realpath(__file__)) + "/Output-Media/wavefunction_for_given_times.pdf"
    print("Saved figure to", f)

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
            if position in [i, M - 1]: mult = 0.5
            else: mult = 1
            integral += mult * wf_prob[position, time] * Dx
        integral_list.append(integral)
    return integral_list

# PLOT TRANSMISSION COEFFICIENT AS A FUNCTION OF ENERGY
def plot_T_vs_E(E, T):
    fig, ax = plt.subplots()
    handles = ["Simulated Transmission Coefficient", "Theoretical Transmission Coefficient"]
    ax.plot(E, T, c='orange', alpha=0.5)
    DE = ( max(E) - min(E) ) / 10000
    E_list = [ i * DE for i in range(10000) if i > 0 ]
    T_list = [ T_theory(i) for i in E_list ]

    ax.plot(E_list, T_list, c='blue', alpha=0.5)
    ax.legend(handles)
    plt.xlabel("Energy (E0)".translate(SUB))
    plt.ylabel("Transmission Coefficient (T)")
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    plt.savefig('Output-Media/transmission_coefficient_vs_energy_sigma_' + re.sub(r'\.', r'\,', str(sigma_0)) + '.pdf', bbox_inches='tight')
    f = os.path.dirname(os.path.realpath(__file__)) + "/Output-Media/transmission_coefficient_vs_energy_sigma_" + re.sub(r'\.', r'\,', str(sigma_0)) + ".pdf"
    print("Saved figure to", f)

    plt.show()

# DETECT HORIZONTAL ASYMPTOTE APPROACHED FROM BELOW
def detect_asympote(values):
    asymptote = 0
    for value in values:
        asymptote = max(asymptote, value)
    return asymptote

# THEORETICAL TRANSMISSION COEFFICIENT AS A FUNCTION OF THE ENERGY
def T_theory(E):
    if E == V_0: return None
    elif E < V_0: return ( 1 + V_0**2 / ( 4 * E * ( V_0 - E ) ) * np.sinh( l * np.sqrt( 2 * ( V_0 - E ) ) )**2 )**(-1)
    else: return ( 1 + V_0**2 / ( 4 * E * ( E - V_0 ) ) * np.sin( l * np.sqrt( 2 * ( E - V_0 ) ) )**2 )**(-1)

# MAIN FUNCTION
if __name__=="__main__":
    N_custom = input(f"Enter the number of time points (press [enter] for default: N = {N}):{bcolors.OKGREEN} ")
    print(f"{bcolors.ENDC}", end="")
    if N_custom and not N_custom.isdigit():
        print(f"{bcolors.FAIL}Invalid Input{bcolors.ENDC}")
        exit()
    else:
        if N_custom:
            N = int(N_custom)
    Dt_custom = input(f"Enter the time step (press [enter] for default: Δt = {Dt}):{bcolors.OKGREEN} ")
    if Dt_custom:
        try:            
            Dt_custom = float(Dt_custom)
            Dt = Dt_custom
        except:
            print(f"{bcolors.FAIL}Invalid Input{bcolors.ENDC}")
            exit()  
    print(f"{bcolors.ENDC}")

    
    print(f"{bcolors.BOLD}1. Compute the wave function from scratch using the initial parameters from file initial_parameters.py (if you have chosen non-default N and Δt, those values will be used)")
    print("2. Import the wave function from the file wavefunction.txt")
    print(f"3. Skip {bcolors.WARNING}(by skipping this step, some of the functions of the program are made unavailable){bcolors.ENDC}")
    choice = input(f"Enter your choice:{bcolors.OKGREEN} ")
    print(f"{bcolors.ENDC}")
    skipped = False
    if choice == "1": wf_modulus_squared, integral_list = main_from_scratch()
    elif choice == "2": wf_modulus_squared, integral_list = main_from_files()
    elif choice == "3": skipped = True
    else:
        print(f"{bcolors.FAIL}Invalid Choice{bcolors.ENDC}")
        exit()

    if skipped:
        print("Choose an option:")
        print(f"{bcolors.WARNING}WARNING: You have skipped computing/importing the wave function, so options 1, 2, 3 and 4 are not available.{bcolors.ENDC}")
    else: 
        print(f"\n{bcolors.OKCYAN}Finished Computations.{bcolors.ENDC} Choose an option:")

    print(f"{bcolors.BOLD}1. Create animation of the evolution of the wave function in time")
    print("2. Plot the wave function for 0Δt, 500Δt, 1000Δt, 1500Δt and 2000Δt ")
    print("3. Analyse the accuracy of the method")
    print("4. Compute probability of finding the electron beyond the barrier as a function of time")
    print(f"5. Compute probability of finding the electron beyond the barrier for different values of", f"k0".translate(SUB) + f"{bcolors.ENDC}")
    choice = input(f"Enter choice:{bcolors.OKGREEN} ")
    print(f"{bcolors.ENDC}")
    if not skipped and choice == "1":
        # PLOT THE FIGURE
        plot_transmission = input("Do you want to plot the transmission probability along with the wavefunction? [Y/n] ")
        if plot_transmission.lower() == "y" or plot_transmission.lower() == "yes":
            integral_list = transmission_prob(wf_modulus_squared)
            plot_transmission = True
        else:
            integral_list = []
            plot_transmission = False

        x = [ x_min + k*Dx for k in range(M) ]
        t = [ k*Dt for k in range(N) ]
        fig = plt.figure()

        main_axes = fig.add_axes([0.15, 0.1, 0.65, 0.8], axes_class=HostAxes)
        T_prob = ParasiteAxes(main_axes)
        main_axes.parasites.append(T_prob)
        ani = FuncAnimation(fig, animate, frames=int(N/10), interval=Dt*k_0, repeat=False)#the interval is Dt*k_0 to avoid very fast and very slow animations when exporting
        
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        plt.show()


        choice = input("Do you want to save the animation as an .mp4? [Y/n] ")
        if choice.lower() == "y" or choice.lower() == "yes":
            if plot_transmission:
                figure_filename = "animation_with_transmission.mp4"
            else:
                figure_filename = "animation.mp4"   
            f = os.path.dirname(os.path.realpath(__file__)) + "/Output-Media/" + figure_filename
            print("Saving animation to", f)
            writervideo = FFMpegWriter(fps=60, bitrate=-1) 
            ani.save(f, writer=writervideo)
            print("Done")
    elif not skipped and choice == "2":
        if N >= 2000:
            plot_times(wf_modulus_squared, [0, 500, 1000, 1500, 2000])
        else:
            print(f"{bcolors.FAIL}Cannot make this computation: N (value: {N}) is smaller than 2000{bcolors.ENDC}")
    elif not skipped and choice == "3":
        print("The accuracy of this method will be analysed by computing the integral of |ψ|2 ".translate(SUP) + "and calculating its average and deviation from the theroetical value, which is 1.")
        avg = list_average(integral_list)
        sd = list_standard_deviation(integral_list, 1)
        sd = format_exp(sd)
        print("Average of ∫ |ψ|2 dx =".translate(SUP) + f"{bcolors.OKCYAN}", avg, f"{bcolors.ENDC}")
        print("Standard deviation of ∫ |ψ|2 dx with respect to one =".translate(SUP)+ f"{bcolors.OKCYAN}", sd, f"{bcolors.ENDC}")
    elif not skipped and choice == "4":
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

        plt.savefig('Output-Media/probability_of_transmission.pdf', bbox_inches='tight')
        f = os.path.dirname(os.path.realpath(__file__)) + "/Output-Media/probability_of_transmission.pdf"
        print("Saved figure to", f)

        plt.show()
    elif choice == "5":
        k_0_list = [0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8]
        print("Enter the values of k0".translate(SUB)  + " (separated by commas) for which you would like to calculate the transmission probability.")
        k_0_list_custom = input("Press [enter] for default, " + str(k_0_list) + f":{bcolors.OKGREEN} ")
        if k_0_list_custom:
            k_0_list_custom = re.findall(r'[0123456789\.]+(?![0123456789\.])', k_0_list_custom)
            k_0_list_custom_correct = []
            for k in k_0_list_custom:
                try:
                    k = float(k)
                    k_0_list_custom_correct.append(k)
                except:
                    pass
            if k_0_list_custom_correct:
                k_0_list = k_0_list_custom_correct
                print(f"{bcolors.ENDC}Using values: " + str(k_0_list) + "\n")
            else:
                print(f"{bcolors.ENDC}" + "Using default values of k0".translate(SUB) + "\n")
        else:
            print(f"{bcolors.ENDC}" + "Using default values of k0".translate(SUB) + "\n")


        
        #k_0_list = [ 0.2 * i for i in range(1,40) ]
        transmission_probability_functions = []
        for i in k_0_list:
            k_0 = i
            print(f"{bcolors.UNDERLINE}Making Calculations for:{bcolors.ENDC}{bcolors.OKBLUE}", "K0 =".translate(SUB), str(k_0) + f"{bcolors.ENDC}")
            wf_modulus_squared = main_from_scratch(export_and_integrate=False)
            transmission_probability_functions.append(transmission_prob(wf_modulus_squared))
            print("")

        t = [ num*Dt for num in range(N) ]
        fig, ax = plt.subplots()
        handlers = []
        #asymptotic_T = [ detect_asympote(integral_list) for integral_list in transmission_probability_functions ] 


        cycol = cycle('bgrcmk')
        for i in range(len(transmission_probability_functions)):
            col = next(cycol)
            ax.plot(t, transmission_probability_functions[i], c=col, alpha=0.5)
            #plt.axhline(y=asymptotic_T[i], color=col, linestyle='dashed')
            handlers.append("k0 = ".translate(SUB) + str(k_0_list[i]))
            #handlers.append("T = " + str(round( asymptotic_T[i] ,4)))         

        
        ax.legend(handlers)
        
        ax.set_xlim([0,Dt * N])
        ax.set_ylim([10e-6,None])
        plt.xlabel("Time")
        plt.ylabel("Transmission Probability")
        plt.yscale("log")

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        plt.savefig('Output-Media/probability_of_transmission_comparison_sigma_' + re.sub(r'\.', r'\,', str(sigma_0)) + '.pdf', bbox_inches='tight')
        f = os.path.dirname(os.path.realpath(__file__)) + "/Output-Media/probability_of_transmission_comparison_sigma_" + re.sub(r'\.', r'\,', str(sigma_0)) + ".pdf"
        print("Saved figure to", f)

        plt.show()
        choice = input("Do you want to study the dependence of the (asymptotic) transmission probability on the energy E0? [Y/n]".translate(SUB) + f" {bcolors.OKGREEN}")
        print(f"{bcolors.ENDC}")
        if choice.lower() in ["yes", "y"]:
            E_0_list = [ ( k**2 + 1 / ( 2 * sigma_0**2 ) ) / 2 for k in k_0_list ]
            asymptotic_T = [ detect_asympote(integral_list) for integral_list in transmission_probability_functions ] 
            plot_T_vs_E(E_0_list, asymptotic_T)
        elif choice.lower() not in ["no", "n"]:
            print(f"{bcolors.FAIL}Invalid Choice{bcolors.ENDC}")
    else:
        print(f"{bcolors.FAIL}Invalid Choice{bcolors.ENDC}")
