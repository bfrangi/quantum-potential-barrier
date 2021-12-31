from initial_parameters import *
from quantumpython import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings("error")

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, Column
from rich.color import Color
console = Console(log_path=False)

warn_number = 0


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
def modulus_squared(mat, warn_number):
    text_column = TextColumn("{task.description}", table_column=Column(width=35))
    bar_column = BarColumn(bar_width=None, table_column=Column(ratio=1), style="bar.back", complete_style="default")
    data_column = TextColumn("[right]{task.completed} of {task.total}", table_column=Column(width=11))

    with Progress(text_column, bar_column, data_column, console=console, auto_refresh=False, expand=True, transient=True) as progress:
        task1 = progress.add_task('Modulus Squared of the Wave Function ', total=M)
        
        mod_sq = np.zeros([M, N])
        for row in range(M):
            overflow = False
            for col in range(N):
                modulus = 0
                try:
                    modulus = np.absolute(mat[row, col])**2
                except RuntimeWarning:
                    overflow = True
                finally:
                    mod_sq[row, col] = modulus
            if overflow:
                print('\x1b[2k\x1b[100D\x1b[1A')
                console.print(f"[red]Warning {warn_number}: Encountered large number!! Your simulation is exploding, please take cover =).")
                #print(f"{bcolors.WARNING}Warning {warn_number}: Encountered large numbers!! Your simulation is exploding, please take cover =).{bcolors.ENDC}")
                warn_number += 1
            progress.update(task1, advance=1, refresh=True)

    return mod_sq


# MAIN FUNCTIONS
def main(x, t, warn_number):
    # CREATE WAVE FUNCTION MATRIX
    wf = np.zeros([M, N], dtype='complex')# Create a matrix to store the wave function accross space (rows) and time (columns)
    wf[:,0] = initial_wf(x)# Set the initial wave function

    # FILL WAVE FUNCTION MATRIX
    text_column = TextColumn("{task.description}", table_column=Column(width=22))
    bar_column = BarColumn(bar_width=None, table_column=Column(ratio=1), style="bar.back", complete_style="default")
    data_column = TextColumn("[right]{task.completed} of {task.total}", table_column=Column(width=11))

    with Progress(text_column, bar_column, data_column, console=console, auto_refresh=False, expand=True, transient=True) as progress:
        task1 = progress.add_task('Computing Wave Function ', total=(N - 1))
        for t in range(1, N):                   #iterate over time
            prev_wf = wf[:,t - 1]
            next_wf = []
            overflow = False
            for j in range(M):                  #iterate over the position
                phi_k_plus_1 = 0
                try:
                    phi_j_k = prev_wf[j]
                    if j < N:
                        phi_j_plus_1_k = prev_wf[j + 1]
                    else:
                        phi_j_plus_1_k = 0
                    if j > 1:
                        phi_j_minus_1_k = prev_wf[j - 1]
                    else:
                        phi_j_minus_1_k = 0

                    curr_x = x[j]
                    phi_k_plus_1 = phi_j_k + I * Dt/Dx**2 * (phi_j_plus_1_k - 2*phi_j_k + phi_j_minus_1_k) - 2*I*Dt*phi_j_k*V(curr_x)
                except RuntimeWarning:
                    overflow = True
                finally:
                    next_wf.append(phi_k_plus_1)
            wf[:,t] = next_wf
            if overflow:
                print('\x1b[2k\x1b[100D\x1b[1A')
                console.print(f"[red]Warning {warn_number}: Encountered large number!! Your simulation is exploding, please take cover =).")
                #print(f"{bcolors.WARNING}Warning {warn_number}: Encountered large numbers!! Your simulation is exploding, please take cover =).{bcolors.ENDC}")
                warn_number += 1
            progress.update(task1, advance=1, refresh=True, date_time=False)


    # CALCULATE THE MODULUS
    wf_modulus_squared = modulus_squared(wf, warn_number)
    return wf_modulus_squared


# PLOT ANIMATION
def animate(i):
    y = wf_modulus_squared[:,i]

    ax.clear()
    ax.axvspan(0, l, alpha=0.5, color='black')
    ax.plot(x, y, color='blue', alpha=0.5)
    ax.fill_between(x, y, alpha=0.5, color='blue')
    ax.set_xlim([-60,60])
    ax.set_ylim([0,1])
    ax.set_xlabel("Position")
    ax.set_ylabel("Squared Wavefunction Modulus")

# MAIN FUNCTION
if __name__=="__main__":
    x, t = generate_x_and_t(M, Dx, x_min, N, Dt)#create the space and time vectors
    wf_modulus_squared = main(x, t, warn_number)

    print(f"{bcolors.FAIL}BOOM!! Your simulation exploded! Good old Euler!{bcolors.ENDC}")
    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, animate, frames=100, interval=10, repeat=True)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    
    





    
