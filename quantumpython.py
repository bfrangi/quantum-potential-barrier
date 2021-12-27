import re
import numpy as np
from numpy import pi
from progressbar import AnimatedMarker, Bar, ETA, FileTransferSpeed, Percentage, ProgressBar
widgets = ['Computing:', Percentage(), ' ', AnimatedMarker(markers='-\|/'),' ', Bar('█'), ' ', ETA(), ' ', FileTransferSpeed(unit="it")]

# MATHEMATICAL CONSTANTS
e = np.exp(1) #Constant e
I = 1j #Complex unit

# CONSTANTS FOR NORMALIZATION
m_e = 9.1093837015*10**(-31) #Mass of the electron
epsilon_0 = 8.85*10**(-12) #Vacuum permitivity
h_bar = 1.054571*10**(-34) #Reduced Plank's constant
q_e = 1.6*10**(-19) #Elementary charge
a_0 = 4 * pi * epsilon_0 * h_bar**2 / ( m_e * q_e**2 ) #Bohr's Radius
E_h = h_bar**2/ ( m_e * a_0**2 ) 

# TEXT COLORS
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# SUBSCRIPTS AND SUPERSCRIPTS
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

# IMPORT WAVE FUNCTION MATRIX FROM .TXT
def import_wavefunction_mat(filename="wavefunction.txt"):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    rows = len(lines[0].strip().split("\t"))
    columns = len(lines)
    
    mat = np.zeros([rows, columns], dtype='complex')
    
    print("Importing Wave Function Matrix from File", filename)
    progressbar = ProgressBar(widgets=widgets, maxval=10000000)
    for column in progressbar(range(columns)):
        col = lines[column].strip().split("\t")
        for row in range(rows):
            num = col[row].strip()
            reg = re.findall(r'[\(]?([\+\-0123456789\.]+(?:e[\-]?[0-9]+)?(?=[\+\-]))?[\+]?([\-0123456789\.]+(?:e[\-]?[0-9]+)?)j[\)]?', num)
            real_part, complex_part = reg[0][0], reg[0][1]
            if real_part:
                real_part = float(real_part)
            else:
                real_part = 0
            if complex_part:
                complex_part = float(complex_part)
            else:
                complex_part = 0
            mat[row, column] = real_part + complex_part * I
    
    return mat

# IMPORT WAVEFUNCTION MODULUS SQUARED MATRIX FROM .TXT
def import_wavefunction_modulus_mat(filename="wavefunction_modulus_squared.txt"):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    rows = len(lines[0].strip().split("\t"))
    columns = len(lines)

    mat = np.zeros([rows, columns])

    print("Importing Wave Function Modulus Squared Matrix from File", filename)
    progressbar = ProgressBar(widgets=widgets, maxval=10000000)
    for column in progressbar(range(columns)):
        col = lines[column].strip().split("\t")
        for row in range(rows):
            mat[row, column] = float(col[row].strip())
    
    return mat

# EXPORT MATRIX TO .TXT
def export_mat(mat, filename="wavefunction.txt"):
    progressbar = ProgressBar(widgets=widgets, maxval=10000000)
    rows, columns = np.shape(mat)
    f = open(filename, 'w')
    for column in progressbar(range(columns)):
        for row in range(rows):
            f.write(str(np.round(mat[row, column], 10)) + "\t")
        f.write("\n")
    f.close()