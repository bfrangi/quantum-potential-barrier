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

# SUBSCRIPTS, SUPERSCRIPTS AND FORMATTING
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

def format_exp(num):
    num = str(num)
    match = re.findall(r'[\+]?([0123456789\.\-]+)(?:e[+]?([0123456789\-]+))?', num)[0]
    num = match[0] + "·10" + match[1].translate(SUP)
    return num

# IMPORT MATRIX FROM .NPY
def import_matrix(filename):
	f = open(filename, 'rb')
	matrix = np.load(f)
	f.close()
	return matrix

# EXPORT MATRIX TO .NPY
def export_matrix(matrix, filename):
	f = open(filename, 'wb')
	np.save(f, matrix)
	f.close()


# STATISTICS FUNCTIONS
def list_average(lst):
    return sum(lst) / float(len(lst))

def list_standard_deviation(lst, num):
    return np.sqrt( sum( [ ( value - num )**2 for value in lst ] ) / float( len( lst ) ) )