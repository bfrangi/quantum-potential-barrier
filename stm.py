N = 10 #Number of x values
x_min = 0 #Min value of x
x_max = 100 #Max value of x
Dt = (x_max - x_min) / float(N - 1) #Time interval


def generate_mesh(N, Dt, x_min):
    mesh = []
    for k in range(1, N + 1):
        space = {
            't': (k-1)*Dt,
            'x': []
        }
        for j in range(1, N + 1):
            x = x_min + (j - 1) * Dt
            space['x'].append(x)
        mesh.append(space)
    return mesh

def print_mesh(mesh):
    for i in mesh:
        print("t =", round(i['t'], 4), end='; x = [')
        for x_val in i['x']:
            print(round(x_val, 4), end=' ')
        print("]")

if __name__=="__main__":
    print_mesh(generate_mesh(N, Dt, x_min))