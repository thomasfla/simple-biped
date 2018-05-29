import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

def plot_gain_stability(Kp_grid,Kd_grid,stab_grid):
    plt.contourf(Kp_grid,Kd_grid,stab_grid,1)
    plt.plot(Kp_grid.flatten(), Kd_grid.flatten(),'+')
    Kps = np.linspace(Kp_grid.min(), Kp_grid.max(),1000)
    Kds = 2.*np.sqrt(Kps)
    
    plt.xlabel('$K_p$')
    plt.ylabel('$K_d$')
    plt.plot(Kps,Kds)
    return plt


def main():
    num = 1527601104
    inDir = "./data/{}/".format(num)
    data = np.load(inDir + "stab.npz")
    stab_grid = data['stab_grid']
    Kp_grid   = data['Kp_grid']
    Kd_grid   = data['Kd_grid']
    plot_gain_stability(Kp_grid,Kd_grid,stab_grid)
    plt.savefig(inDir + "stab.png")
    plt.show()
    embed()
    return 0

if __name__ == '__main__':
	main()


