import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix

np.random.seed(31415)
n_train, n_val = 1000, 100

def generate_data(n, path):
    fig = plt.figure()
    for i in range(n):
        if i%100 == 0:
            print(i)
        k = np.random.randint(3, 8)
        mean = (np.random.rand(k, 2) - 0.5) * 20
        generated = np.random.dirichlet(np.ones(k)/10.0,size=1)[0]

        x = np.linspace(-10., 10., 100)
        y = np.linspace(-10., 10., 100)
        Z = np.zeros((100, 100))
        XY = [[(xx, yy) for xx in x] for yy in y]
        for j in range(k):
            done = 0
            while done==0:
                mat = 5 * np.random.rand(2, 2)
                mat = mat.T @ mat
                done = 1
                try:
                    Z += generated[j] * multivariate_normal.pdf(XY, mean[j], mat)
                except:
                    done = 0

        plt.axis('off')
        plt.pcolormesh(Z, cmap='gray_r', )
        plt.savefig(path+str(i)+'.png', bbox_inches='tight')
        plt.clf()

        
generate_data(n_train, '../input/train/')
generate_data(n_val, '../input/val/')
