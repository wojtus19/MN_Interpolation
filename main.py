import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pandas as pd
from math import sqrt


def Lagrange(x,y, x1, n):
    sum = 0
    for i in range (n):
        Lvalue = 1
        for j in range(n):
            if i != j:
                Lvalue *= (x1 - x[j])/(x[i] - x[j])
        Lvalue *= y[i]     
        sum += Lvalue
    return sum

def Spline(x0, x, y):
    """
    x0 - tablica wartości do interpolowania
    x - tablica xów do stworzenia funkcji (splajnów)
    y - tablica wartości do stworzenia funkcji (splajnów)
    """
    x = np.asfarray(x)
    y = np.asfarray(y)

    size = len(x)

    # tablice różnic między kolejnymi xami i ygrekami
    xdiff = np.diff(x)
    ydiff = np.diff(y)

    # tablice - bufory
    Li = np.empty(size)
    Li_1 = np.empty(size-1)
    z = np.empty(size)

    # wypełnienie Li i Li-1 i rozwiązanie [L][y] = [B]
    Li[0] = sqrt(2*xdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0 # zerowanie się na krańcach drugiej pochodnej - naturalna granica
    z[0] = B0 / Li[0]

    for i in range(1, size-1, 1):
        Li_1[i] = xdiff[i-1] / Li[i-1]
        Li[i] = sqrt(2*(xdiff[i-1]+xdiff[i]) - Li_1[i-1] * Li_1[i-1])
        Bi = 6*(ydiff[i]/xdiff[i] - ydiff[i-1]/xdiff[i-1])
        z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]

    i = size - 1
    Li_1[i-1] = xdiff[-1] / Li[i-1]
    Li[i] = sqrt(2*xdiff[-1] - Li_1[i-1] * Li_1[i-1])
    Bi = 0.0 # zerowanie się na krańcach drugiej pochodnej - naturalna granica
    z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]

    # rozwiąż [L.T][x] = [y]
    i = size-1
    z[i] = z[i] / Li[i]
    for i in range(size-2, -1, -1):
        z[i] = (z[i] - Li_1[i-1]*z[i+1])/Li[i]

    # znajdź index
    index = x.searchsorted(x0)
    np.clip(index, 1, size-1, index)

    xi1, xi0 = x[index], x[index-1]
    yi1, yi0 = y[index], y[index-1]
    zi1, zi0 = z[index], z[index-1]
    hi1 = xi1 - xi0

    # oblicz wielomian trzeciego stopnia
    f0 = zi0/(6*hi1)*(xi1-x0)**3 + \
         zi1/(6*hi1)*(x0-xi0)**3 + \
         (yi1/hi1 - zi1*hi1/6)*(x0-xi0) + \
         (yi0/hi1 - zi0*hi1/6)*(xi1-x0)
    return f0


script_dir = os.path.dirname(os.path.abspath(__file__)) + "\\"
csvs = ['Hel.csv', "MountEverest.csv", "WielkiKanionKolorado.csv"]

densities = [5, 10,40,50,100]
d = 1
for density in densities:
    for i in range((len(csvs))):
        text = csvs[i]
        text = text.replace(".csv","")
        n = density
        num_lines = sum(1 for l in open(script_dir + csvs[i]))
        skip_idx = [x for x in range(0, num_lines) if x % n != 0]
        data = pd.read_csv(script_dir + csvs[i], skiprows=skip_idx,header=None)
        daneF = pd.read_csv(script_dir + csvs[i], header=None)
        

        x = []
        y = []
        for j in range (len(data)):
            x.append(data.iloc[j,0])
            y.append(data.iloc[j,1])

        xf =[]
        yf = []
        for i in range(num_lines):
            xf.append(daneF.iloc[i][0])
            yf.append(daneF.iloc[i][1])

        lagrangeOutput = []
        dataX = []
        dataY = []
        for k in range(num_lines):
            if (k % d != 0):
                continue
            res = Lagrange(x,y,xf[k], num_lines - len(skip_idx))
            dataX.append(xf[k])
            dataY.append(yf[k])
            lagrangeOutput.append(res)


        splineOutput = Spline(dataX, x, y)

        plt.rc('figure', figsize = [18,8], autolayout = True)
        plt.figure(num='Wojciech Niewiadomski')

        plt.scatter(x, y, marker = '.', color='red')
        plt.plot(dataX, lagrangeOutput, label='Lagrange')
        plt.plot(daneF.iloc[:,0], daneF.iloc[:,1], color="green")
        plt.legend(('F(x)', 'f(x)', 'Punkty wezlowe'),loc = 'upper right')
        plt.title("Wykres " + str(text) + str(density) + " - Lagrange")
        plt.ylabel("Wysokosc [m]")
        plt.xlabel("Odleglosc [m]")
        plt.grid(color = '0.75', linestyle='-', linewidth=0.22)
        plt.savefig(script_dir + "Wykres" + str(text) + str(density) + "Lagrange" + '.png')
        plt.close()

        plt.rc('figure', figsize = [18,8], autolayout = True)
        plt.figure(num='Wojciech Niewiadomski')
        plt.scatter(x, y, marker = '.', color='red')
        plt.plot(dataX, splineOutput, label="Splajny")
        plt.plot(daneF.iloc[:,0], daneF.iloc[:,1], color="green")
        plt.legend(('F(x)', 'f(x)', 'Punkty wezlowe'),loc = 'upper right')
        plt.title("Wykres " + str(text) + str(density) + " - Splajny")
        plt.ylabel("Wysokosc [m]")
        plt.xlabel("Odleglosc [m]")
        plt.grid(color = '0.75', linestyle='-', linewidth=0.22)
        plt.savefig(script_dir + "Wykres" + str(text) + str(density) + "Splajny" + '.png')
        #plt.close()
        plt.show()
