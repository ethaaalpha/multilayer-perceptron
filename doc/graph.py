from matplotlib import pyplot as pl
import numpy as np

def sigmoide():
    pl.figure("Sigmoide")

    x = np.linspace(-5, 5)
    y = 1/(1+np.exp(-x))

    pl.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    pl.plot(x, y)
    pl.show()

def linear_function():
    pl.figure("Linear function")

    # This display a fake "dataset"
    x_points = np.linspace(0, 7, 10)
    y_points = np.linspace(10, 15, 10)
    noise = np.random.uniform(-5, 5, size=y_points.shape)
    pl.plot(x_points, y_points + noise, 'o', color="red")

    x_points = np.linspace(10, 15, 10)
    y_points = np.linspace(0, 5, 10)
    noise = np.random.uniform(-5, 5, size=y_points.shape)
    pl.plot(x_points, y_points + noise, 'o', color="green")

    # This is the linear function
    x_points = np.linspace(0, 15)
    pl.plot(x_points, x_points*1.5 - 3.5)
    pl.show()

