import numpy as np
import matplotlib.pyplot as plt


def plot_io(u, y, title):
    plt.figure(figsize=(16, 9))
    plt.plot(u, label="u", color="k")
    plt.plot(y, label="y", color="r")
    plt.xlabel("timestep")
    plt.title(title)
    plt.legend()
    plt.grid(True)


def plot_y(y, y_pred, title):
    plt.figure(figsize=(16, 9))
    plt.plot(y, label="y", color="k", linestyle="--")
    plt.plot(y_pred, label="y_pred", color="r")
    plt.xlabel("timestep")
    plt.title(title)
    plt.legend()
    plt.grid(True)
