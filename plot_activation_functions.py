# --- coding: utf-8 ---
# @FileName  :plot_activation_functions.py
# @Time      :1/13/2023 4:37 PM
# @Author    :Abraham
# @Software  :PyCharm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def linear(x):
    """ Linear activation function (also called "no activation").
    Args:
        x: float tensor to perform activation.
    Returns:
        x
    """
    return x


def sigmoid(x):
    """ Sigmoid activation function return a value which lies between 0 and 1.
    Args:
         x: float tensor to perform activation.
    Returns:
        1/(1+exp(-x))
    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """ Hyperbolic tangent function (Tanh) returns a value in between -1 and 1.
    Args:
        x: float tensor to perform activation.
    Returns:
        '(e^x - e^-x) / (e^x + e^-x)'
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    """ Rectifier Linear Unit (ReLU).
    Args:
        x: float tensor to perform activation.
    Returns:
        'x' if the value of x is greater than 0 otherwise it returns 0."""
    return np.maximum(0, x)


def leaky_relu(x):
    """ Leaky Rectifier Linear Unit activation function.
    Args:
        x: float tensor to perform activation.
    Returns:
        'x' if x greater than zero, otherwise it returns a*x.
        where a is a small constant number.
    """
    return np.where(x > 0, x, x * 0.05)


def elu(x):
    """ Exponential Linear Units (ELUs)
    Args:
        x: float tensor to perform activation.
    Returns:
        'x' for positive x, and exp(x) - 1 for negative value of x.
    """
    return np.where(x > 0, x, np.exp(x) - 1)


def softmax(x):
    """ Compute softmax values for each sets of scores in x.
    Args:
        x: float tensor to perform activation.
    Returns:
        'x' with the softmax activation applied
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def gelu(x):
    """ Gaussian Error Linear Unit (GELU).
    Args:
        x: float tensor to perform activation.
    Returns:
        'x' with the GELU activation applied.
    """
    return x * norm.cdf(x)


def gelu_approx(x):
    """ Approximation of Gaussian Error Linear Unit (GELU).
    Args:
        x: float tensor to perform activation.
    Returns:
        'x' with the GELU activation applied.
    """
    return 0.5 * x * (1 + tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def plot():
    """ Plots different types of activation function. """
    x = np.linspace(-6, 6)
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(10, 7))
    # For Linear activation function
    ax1 = plt.subplot(3, 3, 1, frameon=True)
    ax1.plot(x, linear(x))
    ax1.set_title("Linear function")
    # For Sigmoid
    ax2 = plt.subplot(3, 3, 2, sharex=ax1, facecolor='orange')
    ax2.plot(x, sigmoid(x), '-r')
    ax2.set_title("Sigmoid function")
    # For hyperbolic tangent (tanh)
    ax3 = plt.subplot(3, 3, 3, sharex=ax1)
    ax3.plot(x, tanh(x), '-g')
    ax3.set_title("Tanh function")
    # For ReLU
    ax4 = plt.subplot(3, 3, 4, facecolor='orchid')
    ax4.plot(x, relu(x), '-b')
    ax4.set_title("ReLU function")
    # For leaky ReLU
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(x, leaky_relu(x), '-c')
    ax5.set_title("Leaky ReLU function")
    # For ELU
    ax6 = plt.subplot(3, 3, 6, facecolor='green')
    ax6.plot(x, elu(x), '-m')
    ax6.set_title("ELU function")
    # For softmax activation function
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(x, softmax(x), '-y')
    ax7.set_title("Softmax function")
    # For GELU activation function
    ax8 = plt.subplot(3, 3, 8, facecolor='blue')
    ax8.plot(x, gelu(x), '-r')
    ax8.set_title("GELU function")
    # For GELU approximate, same to GELU
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(x, gelu_approx(x), '-k')
    ax9.set_title("GELU approximate function")
    fig.tight_layout()
    plt.savefig("Activation functions.jpeg")
    plt.show()


plot()

