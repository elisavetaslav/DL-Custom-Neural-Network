import numpy as np
import scipy
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(0, input)


    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """

        output = np.zeros(input.shape)
        output[input > 0] = 1
        return grad_output * output


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return 1 / (1 + np.exp(-input))

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        compute_out = self.compute_output(input)
        return grad_output * compute_out * (1 - compute_out)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        exp_input = np.exp(input)
        sum_exp = np.sum(exp_input, axis=1, keepdims=True) #(n, ) РґР°?? РІСЂРѕРґРµ РЅРµ tuple СѓСЂР°
        return exp_input / sum_exp

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        compute_out = self.compute_output(input)
        return compute_out * (grad_output - np.sum(grad_output * compute_out, axis=1, keepdims=True))


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return scipy.special.log_softmax(input, axis=1)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        exp_input = np.exp(input)
        sum_exp = np.sum(exp_input, axis=1, keepdims=True)
        s_out = (exp_input / sum_exp)
        grad_soft = grad_output / s_out
        return s_out * (grad_soft - np.sum(grad_soft * s_out, axis=1, keepdims=True))
