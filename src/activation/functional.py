import numpy as np


class ReLU():
    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.output = np.maximum(0, x)   
        return self.output

    def backward(self, grad):
        return grad * np.clip(self.output, 0, 1)
    

class Softmax:
    def __init__(self, axis=1):
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_max = np.amax(x, axis=self.axis, keepdims=True)
        exp_x_shifted = np.exp(x - x_max)
        return exp_x_shifted / np.sum(exp_x_shifted, axis=self.axis, keepdims=True)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        pass
    
    
class MSELoss():
    def __init__(self, reduction='mean'):
        assert reduction in ['mean', 'sum']
        self.reduction = reduction
        
    def __call__(self, y: np.ndarray, true: np.ndarray) -> float:
        assert y.shape == true.shape
        if self.reduction == 'mean':
            return np.mean((y - true) ** 2) / 2
        elif self.reduction == 'sum':
            return np.sum((y - true) ** 2) / 2
