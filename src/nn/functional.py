import numpy as np

from src.utils.template import BaseClass


class ReLU(BaseClass):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = np.maximum(0, x)   
        return self.output

    def backward(self, grad):
        return grad * np.clip(self.output, 0, 1)
    

class Softmax(BaseClass):
    def __init__(self, axis=1):
        self.axis = axis
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x_shifted = np.exp(x - np.max(x, axis=self.axis, keepdims=True))
        value = exp_x_shifted / np.sum(exp_x_shifted, axis=self.axis, keepdims=True)
        return value
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.value = self.softmax(x)
        return self.value

    def backward(self, input_grad: np.ndarray) -> np.ndarray:
        #TODO: fix softmax backward function
        
        #output_grad = np.zeros(input_grad.shape)
        #for i in range(input_grad.shape[0]):
        #    d = np.diagflat(self.value[i]) - np.dot(self.value[i], self.value[i])
        #    output_grad[i] = np.dot(input_grad[i], d)

        return 1 * input_grad
    
    
class MSELoss():
    def __init__(self, reduction='mean'):
        assert reduction in ['mean', 'sum']
        self.reduction = reduction
        
    def __call__(self, y: np.ndarray, true: np.ndarray) -> float:
        self.y = y
        self.true = true
        
        assert y.shape == true.shape
        if self.reduction == 'mean':
            return np.mean((self.y - self.true) ** 2)
        elif self.reduction == 'sum':
            return np.sum((self.y - self.true) ** 2)
        
    def backward(self) -> np.ndarray:
        return 2 * (self.y - self.true) / self.y.shape[0]
