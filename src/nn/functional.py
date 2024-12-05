import numpy as np

from src.utils.template import BaseClass


class ReLU(BaseClass):
    def forward(self, x: np.ndarray) -> np.ndarray:  
        self.input = x
        return np.maximum(0, x) 

    def backward(self, grad):
        return np.where(self.input > 0, 1, 0) * grad
    

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

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        jac1 = np.zeros((self.value.shape[0], self.value.shape[1], self.value.shape[1]))
        jac2 = np.zeros((self.value.shape[0], self.value.shape[1], self.value.shape[1]))
        jac1 = np.einsum('ij,jk->ijk', self.value, np.eye(self.value.shape[-1]))
        jac2 = np.einsum('ij,ik->ijk', self.value, self.value)

        return ((jac1 - jac2) @ gradient[:, :, None]).squeeze(2)
    
    
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
