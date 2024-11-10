import numpy as np


class Relu():
    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.output = np.maximum(0, x)   
        return self.output

    def backward(self, grad):
        return grad * np.clip(self.output, 0, 1)
    

class Softmax:
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        exps = exps - np.max(exps)
        sm_logits = np.exp(exps) / np.sum(np.exp(exps))
        return sm_logits

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        pass