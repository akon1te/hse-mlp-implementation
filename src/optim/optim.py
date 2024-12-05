from typing import Any

import numpy as np


class Optimizer:
    def __init__(self) -> None:
        pass
    
    def step(self, layers: Any, loss: Any) -> None:
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        
    def step(self, layers: Any, loss: Any) -> None:
        #TODO: do passing class by reference
        gradient = loss.backward()
        for layer in layers[::-1]:
            gradient = layer.backward(gradient)
            if hasattr(layer, 'weights'):
                layer.weights -= self.learning_rate * layer.weights_grad.T
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias -= self.learning_rate * layer.bias_grad
        
        return layers
