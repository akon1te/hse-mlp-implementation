from typing import Any

import numpy as np


class Optimizer:
    def __init__(self) -> None:
        pass
    
    def step(self, model: Any, loss: Any) -> None:
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        
    def step(self, model: Any, loss: Any) -> None:
        error_gradient = loss.backward()
        for layer in model.layers[::-1]:
            layer.backward(error_gradient)
            layer.weights -= self.learning_rate * layer.weights_grad
            if layer.bias is not None:
                layer.bias -= self.learning_rate * layer.bias_grad
                
#TODO: V2 implement Adam (RMSProp / Adagrad) optimizers
         

