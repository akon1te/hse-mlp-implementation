import math
from typing import Any

import numpy as np

import torch
from torch.nn import init

from src.utils.template import BaseClass


class Linear(BaseClass):
    
    in_features: int
    out_features: int
    weights: np.ndarray
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        dtype=np.float32,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        
        self.weights = np.zeros((self.out_features, self.in_features), dtype=self.dtype)
        if bias:
            self.bias = np.zeros(self.out_features, dtype=self.dtype)    
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch_weights = torch.from_numpy(self.weights)
        init.kaiming_uniform_(torch_weights, a=math.sqrt(5))
        self.weights = torch_weights.numpy().astype(self.dtype)
        
        if self.bias is not None:
            torch_bias = torch.from_numpy(self.bias)
            fan_in, _ = init._calculate_fan_in_and_fan_out(torch_weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(torch_bias, -bound, bound)
            self.bias = torch_bias.numpy().astype(self.dtype)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        
        assert inputs.shape[1] == self.weights.shape[1]
        self.inputs = inputs
        outputs = np.zeros((inputs.shape[0], self.out_features))
        for idx in range(inputs.shape[0]):
            outputs[idx] = np.dot(inputs[idx], self.weights.T) + self.bias
            
        assert outputs.shape == (inputs.shape[0], self.out_features)
        return outputs  
        
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        self.weights_grad = np.dot(self.inputs.T, gradient)
        self.bias_grad = np.sum(gradient, axis=0)
        return np.dot(gradient, self.weights)
        