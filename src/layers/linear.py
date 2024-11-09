import numpy as np

class Linear:
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = np.zeros((self.in_features, self.out_features))
        if bias:
            self.bias = np.zeros(self.out_features)    
        else:
            self.bias = None

        self.init_weights()

    def init_weights(self, how: str='normal') -> None:
        if how == 'normal':
            self.weight = np.random.normal(0.0, 0.01, size = (self.out_features, self.in_features))
            if self.bias is not None:
                self.bias = np.random.normal(0, 0.01, size = (self.out_features))
    
    def forwad(self, inputs: np.array) -> np.array:
        assert inputs.shape[0] == self.weight.shape[1]
        return np.dot(self.weight, inputs) + self.bias        
        
    def backward(self, gradient, lr: float):
        pass
        
        