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
        
        self.weights = np.zeros((self.in_features, self.out_features))
        if bias:
            self.bias = np.zeros(self.out_features)    
        else:
            self.bias = None

        self.init_weights()

    def init_weights(self, how: str='normal') -> None:
        if how == 'normal':
            self.weights = np.random.normal(0.0, 0.01, size = (self.out_features, self.in_features))
            if self.bias is not None:
                self.bias = np.random.normal(0, 0.01, size = (self.out_features))
    
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        assert inputs.shape[0] == self.weights.shape[1]
        return np.dot(self.weights, inputs) + self.bias        
        
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        self.weights_grad = np.dot(self.input.T, gradient)
        self.bias_grad = np.sum(gradient, axis=0)
        return gradient @ self.weights.T
        