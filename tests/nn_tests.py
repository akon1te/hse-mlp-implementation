import pytest

import numpy as np

import torch
from torch import nn

from src.nn.linear import Linear
from src.nn.functional import ReLU, Softmax, MSELoss


@pytest.mark.parametrize("nn_input, random_shape2", [(5, 5), (10, 10), (1, 1), (20, 20)])
@pytest.mark.parametrize("nn_output, random_shape1", [(10, 10), (10, 20), (3, 3), (1, 1)])
def test_comparte_linear_forward_with_torch(nn_input, nn_output, random_shape1, random_shape2):
    linear = Linear(nn_input, nn_output)
    torch_linear = nn.Linear(nn_input, nn_output)
    torch_linear.weight = torch.nn.Parameter(torch.tensor(linear.weights, dtype=torch.float32), requires_grad=True)
    torch_linear.bias = torch.nn.Parameter(torch.tensor(linear.bias, dtype=torch.float32), requires_grad=True)

    x = np.random.rand(random_shape1, random_shape2)
    y = linear.forward(x)
    torch_y = torch_linear(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    
    #check if my_linear.shape is right
    assert y.shape == (random_shape1, nn_output)
    #check if my_linear == torch_linear
    assert (y - torch_y < 1e-6).all() == True
    
    
@pytest.mark.parametrize("random_shape1, random_shape2", [(5, 5), (10, 10), (1, 1), (20, 20)])
def test_comparte_relu_with_torch(random_shape1, random_shape2):
    relu = ReLU()
    torch_relu= nn.ReLU()

    x = np.random.rand(random_shape1, random_shape2)
    y = relu.forward(x)
    torch_y = torch_relu(torch.tensor(x, dtype=torch.float32)).detach().numpy()

    assert (y - torch_y < 1e-6).all() == True
    
    
@pytest.mark.parametrize("random_shape1, random_shape2", [(5, 5), (10, 10), (1, 1), (20, 20)])
def test_comparte_sm_with_torch(random_shape1, random_shape2):
    sm = Softmax()
    torch_sm = nn.Softmax()

    x = np.random.rand(random_shape1, random_shape2)
    y = sm.forward(x)
    torch_y = torch_sm(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    
    #check if sum of softmax == 1
    assert np.abs((np.sum(y, axis=1) - 1.) < 1e-6 ).all()
    #check if my_sm == torch_sm
    assert (np.abs(y - torch_y) < 1e-6).all()
    