from typing import Any

import numpy as np


class Trainer:
    def __init__(
        self, 
        model: Any, 
        optimizer: Any, 
        criterion: Any, 
        epochs: int,
        do_train: bool=True,
        do_eval: bool=False,
        eval_metric: Any=None,                   
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.do_train = do_train
        self.do_eval = do_eval
        self.eval_metric = eval_metric
        
    def train(self, 
        train_data=None, 
        eval_data=None
    ) -> None:
        
        if self.do_train:
            assert train_data != None
            for epoch in range(self.epochs):
                print(f'Epoch {epoch + 1}/{self.epochs}')
                loss = self.train_step(train_data)
                print(f'Loss: {loss}')
            
                if self.do_eval:
                    assert eval_data != None
                    eval_loss = self.count_eval_loss(eval_data)
                    print(f'Eval loss: {eval_loss}')
                    
    def train_step(self, train_data: dict) -> None:        
        loss_sum = 0
        X = train_data['features']
        y = train_data['labels']

        for x, y in zip(X, y):
            predictions = self.model.forward(x)
            loss = self.criterion(predictions, y)
            loss_sum += loss
            self.model.layers = self.optimizer.step(self.model.layers, self.criterion)
        
        return loss_sum / len(X)
    
    def count_eval_loss(self, eval_data: dict) -> float:
        loss_sum = 0
        X = eval_data['features']
        y = eval_data['labels']
        
        for x, y in zip(X, y):
            y_hat = self.model.forward(x)
            loss_sum += self.criterion(y_hat, y)
            
        return loss_sum / len(X)
    
