from typing import Any

import numpy as np

from torch.utils.data import DataLoader

from tqdm import tqdm

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self, 
        model: Any, 
        optimizer: Any, 
        criterion: Any, 
        epochs: int,
        do_train: bool=True,
        do_eval: bool=False,
        batch_size: int=64,
        eval_metric: Any=None,                   
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.do_train = do_train
        self.do_eval = do_eval
        self.batch_size = batch_size
        
        self._init_metric(eval_metric)
    
    def _prepare_data(self, train_data=None, eval_data=None) -> None:
        train_dataloader = train_data
        eval_dataloader = eval_data
        
        if self.do_train and train_data:
            if not isinstance(train_data, DataLoader):
                print(f'Input data has {type(train_data)} format, creating train dataloader')
                train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        if self.do_eval and eval_data:
            if not isinstance(eval_data, DataLoader):
                print(f'Input data has {type(eval_data)} format, creating test dataloader')
                eval_dataloader = DataLoader(eval_data, batch_size=1)

        return train_dataloader, eval_dataloader
        
    def train(self, 
        train_data=None, 
        eval_data=None
    ) -> None:
        
        train_dataloader, eval_dataloader = self._prepare_data(train_data, eval_data)
        
        if self.do_train:
            assert train_dataloader != None
            train_losses = []
            eval_losses = [] if self.do_eval else None
            for epoch in range(self.epochs):
                loss = self._train_loop(train_dataloader, epoch)
                train_losses.append(loss)
                print(f'Train Loss: {loss}')
            
                if self.do_eval:
                    assert eval_dataloader != None
                    with_preds = True if self.eval_metric else False
                    eval_results = self._eval_loop(eval_dataloader, with_preds=with_preds)
                    eval_losses.append(eval_results[0])
                    print(f'Eval loss: {eval_results[0]}')
                    if with_preds:
                        preds = eval_results[1]
                        gt = eval_results[2]
                        print(f'Eval metric: {self.eval_metric(preds, gt)}')  
                        
        self._plot(train_losses, eval_losses)
        
    def predict(self, test_dataset, return_preds=None) -> None:
        assert self.eval_metric != None
        if not isinstance(test_dataset, DataLoader):
            test_dataloader = DataLoader(test_dataset, batch_size=1)
                
        eval_results = self._eval_loop(test_dataloader, with_preds=True)
        preds = eval_results[1]
        gt = eval_results[2]
        
        if return_preds:
            return preds, gt, self.eval_metric(preds, gt)

        return self.eval_metric(preds, gt)
            
    def _train_loop(self, train_data: DataLoader, epoch: int) -> None:        
        loss_sum = 0

        for batch in tqdm(train_data, desc=f'Epoch {epoch}'):
            images = batch[0]
            if not isinstance(images, np.ndarray):
                images = images.numpy()
            
            labels = batch[1]
            if not isinstance(labels, np.ndarray):   
                labels = labels.numpy()

            preds = self.model.forward(images)
            loss = self.criterion(preds, labels)
            loss_sum += loss.item()
            self.model.layers = self.optimizer.step(self.model.layers, self.criterion)
        
        return loss_sum / len(train_data)
    
    def _eval_loop(self, eval_data, with_preds=False) -> float:
        infer_loss = 0
    
        infer_predictions = np.array([], dtype=np.float64) if with_preds else None
        true_labels = np.array([], dtype=np.float64) if with_preds else None
        
        for batch in eval_data:
            images = batch[0]
            if not isinstance(images, np.ndarray):
                images = images.numpy()
            
            labels = batch[1]
            if not isinstance(labels, np.ndarray):   
                labels = labels.numpy()
            
            preds = self.model.forward(images)
            infer_loss += self.criterion(preds, labels).item()
            
            if with_preds:
                infer_predictions = np.append(infer_predictions, np.argmax(preds, axis=1))
                true_labels = np.append(true_labels, np.argmax(labels, axis=1))
            
        infer_loss /= len(eval_data)
        return [infer_loss, infer_predictions, true_labels] if with_preds else [infer_loss]
    
    def _init_metric(self, metric: str) -> None:
        if metric == 'accuracy':
            from sklearn.metrics import accuracy_score
            self.eval_metric = accuracy_score
    
    def _plot(self, train_loss, eval_loss) -> None:
        x = range(1, self.epochs + 1)
        
        plt.plot(x, train_loss, label='train loss')
        plt.plot(x, eval_loss, label='eval loss')
        
        plt.xlabel('epoch')
        plt.ylabel('loss')
        
        plt.legend()
        plt.show()

        
