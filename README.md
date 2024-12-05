# hse-mlp-implementation

<h3 align="center">
    <p>Лабораторная работа 2.<br> Реализация MLP классификатора и инструментов для тренировки на numpy</p>
</h3>

## Выполенные пункты из ТЗ к заданию
- [x] Линейный forward + backward `src/nn/linear.py`
- [x] Классы активации softmax и relu `src/nn/functionals.py`
- [x] Класс MSELoss `src/nn/functionals.py`
- [x] Численное совпадение метода forward всех классов `tests/nn_tests.py`
- [ ] Численное совпадение метода backward всех классов 
- [x] Accuracy больше 50%


## Реализованные элементы
- Линейный слой в файле `src/nn/linear.py`<br>
Реализованы методы: <br> __\_\_init\_\___ с инциализацией весов `init.kaiming_uniform_` из [PyTorch](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_). такая инициализация стоит по дефолту в `nn.liearn`. <br>
__forward__ и __backward__ - методы для  прямого и обратного распространения. Размерности входных и выходных данных покрыты assert, чтобы не допустить проблем с умножением матриц.


- Relu, softmax, MSELoss в файле `src/nn/functionals.py`<br>
Реализованы методы: <br>
__forward__ и __backward__ - методы для  прямого и обратного распространения. 


- SGD optimizer в файле `src/optim/optim.py`<br>
__SGD optimizer__ с функционалом схожим с функционалом из [PyTorch](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_)
После __backward__ в нейронной сети, обновляет веса в каждом слое

```python
>>> preds = self.model.forward(images)
>>> loss = self.criterion(preds, labels) #criterion = MSELoss

>>> self.model.layers = self.optimizer.step(self.model.layers, self.criterion) #optimizer = SGD
```

- Trainer в файле `src/utils/trainer.py`<br>
Общий инструмет для тренировки MLP. Включает в себя __train__, __evaluation__ и __prediction__.<br>
Пример вызова Trainer диспользованная для треировки MLP.
```python
>>> trainer_config = {
    "model": model, 
    "optimizer": optimizer, 
    "criterion": criterion, 
    "do_train": True,
    "do_eval": True,
    "epochs": 15, 
    "batch_size": 64,
    "eval_metric": "accuracy"
}

>>> trainer = Trainer(**trainer_config)
>>> trainer.train(train_data=train_dataloader, eval_data=test_dataloader)
```
Класс вдохновлен инструментом [Trainer из Transformers](https://huggingface.co/docs/transformers/main_classes/trainer)

- MNIST dataset в файле `src/utils/dataset.py`<br>
Основноый датасет в лабораторной работе - MNIST, поэтому подготовлен класс для подготовки этого датасета с методами \_\_len\_\_ и \_\_get\_item\_\_, для возможности использовать его с [torch.dataloader](https://pytorch.org/docs/stable/data.html).


## Тесты
В файле `tests/nn_tests.py` находятся тесты для слоев и функций активаций. Тесты реализованы на __pytests__.

## Результаты тренировки
В `notebooks/pipeline.py` написан пайплайн тренировки и инференса модели на предоставленном MNIST датасете.
После 10 эпох, accuracy метрика достигает ~0.95.  