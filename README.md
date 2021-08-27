# Capture24: Activity recognition on a large activity tracker dataset collected in the wild

Code and benchmark

### Dependencies
* Hydra 1.0
* Pytorch
* Tensorboard
* sklearn

### How to setup


### How to run the benchmark models 
```shell
python cnn_lstm_train.py optim.weighted_cost=True
# this requires input X to be downsampled to 30Hz
python ssl_train.py optim.weighted_cost=True model.freeze_all=False 
```
Change the GPU field in `config_cnnlstm.yaml` if you want to run on CPU.


### How to make inference
```shell
python ssl_train.py eval=True gpu=2
python cnn_lstm_train.py eval=True gpu=1
```

