# Experiments

## Experiment 1:
- run all three models same way thesis did with regression
- run all three models same way thesis did with classification

## Experiment 2:
- run all three models same way thesis did with regression and fine tune one layer
- run all three models same way thesis did with classification and fine tune one layer

Configurations for experiment 1 and 2:

task: classification and regression

model:
  <!-- # name: NASNetMobile -->
  <!-- # name: InceptionV3 -->
  name: ResNet101
  include_top: false
  fine_tune: false
  <!-- true on experiment 2 -->
  save_model: false
  use_l2_regularization: false
  l2_regularization: 1e-4
  use_dropout: false
  dropout: 0.25

training:
  batch_size: 16
  epochs: 100
  learning_rate: 0.01
  early_stopping_patience: 4
  reduce_lr_patience: 3

cross_validation: false

learning rate = 0.1 made the models get stuck in local minimum, so we lowered it to 0.01

## Extending



## Notes - Alexander
Hmm jag gjorde en annan intressant upptäckt i de körningarna jag har gjort nu idag. 
Här är ett exempel från en körning: 
```
Image: data/datasets/public/1870/1877_033met.jpg, True: 1877, Predicted: 1820, Error: 57
Image: data/datasets/public/1870/1877_035met.jpg, True: 1877, Predicted: 1820, Error: 57
Image: data/datasets/public/1870/1877_034met.jpg, True: 1877, Predicted: 1820, Error: 57
Image: data/datasets/public/1870/1876_53washington.jpg, True: 1876, Predicted: 1820, Error: 56
Image: data/datasets/public/1870/1876_449vna.jpg, True: 1876, Predicted: 1820, Error: 56
data/datasets/private/1870/1871_415etsy.jpg, True: 1871, Predicted: 1820, Error: 51
Image: data/datasets/public/1820/1820_023met.jpg, True: 1820, Predicted: 1871, Error: 51
Image: data/datasets/public/1820/1820_115_001wikimedia2.jpg, True: 1820, Predicted: 1871, Error: 51
Image: data/datasets/private/1870/1871_114etsy.jpg, True: 1871, Predicted: 1820, Error: 51
```
Notera hur den väldigt ofta predictar 1820 när rätt är 1871 och vice versa. 
Detta är dock bara när jag kört 15 epoker, 