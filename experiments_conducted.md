# Experiments

## Experiment 1:
- run all three models same way thesis did with regression
- run all three models same way thesis did with classification

## Experiment 2:
- run all three models same way thesis did with regression and fine tune one layer
- run all three models same way thesis did with classification and fine tune one layer

Configurations for experiment 1 and 2:

task: classification

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
