ToDo / Future Work:

1. **Monitor and Log Data Augmentation Parameters**
   - Track and save which fashion plates were flipped or otherwise augmented during training.
   - Store augmentation parameters and image names in a file for later analysis.
   - Use this information to better understand why certain models or approaches may have failed.

2. **Repeat Training for Robustness**
   - Train each model/approach multiple times to better estimate their true capabilities and reduce variance in results.
   - Done

3. **Extend Study to Fashion Artifacts**
   - Apply the methodology to more challenging artifacts such as bodices, skirts, and full ensembles.
   - Address challenges such as limited artifact numbers, varying image quality, and inconsistent photography conditions.

4. **Experiment with Image Augmentation**
   - Investigate the effects of augmenting color and saturation on model performance.
   - Done

5. **Optimize Batch Size**
   - Experiment with different batch sizes, as using a single batch size for all models may have limited performance.
   - Done

6. **Incorporate Knowledge Graphs**
   - Explore the use of knowledge graphs to inject expert knowledge into the system, potentially improving reliability.

7. **Expand Timeframe of Study**
   - Extend the period of study beyond 1820-1880 to include more artifacts.
   - Address challenges such as data sparsity and reduced accuracy with a broader historical range.

8. **Implement new model**
   - Done EfficentNetB3 is included

9.  **Plot ten fold mean correctly**
   - Done


## Experiments:
1. **Hyperparameter tuning:**
   - batch size
   - dropout yes/no value [0 - 1]
   - l2_regularization yes/no value [-> 0]
   - learning rate, start value and factor in ReduceLROnPlateau
   - Add Dense layers and fine tuning layers
     - All above parameters has been tuned on InceptionV3 and EfficientNetB3 which helped a lot.
2. **Different loss function**
     - Testa Ordinal categorical cross entropy: https://github.com/JHart96/keras_ordinal_categorical_crossentropy/blob/master/ordinal_categorical_crossentropy.py 
     - CategoricalFocalCrossentropy: https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalFocalCrossentropy 
       - focal and ordinal, ordinal seems best but no big difference.
3. **larger image size**
   - InceptionV3 trained on 299x299 and EfficientNetB3 on 255x255