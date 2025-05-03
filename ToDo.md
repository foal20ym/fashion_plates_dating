ToDo / Future Work:

1. **Monitor and Log Data Augmentation Parameters**
   - Track and save which fashion plates were flipped or otherwise augmented during training.
   - Store augmentation parameters and image names in a file for later analysis.
   - Use this information to better understand why certain models or approaches may have failed.

2. **Repeat Training for Robustness**
   - Train each model/approach multiple times to better estimate their true capabilities and reduce variance in results.

3. **Extend Study to Fashion Artifacts**
   - Apply the methodology to more challenging artifacts such as bodices, skirts, and full ensembles.
   - Address challenges such as limited artifact numbers, varying image quality, and inconsistent photography conditions.

4. **Experiment with Image Augmentation**
   - Investigate the effects of augmenting color and saturation on model performance.

5. **Optimize Batch Size**
   - Experiment with different batch sizes, as using a single batch size for all models may have limited performance.

6. **Incorporate Knowledge Graphs**
   - Explore the use of knowledge graphs to inject expert knowledge into the system, potentially improving reliability.

7. **Expand Timeframe of Study**
   - Extend the period of study beyond 1820-1880 to include more artifacts.
   - Address challenges such as data sparsity and reduced accuracy with a broader historical range.

Todo i koden: 
   - Fixa så man kan spara ner modellerna
   - FIxa så alla parameterar ligger i en JSON eller txt fil som läses in
   

