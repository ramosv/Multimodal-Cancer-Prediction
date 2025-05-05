## Results summary

## Random Forest classifier on omics data only
```bash
               precision    recall  f1-score   support

            0       0.57      0.80      0.67        15
            1       0.00      0.00      0.00         8
            2       0.20      0.33      0.25         6
            3       0.00      0.00      0.00         2

      accuracy                           0.45        31
      macro avg       0.19      0.28      0.23        31
   weighted avg       0.32      0.45      0.37        31

   0.45161290322580644
```

## CNN only

### with simple Transforms

```bash
Accuracy: 0.4
              precision    recall  f1-score   support

           0       0.50      0.58      0.54        12
           1       0.17      0.12      0.14         8

    accuracy                           0.40        20
   macro avg       0.33      0.35      0.34        20
weighted avg       0.37      0.40      0.38        20

Accuracy: 0.47368421052631576

```

### with brightness and contrast transform



```bash
              precision    recall  f1-score   support

           0       0.54      0.64      0.58        11
           1       0.33      0.25      0.29         8

    accuracy                           0.47        19
   macro avg       0.44      0.44      0.43        19
weighted avg       0.45      0.47      0.46        19
```
### CNN-only with cross validation

```bash
   Starting fold: 0
   Epoch 10, Loss: 1.3029
   Epoch 20, Loss: 1.2329
   Epoch 30, Loss: 1.1866
   Epoch 40, Loss: 1.1703
   Epoch 50, Loss: 1.1634
   Test Accuracy: 0.5000
   Starting fold: 1
   Epoch 10, Loss: 1.3247
   Epoch 20, Loss: 1.2647
   Epoch 30, Loss: 1.2247
   Epoch 40, Loss: 1.2071
   Epoch 50, Loss: 1.1990
   Test Accuracy: 0.5385
   Mean accuracy across all fold: 0.5192
```

### prefusion (concat) (best accuracy yet.)
(103, 11100)
(103, 6930)
Common patients: 27
(27, 18030)
(27, 129)

```bash
               precision    recall  f1-score   support

            0       0.83      1.00      0.91         5
            1       0.00      0.00      0.00         2
            2       0.00      0.00      0.00         1
            3       0.00      0.00      0.00         1

      accuracy                           0.56         9
      macro avg       0.21      0.25      0.23         9
   weighted avg       0.46      0.56      0.51         9

   Accuracy: 0.5555555555555556

```


### binary prefusion
```bash
              precision    recall  f1-score   support

           0       0.83      1.00      0.91         5
           1       1.00      0.75      0.86         4

    accuracy                           0.89         9
   macro avg       0.92      0.88      0.88         9
weighted avg       0.91      0.89      0.89         9

Accuracy: 0.8888888888888888
```

### binary with cross validation
```bash
Name: tumor_stage_pathological, dtype: int64
Starting fold: 0
Accuracy: 0.8571
Classification Report:
              precision    recall  f1-score   support

           0     0.8571    0.8571    0.8571         7
           1     0.8571    0.8571    0.8571         7

    accuracy                         0.8571        14
   macro avg     0.8571    0.8571    0.8571        14
weighted avg     0.8571    0.8571    0.8571        14

Starting fold: 1
Accuracy: 0.7692
Classification Report:
              precision    recall  f1-score   support

           0     0.7000    1.0000    0.8235         7
           1     1.0000    0.5000    0.6667         6

    accuracy                         0.7692        13
   macro avg     0.8500    0.7500    0.7451        13
weighted avg     0.8385    0.7692    0.7511        13

Mean accuracy across all folds: 0.8132
```