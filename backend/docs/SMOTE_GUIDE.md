# SMOTE Configuration Guide

## Common SMOTE Errors

### `ValueError: Expected n_neighbors <= n_samples`
- **Cause**: Too few samples in minority class
- **Solutions**:
  - Reduce `k_neighbors` parameter (default=5)
  - Use SMOTE-NC for categorical features
  - Consider collecting more minority class samples

### Imbalanced Results
- **Symptoms**: Poor minority class performance
- **Solutions**:
  - Adjust `sampling_strategy`:
    ```python
    # For binary classification
    SMOTE(sampling_strategy={0: 1000, 1: 2000})
    
    # For multiclass
    SMOTE(sampling_strategy='minority')
    ```
  - Combine with RandomUnderSampler

## Best Practices

### Pre-SMOTE Preparation
1. Always split data (train/test) before applying SMOTE
2. Scale continuous features first
3. Encode categorical features appropriately

### Parameter Tuning
```python
from imblearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(
        sampling_strategy=0.5,  # Target minority/majority ratio
        k_neighbors=3,         # Reduce for small datasets
        random_state=42
    )),
    ('classifier', RandomForestClassifier())
])
```

## Alternatives to SMOTE
| Method          | Best For                  | Pros                      |
|-----------------|---------------------------|---------------------------|
| ADASYN          | Noisy datasets            | Adaptive sampling         |
| BorderlineSMOTE | Ambiguous boundary cases  | Focuses on edge cases     |
| SVMSMOTE        | High-dimensional data     | Uses SVM decision boundary|

[Back to Top](#smote-configuration-guide)