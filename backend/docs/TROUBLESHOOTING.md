# Data Preparation Troubleshooting Guide

## Common Error Types

### File Errors
- **Symptoms**: FileNotFoundError, PermissionError
- **Solutions**:
  - Verify absolute paths are correct
  - Check file permissions (`chmod` on Linux/Mac)
  - Ensure preprocessing scripts ran successfully
  - Confirm CSV files aren't corrupted

### Validation Errors
- **Symptoms**: ValueError, TypeError
- **Solutions**:
  - Check for NaN/infinite values: `df.isna().sum()`
  - Verify feature dtypes match expectations
  - Ensure categorical features are properly encoded
  - Check for consistent feature shapes

### SMOTE-Related Issues
See [SMOTE Guide](SMOTE_GUIDE.md) for detailed solutions

## General Checks
1. Data Integrity:
   - `df.info()` - Check dtypes and null values
   - `df.describe()` - Verify statistical properties

2. Preprocessing:
   - Confirm all transformers were fit on training data only
   - Verify normalization/standardization ranges

3. Data Splits:
   - Check train/test/validation ratios
   - Verify no data leakage between splits

## Debugging Tips
```python
# Debug snippet for data issues
print("Feature shapes:", X_train.shape, X_test.shape)
print("Class distribution:", np.bincount(y_train))
print("NaN values:", np.isnan(X_train).sum())
```

[Back to Top](#data-preparation-troubleshooting-guide)