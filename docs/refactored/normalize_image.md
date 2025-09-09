# torch::normalize_image / torch::normalizeImage

Normalizes an image tensor using mean and standard deviation values.

## Syntax

### Legacy Syntax (Positional Parameters)
```tcl
torch::normalize_image image mean std ?inplace?
```

### Modern Syntax (Named Parameters)
```tcl
torch::normalize_image -image tensor -mean tensor -std tensor ?-inplace bool?
torch::normalizeImage -image tensor -mean tensor -std tensor ?-inplace bool?  ;# camelCase alias
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| image/tensor | tensor | The input image tensor to normalize |
| mean | tensor | The mean values for normalization |
| std | tensor | The standard deviation values for normalization |
| inplace | boolean | Whether to perform the operation in-place (default: 0) |

## Description

The `normalize_image` command performs image normalization using the formula:
```
output = (input - mean) / std
```

This is commonly used in deep learning preprocessing pipelines to normalize image data before feeding it into neural networks.

The operation can be performed either in-place (modifying the input tensor) or out-of-place (creating a new tensor).

## Return Value

Returns a tensor handle:
- If `inplace` is 0 (default): Returns a new tensor handle for the normalized tensor
- If `inplace` is 1: Returns the input tensor handle

## Examples

### Using Legacy Syntax
```tcl
# Create sample tensors
set image [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
set mean [torch::tensor_create {2.0} float32]
set std [torch::tensor_create {2.0} float32]

# Normalize out-of-place
set result1 [torch::normalize_image $image $mean $std]

# Normalize in-place
set result2 [torch::normalize_image $image $mean $std 1]
```

### Using Modern Syntax
```tcl
# Using named parameters
set result3 [torch::normalize_image -image $image -mean $mean -std $std]

# Using named parameters with in-place operation
set result4 [torch::normalize_image -image $image -mean $mean -std $std -inplace 1]

# Using camelCase alias
set result5 [torch::normalizeImage -image $image -mean $mean -std $std]
```

## Error Conditions

The command will return an error in the following cases:
- If required parameters are missing
- If any of the tensor handles are invalid
- If the mean or std tensors have incompatible shapes with the input tensor

## See Also

- [torch::denormalize_image](denormalize_image.md) - Reverse the normalization operation
- [torch::tensor_create](tensor_create.md) - Create a new tensor 