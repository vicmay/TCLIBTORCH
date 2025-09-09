# torch::upsample_nearest

Performs nearest neighbor upsampling on input tensors. This command supports both positional and named parameter syntax, and includes a camelCase alias.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::upsample_nearest input size
```

### Named Parameter Syntax (Recommended)
```tcl
torch::upsample_nearest -input tensor_handle -size {size_list}
torch::upsample_nearest -input tensor_handle -scale_factor {scale_list}
```

### CamelCase Alias
```tcl
torch::upsampleNearest input size
torch::upsampleNearest -input tensor_handle -size {size_list}
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `input` | tensor | Input tensor handle | Yes |
| `size` | list | Output size for each spatial dimension | Either `size` or `scale_factor` |
| `scale_factor` | list | Scaling factor for each spatial dimension | Either `size` or `scale_factor` |

## Input Tensor Requirements

The input tensor must have the correct dimensionality for PyTorch's interpolation:
- **3D tensors** (N, C, L): For 1D spatial upsampling, requires 1 size/scale value
- **4D tensors** (N, C, H, W): For 2D spatial upsampling, requires 2 size/scale values  
- **5D tensors** (N, C, D, H, W): For 3D spatial upsampling, requires 3 size/scale values

Where N=batch size, C=channels, L=length, H=height, W=width, D=depth.

## Data Type Support

- **Supported**: float32, float64
- **Not supported**: Integer types (int32, int64, etc.)

## Return Value

Returns a tensor handle containing the upsampled tensor.

## Examples

### Basic 1D Upsampling (3D tensor)
```tcl
# Create a 3D tensor (N=1, C=1, L=4) 
set input_data [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
set input [torch::tensor_reshape $input_data {1 1 4}]

# Positional syntax
set result1 [torch::upsample_nearest $input {8}]

# Named syntax
set result2 [torch::upsample_nearest -input $input -size {8}]

# Scale factor
set result3 [torch::upsample_nearest -input $input -scale_factor {2.0}]

# CamelCase alias
set result4 [torch::upsampleNearest $input {8}]
```

### 2D Image Upsampling (4D tensor)
```tcl
# Create a 4D tensor (N=1, C=1, H=2, W=2)
set image_data [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
set image [torch::tensor_reshape $image_data {1 1 2 2}]

# Upsample to 4x4
set upsampled [torch::upsample_nearest -input $image -size {4 4}]

# With scale factor
set upsampled2 [torch::upsample_nearest -input $image -scale_factor {2.0 2.0}]
```

### 3D Volume Upsampling (5D tensor)
```tcl
# Create a 5D tensor (N=1, C=1, D=2, H=2, W=2)
set volume_data [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu false]
set volume [torch::tensor_reshape $volume_data {1 1 2 2 2}]

# Upsample to 4x4x4
set upsampled [torch::upsample_nearest -input $volume -size {4 4 4}]
```

## Mathematical Details

Nearest neighbor upsampling replicates the nearest pixel values when increasing the spatial dimensions. For each output position, it finds the closest input position and copies that value.

**Size-based upsampling**:
- Output size is explicitly specified
- Input: (N, C, L_in) → Output: (N, C, L_out) where L_out is specified

**Scale factor-based upsampling**:
- Output size = Input size × scale_factor
- Input: (N, C, L_in) → Output: (N, C, L_in × scale) where scale is the factor

## Error Handling

The command will return an error in the following cases:

- **Invalid tensor handle**: Tensor does not exist
- **Missing parameters**: Required -input and either -size or -scale_factor
- **Invalid parameter values**: Non-numeric values in size/scale_factor lists
- **Unsupported data types**: Integer tensors
- **Dimension mismatch**: Size/scale_factor length doesn't match spatial dimensions

## Migration Guide

### From Positional to Named Syntax

**Old (Positional)**:
```tcl
set result [torch::upsample_nearest $input {8 8}]
```

**New (Named)**:
```tcl
set result [torch::upsample_nearest -input $input -size {8 8}]
```

### Using CamelCase
```tcl
# Both equivalent
set result1 [torch::upsample_nearest -input $input -size {8}]
set result2 [torch::upsampleNearest -input $input -size {8}]
```

## Performance Notes

- Nearest neighbor interpolation is computationally efficient
- Memory usage increases proportionally with the output size
- Supports GPU acceleration when tensors are on GPU

## See Also

- `torch::upsample_bilinear` - Bilinear interpolation upsampling
- `torch::interpolate` - General interpolation with multiple modes
- `torch::tensor_reshape` - Reshape tensors for proper dimensionality
- `torch::tensor_create` - Create tensors with specific data 