# torch::instance_norm3d

Applies 3D Instance Normalization to a tensor. Instance normalization normalizes across the spatial dimensions for each instance (sample) and channel independently, making it particularly effective for video processing, 3D computer vision, volumetric data analysis, and temporal sequence modeling.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::instance_norm3d -input tensor ?-eps epsilon? ?-momentum momentum? ?-affine affine? ?-track_running_stats track_stats?
torch::instance_norm3d -tensor tensor ?-epsilon epsilon? ?-momentum momentum? ?-affine affine? ?-trackRunningStats track_stats?
```

### Positional Parameters (Legacy)
```tcl
torch::instance_norm3d tensor ?eps? ?momentum? ?affine? ?track_running_stats?
```

### CamelCase Alias
```tcl
torch::instanceNorm3d -input tensor ?-eps epsilon? ?-momentum momentum? ?-affine affine? ?-track_running_stats track_stats?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-input` or `-tensor` | tensor | Yes | - | Input tensor (5D: [N, C, D, H, W]) |
| `-eps` or `-epsilon` | float | No | 1e-5 | Small value for numerical stability |
| `-momentum` | float | No | 0.1 | Momentum for running statistics |
| `-affine` | bool | No | true | Whether to use learnable affine parameters |
| `-track_running_stats` or `-trackRunningStats` | bool | No | true | Whether to track running statistics |

## Returns

Returns a new tensor with the same shape as input, containing the instance-normalized values.

## Description

Instance normalization applies normalization over each spatial location independently for each sample and channel. For a 5D input tensor with shape [N, C, D, H, W]:

- **N**: Batch size
- **C**: Number of channels  
- **D**: Depth (temporal or spatial dimension)
- **H**: Height (spatial dimension)
- **W**: Width (spatial dimension)

The normalization is computed as:
```
output = (input - mean) / sqrt(var + eps)
```

Where `mean` and `var` are computed per instance and channel across the spatial dimensions (D×H×W).

### Key Features
- **Per-instance normalization**: Each sample in the batch is normalized independently
- **Per-channel normalization**: Each channel is normalized independently
- **Spatial-temporal normalization**: Statistics computed across all spatial dimensions (D×H×W)
- **Video processing**: Particularly effective for video analysis and generation
- **Volumetric data**: Ideal for 3D medical imaging and volumetric analysis

## Examples

### Basic Usage

```tcl
# Create a 5D tensor (batch=2, channels=3, depth=4, height=5, width=6)
set tensor [torch::ones -shape {2 3 4 5 6}]

# Apply instance normalization (named parameters)
set result [torch::instance_norm3d -input $tensor]
puts [torch::tensor_shape $result]
;# Output: {2 3 4 5 6}

# Using positional syntax (legacy)
set result2 [torch::instance_norm3d $tensor]
puts [torch::tensor_shape $result2]
;# Output: {2 3 4 5 6}

# Using camelCase alias
set result3 [torch::instanceNorm3d -input $tensor]
puts [torch::tensor_shape $result3]
;# Output: {2 3 4 5 6}
```

### Video Processing

```tcl
# RGB video frames (batch=4, channels=3, frames=8, height=32, width=32)
set rgb_video [torch::ones -shape {4 3 8 32 32}]
set norm_video [torch::instance_norm3d -input $rgb_video]
puts [torch::tensor_shape $norm_video]
;# Output: {4 3 8 32 32}

# Grayscale video frames (batch=8, channels=1, frames=16, height=28, width=28)
set gray_video [torch::ones -shape {8 1 16 28 28}]
set norm_gray [torch::instance_norm3d -input $gray_video]
puts [torch::tensor_shape $norm_gray]
;# Output: {8 1 16 28 28}

# High-resolution video (batch=2, channels=3, frames=16, height=224, width=224)
set hires_video [torch::ones -shape {2 3 16 224 224}]
set norm_hires [torch::instance_norm3d -input $hires_video]
puts [torch::tensor_shape $norm_hires]
;# Output: {2 3 16 224 224}
```

### Volumetric Data Processing

```tcl
# 3D medical scans (batch=2, channels=1, depth=64, height=64, width=64)
set medical_scans [torch::ones -shape {2 1 64 64 64}]
set norm_scans [torch::instance_norm3d -input $medical_scans]
puts [torch::tensor_shape $norm_scans]
;# Output: {2 1 64 64 64}

# Multi-modal medical data (batch=1, channels=4, depth=32, height=128, width=128)
set multimodal [torch::ones -shape {1 4 32 128 128}]
set norm_multimodal [torch::instance_norm3d -input $multimodal]
puts [torch::tensor_shape $norm_multimodal]
;# Output: {1 4 32 128 128}
```

### Custom Parameters

```tcl
# Create input tensor
set tensor [torch::ones -shape {4 8 16 32 64}]

# Custom epsilon for numerical stability
set result [torch::instance_norm3d -input $tensor -eps 1e-4]

# Custom momentum for running statistics
set result2 [torch::instance_norm3d -input $tensor -momentum 0.2]

# Disable affine transformation
set result3 [torch::instance_norm3d -input $tensor -affine 0]

# Disable running statistics tracking
set result4 [torch::instance_norm3d -input $tensor -track_running_stats 0]
```

### Complete Configuration

```tcl
# Full parameter specification
set tensor [torch::ones -shape {2 4 8 16 32}]
set result [torch::instance_norm3d \
    -input $tensor \
    -eps 1e-4 \
    -momentum 0.15 \
    -affine 1 \
    -track_running_stats 1]
```

### 3D Convolutional Networks

```tcl
# 3D CNN feature maps (batch=2, channels=64, depth=8, height=14, width=14)
set feature_maps [torch::ones -shape {2 64 8 14 14}]
set normalized [torch::instance_norm3d -input $feature_maps -eps 1e-5]

# Video classification network
set video_features [torch::ones -shape {4 128 4 7 7}]
set norm_features [torch::instance_norm3d -input $video_features]
puts "Normalized features: [torch::tensor_shape $norm_features]"
```

### Video Generation Example

```tcl
# Video generation with instance normalization
set generated_frames [torch::ones -shape {1 3 32 64 64}]  ;# Generated video frames

# Normalize for consistent style across frames
set norm_frames [torch::instance_norm3d -input $generated_frames -eps 1e-5]

# Instance norm helps maintain temporal consistency in video generation
puts "Generated video: [torch::tensor_shape $norm_frames]"
```

## Parameter Aliases

The function supports multiple parameter names for flexibility:

- `-input` and `-tensor`: Both specify the input tensor
- `-eps` and `-epsilon`: Both specify the epsilon value
- `-track_running_stats` and `-trackRunningStats`: Both specify running statistics tracking

## Error Handling

The function provides clear error messages for invalid inputs:

```tcl
# Missing input tensor
catch {torch::instance_norm3d} msg
puts $msg
;# Output: Required parameters missing: input tensor required

# Invalid epsilon (must be positive)
set tensor [torch::ones -shape {2 3 4 5 6}]
catch {torch::instance_norm3d -input $tensor -eps -1.0} msg
puts $msg
;# Output: Invalid eps: must be positive number

# Invalid momentum (must be non-negative)
catch {torch::instance_norm3d -input $tensor -momentum -0.1} msg
puts $msg
;# Output: Invalid momentum: must be number >= 0

# Unknown parameter
catch {torch::instance_norm3d -input $tensor -unknown_param value} msg
puts $msg
;# Output: Unknown parameter: -unknown_param

# Unsupported tensor type
set int_tensor [torch::ones -shape {2 3 4 5 6} -dtype int64]
catch {torch::instance_norm3d $int_tensor} msg
puts $msg
;# Output: Error in instance_norm3d: "batch_norm" not implemented for 'Long'
```

## Technical Details

### Input Requirements
- **Shape**: Input must be a 5D tensor with shape [N, C, D, H, W]
- **N**: Batch size (number of samples)
- **C**: Number of channels
- **D**: Depth (temporal frames or spatial depth)
- **H**: Height (spatial dimension)
- **W**: Width (spatial dimension)

### Computation
1. **Per-instance/channel statistics**: Mean and variance computed independently for each (batch, channel) pair
2. **Spatial-temporal reduction**: Statistics computed across D×H×W dimensions
3. **Normalization**: `(x - mean) / sqrt(var + eps)`
4. **Affine transformation**: Optional learnable scale and shift parameters
5. **Running statistics**: Optional tracking for inference

### Performance Considerations
- **Memory**: O(N×C×D×H×W) for input tensor
- **Computation**: O(N×C×D×H×W) for normalization
- **Parallelization**: Highly parallelizable across batch and channel dimensions
- **Efficiency**: More efficient than batch norm for small batch sizes
- **Temporal consistency**: Maintains consistency across temporal dimensions

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
torch::instance_norm3d $tensor
torch::instance_norm3d $tensor $eps
torch::instance_norm3d $tensor $eps $momentum
torch::instance_norm3d $tensor $eps $momentum $affine $track_stats

# New named parameter syntax
torch::instance_norm3d -input $tensor
torch::instance_norm3d -input $tensor -eps $eps
torch::instance_norm3d -input $tensor -eps $eps -momentum $momentum
torch::instance_norm3d -input $tensor -eps $eps -momentum $momentum -affine $affine -track_running_stats $track_stats
```

### Advantages of Named Parameters
- **Clarity**: Parameter names make the code self-documenting
- **Flexibility**: Parameters can be specified in any order
- **Partial specification**: Can specify only needed parameters
- **Error prevention**: Less likely to pass arguments in wrong order
- **Extensibility**: Easy to add new parameters without breaking existing code

## Related Functions

- `torch::instance_norm1d`: 1D instance normalization
- `torch::instance_norm2d`: 2D instance normalization
- `torch::batch_norm3d`: Batch normalization (different normalization strategy)
- `torch::layer_norm`: Layer normalization
- `torch::group_norm`: Group normalization

## Use Cases

### Video Analysis and Generation
Instance normalization is particularly effective for video processing:
```tcl
# Video style transfer
set video_frames [torch::ones -shape {1 3 16 256 256}]
set normalized [torch::instance_norm3d -input $video_frames -eps 1e-5]

# Video generation networks
set generated_video [torch::instance_norm3d -input $conv3d_output]
```

### 3D Medical Imaging
Ideal for volumetric medical data processing:
```tcl
# Normalize 3D medical scans
set ct_scan [torch::ones -shape {1 1 128 128 128}]
set normalized_scan [torch::instance_norm3d -input $ct_scan]

# Multi-modal medical imaging (T1, T2, FLAIR, etc.)
set multimodal_mri [torch::ones -shape {1 4 64 64 64}]
set norm_mri [torch::instance_norm3d -input $multimodal_mri]
```

### 3D Computer Vision
For 3D object detection and segmentation:
```tcl
# 3D point cloud processing
set voxel_grid [torch::ones -shape {4 32 32 32 32}]
set normalized_voxels [torch::instance_norm3d -input $voxel_grid]

# 3D scene understanding
set scene_features [torch::instance_norm3d -input $conv3d_features]
```

### Temporal Sequence Modeling
For processing temporal sequences with spatial structure:
```tcl
# Action recognition in videos
set action_features [torch::ones -shape {8 256 8 7 7}]
set norm_actions [torch::instance_norm3d -input $action_features]

# Temporal consistency in video processing
set temporal_data [torch::instance_norm3d -input $sequence_data -eps 1e-5]
```

## Mathematical Background

Instance normalization computes statistics independently for each instance and channel:

- **Mean**: `μ_nc = (1/DHW) Σ_dhw x_ncdhw`
- **Variance**: `σ²_nc = (1/DHW) Σ_dhw (x_ncdhw - μ_nc)²`
- **Normalization**: `y_ncdhw = (x_ncdhw - μ_nc) / √(σ²_nc + ε)`

This differs from:
- **Batch Norm**: Statistics computed across batch dimension
- **Layer Norm**: Statistics computed across channel and spatial dimensions
- **Group Norm**: Statistics computed across channel groups

### Advantages for 3D Data
- **Temporal consistency**: Maintains consistency across time/depth dimensions
- **Volume independence**: Each volume is normalized independently
- **Real-time processing**: No dependency on batch statistics
- **Stable training**: More stable for 3D generative models
- **Memory efficiency**: Better memory usage for large 3D volumes

## Applications

### Video Processing
- **Video style transfer**: Consistent style across frames
- **Video generation**: GANs and autoencoders for video synthesis
- **Action recognition**: Feature normalization for temporal models
- **Video super-resolution**: Upsampling with temporal consistency

### Medical Imaging
- **3D segmentation**: Organ and tumor segmentation in volumetric data
- **Medical image synthesis**: Generating synthetic medical volumes
- **Multi-modal fusion**: Combining different imaging modalities
- **Anomaly detection**: Detecting abnormalities in 3D scans

### 3D Computer Vision
- **3D object detection**: Normalizing voxel-based representations
- **Point cloud processing**: Converting to voxel grids for normalization
- **3D scene understanding**: Feature normalization for 3D CNNs
- **Autonomous driving**: Processing LiDAR and 3D sensor data

## See Also

- [torch::instance_norm1d](instance_norm1d.md) - 1D instance normalization
- [torch::instance_norm2d](instance_norm2d.md) - 2D instance normalization
- [torch::batch_norm3d](batch_norm3d.md) - Batch normalization
- [Normalization Guide](../guides/normalization.md) - Complete normalization comparison 