# torch::roi_align

Performs Region of Interest (ROI) Align operation on feature maps. ROI Align is a pooling operation that extracts fixed-size features from variable-sized regions of interest, with precise spatial locations preserved through bilinear interpolation.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::roi_align -input tensor -boxes tensor -outputSize {height width} \
    ?-spatialScale double? ?-samplingRatio int? ?-aligned bool?
```

### Positional Syntax (Legacy)
```tcl
torch::roi_align input boxes output_size ?spatial_scale? ?sampling_ratio? ?aligned?
```

### CamelCase Alias
```tcl
torch::roiAlign ...  ;# Same syntax options as above
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input`/`-input` | tensor | Yes | - | Input feature map tensor (shape: [N, C, H, W]) |
| `boxes`/`-boxes` | tensor | Yes | - | ROI boxes tensor (shape: [K, 4] or [K, 5]) |
| `outputSize`/`output_size` | list | Yes | - | Output size [height, width] for each ROI |
| `spatialScale`/`spatial_scale` | double | No | 1.0 | Scale factor for input coordinates |
| `samplingRatio`/`sampling_ratio` | int | No | -1 | Number of sampling points (-1 for adaptive) |
| `aligned`/`-aligned` | bool | No | true | If true, use pixel alignment |

## Returns

Returns a tensor handle containing the pooled features for each ROI.

## Description

ROI Align is a crucial operation in object detection networks that extracts fixed-size feature maps from regions of interest (ROIs) in a feature map. Unlike ROI Pooling, ROI Align avoids quantization of ROI boundaries and uses bilinear interpolation to compute exact values of the input features at four regularly sampled locations in each ROI bin.

### Key Features

1. **Precise Spatial Locations**: Preserves exact spatial locations through bilinear interpolation
2. **No Quantization**: Avoids the quantization of ROI boundaries
3. **Configurable Sampling**: Supports both fixed and adaptive sampling points
4. **Alignment Options**: Optional pixel alignment for better precision

### Mathematical Operation

For each ROI:
1. Scale ROI coordinates by `spatialScale`
2. Divide ROI into bins according to `outputSize`
3. Sample points in each bin (number determined by `samplingRatio`)
4. Compute feature values using bilinear interpolation
5. Average the sampled values in each bin

## Examples

### Basic Usage - Named Parameters
```tcl
# Create input feature map (batch_size=1, channels=64, height=32, width=32)
set features [torch::zeros {1 64 32 32}]

# Create ROI boxes (2 boxes, format: [x1, y1, x2, y2])
set boxes [torch::tensor_create {{0.0 0.0 0.5 0.5} {0.2 0.3 0.7 0.8}} float32]

# Apply ROI Align
set pooled_features [torch::roi_align \
    -input $features \
    -boxes $boxes \
    -outputSize {7 7}]
```

### Advanced Configuration
```tcl
# ROI Align with custom parameters
set result [torch::roi_align \
    -input $features \
    -boxes $boxes \
    -outputSize {14 14} \
    -spatialScale 0.0625 \
    -samplingRatio 2 \
    -aligned true]
```

### Legacy Positional Syntax
```tcl
# Same operation using positional syntax
set result [torch::roi_align $features $boxes {7 7} 0.0625 2 true]
```

### CamelCase Alias
```tcl
# Using camelCase alias
set result [torch::roiAlign \
    -input $features \
    -boxes $boxes \
    -outputSize {7 7} \
    -spatialScale 0.0625]
```

## Common Use Cases

1. **Object Detection**: Extract features for detected objects
   ```tcl
   # Process detection boxes
   set features [torch::roi_align \
       -input $backbone_features \
       -boxes $detection_boxes \
       -outputSize {7 7} \
       -spatialScale 0.0625]
   ```

2. **Instance Segmentation**: Generate mask predictions
   ```tcl
   # Higher resolution for mask prediction
   set mask_features [torch::roi_align \
       -input $features \
       -boxes $boxes \
       -outputSize {14 14} \
       -samplingRatio 2]
   ```

3. **Keypoint Detection**: Extract features for keypoint prediction
   ```tcl
   # Fine spatial resolution for keypoints
   set keypoint_features [torch::roi_align \
       -input $features \
       -boxes $person_boxes \
       -outputSize {28 28} \
       -aligned true]
   ```

## Error Handling

The function validates all parameters and provides descriptive error messages:

```tcl
# Missing required parameters
catch {torch::roi_align -input $features} error
puts $error  ;# Missing required parameters

# Invalid input tensor
catch {torch::roi_align -input "invalid" -boxes $boxes -outputSize {7 7}} error
puts $error  ;# Invalid input tensor

# Invalid spatial scale
catch {torch::roi_align -input $features -boxes $boxes -outputSize {7 7} -spatialScale "invalid"} error
puts $error  ;# Invalid spatialScale value
```

## Performance Considerations

1. **Spatial Scale**: Choose appropriate scale factor based on feature map size
2. **Sampling Points**: More sampling points (higher `samplingRatio`) gives better precision but slower computation
3. **Output Size**: Larger output sizes require more computation
4. **Batch Processing**: Process multiple ROIs in parallel for better efficiency

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set result [torch::roi_align $features $boxes {7 7} 0.0625 2 true]

# New named parameter syntax
set result [torch::roi_align \
    -input $features \
    -boxes $boxes \
    -outputSize {7 7} \
    -spatialScale 0.0625 \
    -samplingRatio 2 \
    -aligned true]
```

### Benefits of Named Parameters
- **Self-documenting**: Parameter names make code more readable
- **Optional Parameters**: Easy to specify only needed parameters
- **Order Independence**: Parameters can be specified in any order
- **Maintainability**: Easier to modify specific parameters

## See Also

- [`torch::roi_pool`](roi_pool.md) - ROI Pooling operation
- [`torch::adaptive_avg_pool2d`](adaptive_avg_pool2d.md) - Adaptive average pooling
- [`torch::adaptive_max_pool2d`](adaptive_max_pool2d.md) - Adaptive max pooling
- [`torch::interpolate`](interpolate.md) - General interpolation operations 