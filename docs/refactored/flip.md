# torch::flip

Reverses the order of elements along specified dimensions of a tensor. This operation creates a new tensor with the specified dimensions flipped (reversed) while preserving all other dimensions.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::flip input dims
```

### Named Parameter Syntax
```tcl
torch::flip -input tensor -dims dimensions_list
torch::flip -tensor tensor -dimensions dimensions_list
```

### CamelCase Alias
```tcl
torch::Flip input dims
torch::Flip -input tensor -dims dimensions_list
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` / `-input` / `-tensor` | Tensor | Yes | Input tensor to flip |
| `dims` / `-dims` / `-dimensions` | List | Yes | List of dimension indices to flip |

## Returns

Returns a new tensor with the same shape and data type as the input tensor, but with the specified dimensions reversed.

## Mathematical Foundation

### Flip Operation

For a tensor **T** with dimensions `[d₀, d₁, ..., dₙ]`, flipping along dimension `k` produces a tensor **T'** where:

```
T'[i₀, i₁, ..., iₖ, ..., iₙ] = T[i₀, i₁, ..., (dₖ - 1 - iₖ), ..., iₙ]
```

### Multiple Dimension Flipping

When flipping multiple dimensions simultaneously, each specified dimension is reversed independently:

```
For dims = [k₁, k₂, ..., kₘ]:
T'[..., iₖ₁, ..., iₖ₂, ...] = T[..., (dₖ₁ - 1 - iₖ₁), ..., (dₖ₂ - 1 - iₖ₂), ...]
```

### Properties

1. **Involution**: Flipping twice returns the original tensor
   ```
   flip(flip(T, dims), dims) = T
   ```

2. **Dimension Independence**: Order of flipping multiple dimensions doesn't matter
   ```
   flip(T, [0, 1]) = flip(flip(T, [0]), [1]) = flip(flip(T, [1]), [0])
   ```

3. **Shape Preservation**: Input and output tensors have identical shapes
   ```
   shape(flip(T, dims)) = shape(T)
   ```

## Use Cases

### 1. Computer Vision
- **Image Flipping**: Horizontal/vertical flips for data augmentation
- **Mirror Effects**: Creating symmetric patterns
- **Orientation Correction**: Adjusting image orientation
- **Data Augmentation**: Expanding training datasets

### 2. Signal Processing
- **Time Reversal**: Reversing audio signals
- **Frequency Domain**: Manipulating spectrograms
- **Convolution**: Time-reversed filters
- **Echo Cancellation**: Signal preprocessing

### 3. Natural Language Processing
- **Sequence Reversal**: Bidirectional processing
- **Text Augmentation**: Creating reversed sentences
- **Attention Mechanisms**: Reverse attention patterns
- **Language Models**: Backward sequence processing

### 4. Scientific Computing
- **Boundary Conditions**: Symmetric boundary handling
- **Numerical Methods**: Grid reversals in PDE solving
- **Data Analysis**: Trend reversal analysis
- **Simulation**: Reversing simulation states

## Examples

### Basic Tensor Flipping

```tcl
# 1D tensor flip
set vector [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4}]
set flipped_vector [torch::flip $vector {0}]
# Result: {4.0 3.0 2.0 1.0}

# 2D tensor flip - rows
set matrix [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2}]
set flipped_rows [torch::flip $matrix {0}]
# Flips along first dimension (rows)

# 2D tensor flip - columns  
set flipped_cols [torch::flip $matrix {1}]
# Flips along second dimension (columns)
```

### Named Parameter Syntax

```tcl
# Basic named syntax
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3}]
set result [torch::flip -input $tensor -dims {0}]

# Alternative parameter names
set result [torch::flip -tensor $tensor -dimensions {1}]

# Parameter order independence
set result [torch::flip -dims {0 1} -input $tensor]
```

### Multiple Dimension Flipping

```tcl
# Flip multiple dimensions simultaneously
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 2 2}]

# Flip dimensions 0 and 1
set result [torch::flip $tensor {0 1}]

# Flip all dimensions
set result [torch::flip $tensor {0 1 2}]
```

### Computer Vision Applications

```tcl
# Image horizontal flip (data augmentation)
set image [torch::tensor_create -data $image_data -shape {3 224 224}]  # CHW format
set h_flipped [torch::flip $image {2}]  # Flip width dimension

# Image vertical flip
set v_flipped [torch::flip $image {1}]  # Flip height dimension

# Both horizontal and vertical flip (180-degree rotation)
set rotated [torch::flip $image {1 2}]
```

### Batch Processing

```tcl
# Flip batch of images
set batch [torch::tensor_create -data $batch_data -shape {32 3 64 64}]  # NCHW format

# Flip all images horizontally
set h_flipped_batch [torch::flip $batch {3}]  # Flip width

# Flip all images vertically
set v_flipped_batch [torch::flip $batch {2}]  # Flip height
```

### Signal Processing

```tcl
# Time series reversal
set signal [torch::tensor_create -data $audio_data -shape {1 44100}]  # 1 second of audio
set reversed_signal [torch::flip $signal {1}]

# Spectrogram manipulation
set spectrogram [torch::tensor_create -data $spec_data -shape {512 1000}]  # Freq x Time
set time_reversed [torch::flip $spectrogram {1}]  # Reverse time axis
set freq_reversed [torch::flip $spectrogram {0}]  # Reverse frequency axis
```

### NLP Applications

```tcl
# Reverse token sequences
set tokens [torch::tensor_create -data $token_ids -shape {32 128}]  # Batch x Sequence
set reversed_tokens [torch::flip $tokens {1}]  # Reverse sequence dimension

# Bidirectional processing preparation
set forward_seq $tokens
set backward_seq [torch::flip $tokens {1}]
```

### Data Augmentation Pipeline

```tcl
# Random horizontal flip for training
proc random_h_flip {image prob} {
    if {[expr {rand()}] < $prob} {
        return [torch::flip $image {2}]  # Assuming CHW format
    }
    return $image
}

# Apply to training batch
set augmented_batch {}
foreach image $training_batch {
    lappend augmented_batch [random_h_flip $image 0.5]
}
```

### Mathematical Operations

```tcl
# Create symmetric boundary conditions
set data [torch::tensor_create -data $boundary_data -shape {100}]
set padded_left [torch::flip [torch::tensor_slice $data 0 10] {0}]
set padded_right [torch::flip [torch::tensor_slice $data -10 -1] {0}]

# Convolution with time-reversed filter
set filter [torch::tensor_create -data $filter_data -shape {5}]
set time_reversed_filter [torch::flip $filter {0}]
```

### Advanced Multi-dimensional Operations

```tcl
# 3D volume flip operations
set volume [torch::tensor_create -data $volume_data -shape {64 64 64}]

# Flip along each axis independently
set x_flipped [torch::flip $volume {0}]
set y_flipped [torch::flip $volume {1}]  
set z_flipped [torch::flip $volume {2}]

# Flip along multiple axes
set xy_flipped [torch::flip $volume {0 1}]
set xyz_flipped [torch::flip $volume {0 1 2}]
```

## Performance Considerations

### Memory Usage
```tcl
# Flip creates a new tensor, so memory usage doubles temporarily
set large_tensor [torch::tensor_create -data $large_data -shape {1000 1000 1000}]
set flipped [torch::flip $large_tensor {0}]  # Requires additional 4GB for float32
```

### Optimization Tips

1. **Minimize Multiple Flips**: Combine dimension flips when possible
   ```tcl
   # Efficient: single operation
   set result [torch::flip $tensor {0 1 2}]
   
   # Less efficient: multiple operations
   set temp1 [torch::flip $tensor {0}]
   set temp2 [torch::flip $temp1 {1}]
   set result [torch::flip $temp2 {2}]
   ```

2. **Batch Processing**: Flip entire batches rather than individual items
   ```tcl
   # Efficient: flip entire batch
   set flipped_batch [torch::flip $batch {3}]
   
   # Less efficient: flip each item separately
   set flipped_batch {}
   foreach item $batch_items {
       lappend flipped_batch [torch::flip $item {2}]
   }
   ```

## Common Patterns

### Data Augmentation
```tcl
proc augment_image {image} {
    # 50% chance of horizontal flip
    if {[expr {rand()}] < 0.5} {
        set image [torch::flip $image {2}]
    }
    
    # 25% chance of vertical flip
    if {[expr {rand()}] < 0.25} {
        set image [torch::flip $image {1}]
    }
    
    return $image
}
```

### Symmetric Padding
```tcl
proc symmetric_pad {tensor dim pad_size} {
    # Get boundary slices
    set left_slice [torch::tensor_slice $tensor $dim 0 $pad_size]
    set right_slice [torch::tensor_slice $tensor $dim [expr -$pad_size] -1]
    
    # Flip for symmetric padding
    set left_pad [torch::flip $left_slice $dim]
    set right_pad [torch::flip $right_slice $dim]
    
    # Concatenate
    return [torch::cat [list $left_pad $tensor $right_pad] $dim]
}
```

### Time Series Processing
```tcl
proc create_bidirectional_sequence {tokens} {
    set forward $tokens
    set backward [torch::flip $tokens {1}]
    
    return [list $forward $backward]
}
```

## Error Handling

The command validates parameters and provides descriptive error messages:

```tcl
# Missing arguments
catch {torch::flip} result
# Error: Usage: torch::flip input dims | torch::flip -input tensor -dims list

# Invalid tensor
catch {torch::flip "invalid_tensor" {0}} result
# Error: Invalid input tensor

# Missing dims parameter
catch {torch::flip $tensor} result  
# Error: Usage: torch::flip input dims

# Unknown parameter
catch {torch::flip -input $tensor -invalid {0}} result
# Error: Unknown parameter: -invalid

# Empty dimensions list
catch {torch::flip -input $tensor -dims {}} result
# Error: Required parameters missing: -input and -dims
```

## Migration Guide

### From Positional to Named Syntax

```tcl
# Old positional syntax
set old_result [torch::flip $tensor {0 1}]

# New named syntax (equivalent)
set new_result [torch::flip -input $tensor -dims {0 1}]

# Alternative parameter names
set alt_result [torch::flip -tensor $tensor -dimensions {0 1}]
```

### Best Practices

1. **Use Named Parameters**: More readable and maintainable
2. **Combine Dimensions**: More efficient than multiple operations
3. **Validate Dimensions**: Ensure dimension indices are valid
4. **Consider Memory**: Large tensors require significant memory for flipping

## Mathematical Properties

### Involution Property
```tcl
set tensor [torch::tensor_create -data {1.0 2.0 3.0} -shape {3}]
set flipped [torch::flip $tensor {0}]
set double_flipped [torch::flip $flipped {0}]
# double_flipped equals original tensor
```

### Commutativity of Multiple Dimensions
```tcl
set tensor [torch::tensor_create -data $data -shape {4 4}]

# These are equivalent
set result1 [torch::flip $tensor {0 1}]
set temp [torch::flip $tensor {0}]
set result2 [torch::flip $temp {1}]
# result1 equals result2
```

## See Also

- [`torch::rot90`](rot90.md) - Rotate tensor by 90 degrees
- [`torch::roll`](roll.md) - Roll tensor elements along dimensions
- [`torch::transpose`](transpose.md) - Transpose tensor dimensions
- [`torch::permute`](permute.md) - Permute tensor dimensions
- [`torch::tensor_slice`](tensor_slice.md) - Extract tensor slices

## References

- [PyTorch flip Documentation](https://pytorch.org/docs/stable/generated/torch.flip.html)
- [NumPy flip Documentation](https://numpy.org/doc/stable/reference/generated/numpy.flip.html)
- [Data Augmentation Techniques](https://pytorch.org/vision/stable/transforms.html) 