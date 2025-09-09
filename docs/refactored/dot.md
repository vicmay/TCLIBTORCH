# torch::dot

Computes the dot product (scalar product) of two 1-D tensors.

## Syntax

### Current Syntax
```tcl
torch::dot input other
```

### Named Parameter Syntax  
```tcl
torch::dot -input tensor -other tensor
```

Both syntaxes are fully supported and equivalent.

## Parameters

### Named Parameters
- `-input` (required): First input tensor name
- `-other` (required): Second input tensor name

### Positional Parameters
1. `input` (required): First input tensor name
2. `other` (required): Second input tensor name

## Description

The `torch::dot` function computes the dot product (also called scalar product or inner product) of two 1-D tensors. The dot product is a fundamental operation in linear algebra that produces a scalar value representing the "similarity" or projection of one vector onto another.

## Mathematical Details

For two vectors **a** = [a₁, a₂, ..., aₙ] and **b** = [b₁, b₂, ..., bₙ]:
- **a** · **b** = a₁b₁ + a₂b₂ + ... + aₙbₙ = Σᵢ aᵢbᵢ

Key properties:
- **Commutative**: **a** · **b** = **b** · **a**
- **Distributive**: **a** · (**b** + **c**) = **a** · **b** + **a** · **c**
- **Associative** (with scalars): (k**a**) · **b** = k(**a** · **b**) = **a** · (k**b**)
- **Geometric interpretation**: **a** · **b** = |**a**| |**b**| cos(θ) where θ is the angle between vectors
- **Orthogonality**: **a** · **b** = 0 if and only if **a** ⊥ **b**

## Examples

### Basic Usage

#### Positional Syntax
```tcl
# Basic dot product of two vectors
set v1 [torch::tensor_create {1.0 2.0 3.0} -dtype float32]
set v2 [torch::tensor_create {4.0 5.0 6.0} -dtype float32]
set result [torch::dot $v1 $v2]
# Result: 32.0 (1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32)
```

#### Named Parameter Syntax
```tcl
# Same operation using named parameters
set v1 [torch::tensor_create {1.0 2.0 3.0} -dtype float32]
set v2 [torch::tensor_create {4.0 5.0 6.0} -dtype float32]
set result [torch::dot -input $v1 -other $v2]
```

#### Parameter Order Independence
```tcl
# Named parameters can be in any order
set v1 [torch::tensor_create {1.0 2.0 3.0} -dtype float32]
set v2 [torch::tensor_create {4.0 5.0 6.0} -dtype float32]
set result [torch::dot -other $v2 -input $v1]
```

### Mathematical Properties Demonstration

```tcl
# Commutativity: a · b = b · a
set a [torch::tensor_create {1.0 2.0 3.0} -dtype float32]
set b [torch::tensor_create {4.0 5.0 6.0} -dtype float32]
set dot_ab [torch::dot $a $b]
set dot_ba [torch::dot $b $a]
# dot_ab and dot_ba should be equal

# Zero dot product indicates orthogonality
set i [torch::tensor_create {1.0 0.0 0.0} -dtype float32]
set j [torch::tensor_create {0.0 1.0 0.0} -dtype float32]
set orthogonal [torch::dot -input $i -other $j]
# Result: 0.0 (orthogonal vectors)

# Self dot product gives squared magnitude
set v [torch::tensor_create {3.0 4.0} -dtype float32]
set magnitude_squared [torch::dot $v $v]
# Result: 25.0 (3² + 4² = 9 + 16 = 25)
```

### Working with Different Vector Sizes

```tcl
# Single element vectors
set scalar1 [torch::tensor_create {5.0} -dtype float32]
set scalar2 [torch::tensor_create {3.0} -dtype float32]
set result [torch::dot -input $scalar1 -other $scalar2]
# Result: 15.0

# Large vectors
set large_v1 [torch::ones {1000}]
set large_v2 [torch::ones {1000}]
set sum_result [torch::dot $large_v1 $large_v2]
# Result: 1000.0 (sum of 1000 ones)

# Zero vectors
set zero1 [torch::zeros {5}]
set any_vector [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} -dtype float32]
set zero_result [torch::dot -input $zero1 -other $any_vector]
# Result: 0.0
```

### Similarity and Distance Calculations

```tcl
# Vector similarity (normalized dot product)
set a [torch::tensor_create {1.0 2.0 3.0} -dtype float32]
set b [torch::tensor_create {2.0 3.0 4.0} -dtype float32]

# Calculate dot product
set dot_ab [torch::dot $a $b]

# Calculate magnitudes (using self dot product and sqrt)
set mag_a_sq [torch::dot $a $a]
set mag_b_sq [torch::dot $b $b]
# Note: In practice, you'd use torch::sqrt for actual magnitude

# Cosine similarity would be: dot_ab / (sqrt(mag_a_sq) * sqrt(mag_b_sq))
```

### Physics and Engineering Applications

```tcl
# Work calculation: W = F · d
set force [torch::tensor_create {10.0 5.0 0.0} -dtype float32]
set displacement [torch::tensor_create {2.0 3.0 0.0} -dtype float32]
set work [torch::dot -input $force -other $displacement]
# Result: 35.0 Joules

# Power calculation: P = F · v
set force [torch::tensor_create {100.0 0.0 0.0} -dtype float32]
set velocity [torch::tensor_create {5.0 0.0 0.0} -dtype float32]
set power [torch::dot -input $force -other $velocity]
# Result: 500.0 Watts

# Projection calculation component
set vector [torch::tensor_create {3.0 4.0 0.0} -dtype float32]
set unit_x [torch::tensor_create {1.0 0.0 0.0} -dtype float32]
set x_component [torch::dot $vector $unit_x]
# Result: 3.0 (x-component of vector)
```

### Machine Learning Applications

```tcl
# Feature similarity in ML
set feature1 [torch::tensor_create {0.8 0.6 0.3 0.9} -dtype float32]
set feature2 [torch::tensor_create {0.7 0.5 0.4 0.8} -dtype float32]
set similarity [torch::dot -input $feature1 -other $feature2]

# Linear layer computation (simplified)
set input [torch::tensor_create {1.0 0.5 2.0} -dtype float32]
set weights [torch::tensor_create {0.3 0.7 0.1} -dtype float32]
set linear_output [torch::dot $input $weights]
# This would be one neuron's output before bias and activation

# Attention mechanism component
set query [torch::tensor_create {0.2 0.8 0.3} -dtype float32]
set key [torch::tensor_create {0.1 0.9 0.2} -dtype float32]
set attention_score [torch::dot -input $query -other $key]
```

### Statistical Applications

```tcl
# Correlation component (before normalization)
set x_centered [torch::tensor_create {-1.0 0.0 1.0} -dtype float32]
set y_centered [torch::tensor_create {-2.0 0.0 2.0} -dtype float32]
set covariance_component [torch::dot $x_centered $y_centered]

# Weighted sum
set values [torch::tensor_create {10.0 20.0 30.0} -dtype float32]
set weights [torch::tensor_create {0.2 0.3 0.5} -dtype float32]
set weighted_sum [torch::dot -input $values -other $weights]
# Result: 23.0 (weighted average * sum of weights)
```

## Input Requirements

- **Input Tensors**: Both tensors must be 1-D (vectors)
- **Shape**: Both tensors must have the same length
- **Data Type**: Numerical tensors (int, float32, float64, etc.)
- **Device**: CPU and CUDA tensors supported (both tensors must be on the same device)

## Output

Returns a scalar tensor containing the dot product value. The output tensor:
- Has shape `{}` (scalar)
- Has the same device as the input tensors
- Has dtype determined by PyTorch's type promotion rules

## Error Handling

The function will raise an error if:
- Input tensor names are invalid or don't exist
- Required parameters are missing
- Unknown parameters are provided
- Missing parameter values
- Input tensors are not 1-D
- Input tensors have different shapes
- Input tensors are on different devices

## Performance Considerations

- **Optimized Implementation**: Uses PyTorch's highly optimized native dot product
- **GPU Acceleration**: Full CUDA support for GPU tensors
- **Memory Efficient**: Minimal memory overhead, returns single scalar
- **Computational Complexity**: O(n) where n is the vector length
- **SIMD Optimization**: Vectorized operations on CPU
- **Parallelization**: Automatic parallelization for large vectors

## Common Use Cases

1. **Machine Learning**: Feature similarity, attention mechanisms, linear layers
2. **Computer Graphics**: Vector projections, lighting calculations
3. **Physics Simulations**: Work, power, and energy calculations
4. **Signal Processing**: Correlation, convolution components
5. **Statistics**: Covariance calculation, weighted sums
6. **Optimization**: Gradient dot products, line search
7. **Geometry**: Angle calculations, orthogonality testing
8. **Data Science**: Similarity metrics, dimensionality reduction

## Related Functions

- `torch::cross` - Cross product of 3D vectors
- `torch::outer` - Outer product of vectors
- `torch::tensor_matmul` - Matrix multiplication
- `torch::tensor_bmm` - Batch matrix multiplication
- `torch::linalg_norm` - Vector/matrix norms
- `torch::cosine_similarity` - Normalized dot product for similarity

## Migration Guide

### From Positional to Named Parameters

```tcl
# OLD: Positional syntax
set result [torch::dot $vector1 $vector2]

# NEW: Named parameter syntax
set result [torch::dot -input $vector1 -other $vector2]

# Both syntaxes work identically
set old_result [torch::dot $v1 $v2]
set new_result [torch::dot -input $v1 -other $v2]
# old_result and new_result are identical
```

### Error Migration

```tcl
# OLD: Less descriptive errors
# torch::dot $invalid_tensor $v2  # Generic error

# NEW: More descriptive errors
# torch::dot -input $invalid_tensor -other $v2  # "Invalid input tensor"
# torch::dot -input $v1 -unknown $v2            # "Unknown parameter: -unknown"
# torch::dot -input $v1                         # "Required parameters missing"
```

## Version History

- **Original**: Positional syntax only
- **Current**: Dual syntax support (positional + named parameters)
- **Features**: Enhanced error messages, parameter validation
- **Compatibility**: 100% backward compatible

## See Also

- [Linear Algebra Operations](../linear_algebra.md)
- [Tensor Operations Guide](../tensors.md)
- [Performance Optimization](../performance.md)
- [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.dot.html) 