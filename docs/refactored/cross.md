# torch::cross

Computes the cross product of two tensors along a specified dimension.

## Syntax

### Current Syntax
```tcl
torch::cross input other ?dim?
```

### Named Parameter Syntax  
```tcl
torch::cross -input tensor -other tensor ?-dim int?
torch::cross -tensor tensor -other tensor ?-dim int?
```

Both syntaxes are fully supported and equivalent.

## Parameters

### Named Parameters
- `-input` (required): First input tensor name
- `-tensor` (required): First input tensor name (alias for `-input`)
- `-other` (required): Second input tensor name
- `-dim` (optional): Dimension along which to compute cross product (integer)

### Positional Parameters
1. `input` (required): First input tensor name
2. `other` (required): Second input tensor name
3. `dim` (optional): Dimension along which to compute cross product (integer)

## Description

The `torch::cross` function computes the cross product of two tensors. Cross products are typically computed between 3D vectors, but this function can handle tensors of various shapes and compute cross products along different dimensions.

## Mathematical Details

For two 3D vectors **a** = [a₁, a₂, a₃] and **b** = [b₁, b₂, b₃]:
- **a** × **b** = [a₂b₃ - a₃b₂, a₃b₁ - a₁b₃, a₁b₂ - a₂b₁]

Key properties:
- **Anti-commutative**: **a** × **b** = -(**b** × **a**)
- **Orthogonal**: **a** × **b** is orthogonal to both **a** and **b**
- **Magnitude**: |**a** × **b**| = |**a**| |**b**| sin(θ) where θ is the angle between vectors
- **Right-hand rule**: Direction follows the right-hand rule

## Examples

### Basic Usage

#### Positional Syntax
```tcl
# Basic cross product of two 3D vectors
set v1 [torch::tensor_create {1.0 0.0 0.0} -dtype float32]  # i vector
set v2 [torch::tensor_create {0.0 1.0 0.0} -dtype float32]  # j vector
set result [torch::cross $v1 $v2]
# Result: {0.0 0.0 1.0} (k vector)
```

#### Named Parameter Syntax
```tcl
# Same operation using named parameters
set v1 [torch::tensor_create {1.0 0.0 0.0} -dtype float32]
set v2 [torch::tensor_create {0.0 1.0 0.0} -dtype float32]
set result [torch::cross -input $v1 -other $v2]
```

#### Using -tensor alias
```tcl
# Alternative named parameter
set v1 [torch::tensor_create {1.0 2.0 3.0} -dtype float32]
set v2 [torch::tensor_create {4.0 5.0 6.0} -dtype float32]
set result [torch::cross -tensor $v1 -other $v2]
```

### Working with Different Tensor Shapes

```tcl
# Cross product of multiple vector pairs (2D tensor)
set vectors1 [torch::tensor_create {{1.0 0.0 0.0} {0.0 1.0 0.0}} -dtype float32]
set vectors2 [torch::tensor_create {{0.0 1.0 0.0} {0.0 0.0 1.0}} -dtype float32]
set result [torch::cross -input $vectors1 -other $vectors2]
# Result shape: [2, 3] - cross product for each row

# Cross product with batch dimensions
set batch1 [torch::tensor_create {{{1.0 0.0 0.0}} {{0.0 1.0 0.0}}} -dtype float32]
set batch2 [torch::tensor_create {{{0.0 1.0 0.0}} {{0.0 0.0 1.0}}} -dtype float32]
set result [torch::cross -input $batch1 -other $batch2]
# Result shape: [2, 1, 3]
```

### Using the Dimension Parameter

```tcl
# Specify explicit dimension for cross product
set matrix1 [torch::tensor_create {{1.0 0.0 0.0} {0.0 1.0 0.0} {0.0 0.0 1.0}} -dtype float32]
set matrix2 [torch::tensor_create {{0.0 1.0 0.0} {0.0 0.0 1.0} {1.0 0.0 0.0}} -dtype float32]

# Cross product along dimension 0 (columns)
set result0 [torch::cross -input $matrix1 -other $matrix2 -dim 0]

# Cross product along dimension 1 (rows)
set result1 [torch::cross -input $matrix1 -other $matrix2 -dim 1]

# Positional syntax with dimension
set result [torch::cross $matrix1 $matrix2 1]
```

### Mathematical Properties Demonstration

```tcl
# Anti-commutativity: a × b = -(b × a)
set a [torch::tensor_create {1.0 2.0 3.0} -dtype float32]
set b [torch::tensor_create {4.0 5.0 6.0} -dtype float32]
set cross_ab [torch::cross $a $b]
set cross_ba [torch::cross $b $a]
# cross_ab and cross_ba should be negatives of each other

# Standard basis vectors
set i [torch::tensor_create {1.0 0.0 0.0} -dtype float32]
set j [torch::tensor_create {0.0 1.0 0.0} -dtype float32]
set k [torch::tensor_create {0.0 0.0 1.0} -dtype float32]

set i_cross_j [torch::cross $i $j]  # Should equal k
set j_cross_k [torch::cross $j $k]  # Should equal i
set k_cross_i [torch::cross $k $i]  # Should equal j
```

### Physics and Engineering Applications

```tcl
# Angular momentum: L = r × p
set position [torch::tensor_create {1.0 2.0 0.0} -dtype float32]
set momentum [torch::tensor_create {0.0 0.0 3.0} -dtype float32]
set angular_momentum [torch::cross -input $position -other $momentum]

# Torque: τ = r × F
set lever_arm [torch::tensor_create {2.0 0.0 0.0} -dtype float32]
set force [torch::tensor_create {0.0 10.0 0.0} -dtype float32]
set torque [torch::cross -input $lever_arm -other $force]

# Magnetic force: F = q(v × B)
set velocity [torch::tensor_create {1.0 0.0 0.0} -dtype float32]
set magnetic_field [torch::tensor_create {0.0 0.0 1.0} -dtype float32]
set force_direction [torch::cross -input $velocity -other $magnetic_field]
```

## Input Requirements

- **Input Tensors**: Both tensors must have the same shape
- **Vector Size**: For cross product, the last dimension (or specified dimension) must have size 3 or 2
- **Data Type**: Floating-point tensors (float32, float64)
- **Shape**: Any tensor shape is supported as long as the cross product dimension requirements are met
- **Device**: CPU and CUDA tensors supported (both tensors must be on the same device)

## Output

Returns a new tensor with the same shape as the input tensors, containing the cross product values. The output tensor has the same device and dtype as the input tensors.

## Error Handling

The function will raise an error if:
- Input tensor names are invalid or don't exist
- Required parameters are missing
- Unknown parameters are provided
- Input tensors have incompatible shapes
- Input tensors are on different devices
- Dimension parameter is invalid
- Cross product dimension doesn't have size 2 or 3

## Performance Considerations

- The operation is element-wise and parallelizable
- GPU acceleration available for CUDA tensors
- Memory usage: Creates a new tensor, doesn't modify input tensors
- Computational complexity: O(n) where n is the number of elements
- Highly optimized using PyTorch's native cross product implementation

## Common Use Cases

1. **Computer Graphics**: Normal vector calculation, rotation operations
2. **Physics Simulations**: Angular momentum, torque, magnetic force calculations
3. **Robotics**: Joint rotations, end-effector calculations
4. **Computer Vision**: Camera pose estimation, epipolar geometry
5. **Machine Learning**: Attention mechanisms with geometric constraints
6. **Engineering**: Structural analysis, fluid dynamics

## Related Functions

- `torch::dot` - Dot product (scalar product) of tensors
- `torch::outer` - Outer product of tensors
- `torch::tensor_matmul` - Matrix multiplication
- `torch::tensor_bmm` - Batch matrix multiplication
- `torch::linalg_norm` - Vector/matrix norms

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax (still supported)
set result [torch::cross $input $other]
set result [torch::cross $input $other $dim]

# New named parameter syntax
set result [torch::cross -input $input -other $other]
set result [torch::cross -input $input -other $other -dim $dim]

# Alternative with -tensor alias
set result [torch::cross -tensor $input -other $other]

# All produce identical results
```

### Benefits of Named Parameters

1. **Clarity**: Parameter purpose is explicit
2. **Flexibility**: Parameters can be provided in any order  
3. **Extensibility**: Easy to add optional parameters in the future
4. **Consistency**: Matches modern TCL conventions
5. **Self-documenting**: Code is more readable and maintainable

## Mathematical Properties Reference

### Vector Cross Product Identities
- **Anti-commutativity**: **a** × **b** = -(**b** × **a**)
- **Distributivity**: **a** × (**b** + **c**) = **a** × **b** + **a** × **c**
- **Scalar multiplication**: (k**a**) × **b** = k(**a** × **b**)
- **Triple scalar product**: **a** · (**b** × **c**) = **b** · (**c** × **a**) = **c** · (**a** × **b**)

### Geometric Interpretation
- **Magnitude**: |**a** × **b**| equals the area of parallelogram formed by **a** and **b**
- **Direction**: Perpendicular to both **a** and **b** following right-hand rule
- **Zero result**: **a** × **b** = **0** if and only if **a** and **b** are parallel

### Standard Basis Vectors
- **i** × **j** = **k**
- **j** × **k** = **i**  
- **k** × **i** = **j**
- **j** × **i** = -**k**
- **k** × **j** = -**i**
- **i** × **k** = -**j**

## Technical Notes

- Implements PyTorch's `torch.cross()` function
- Preserves tensor properties (device, requires_grad, etc.)
- Supports automatic differentiation when `requires_grad=true`
- Thread-safe operation
- Uses highly optimized mathematical libraries (Intel MKL, cuBLAS)
- Supports both 2D and 3D cross products

## Version History

- Added dual syntax support in refactoring phase
- Original positional syntax maintained for backward compatibility
- Named parameter syntax added for modern TCL conventions
- Added `-tensor` parameter alias for enhanced flexibility
- Enhanced error handling and parameter validation

## Physical Interpretation

The cross product represents:
- **Rotation axis**: Direction of rotation axis for rotating from first to second vector
- **Angular velocity**: In rigid body mechanics
- **Magnetic force**: Direction of force on charged particle in magnetic field
- **Torque direction**: Axis of rotational force
- **Surface normal**: Perpendicular direction to a surface defined by two vectors

## Advanced Examples

### Normal Vector Calculation
```tcl
# Calculate normal to a triangle surface
set edge1 [torch::tensor_create {1.0 0.0 0.0} -dtype float32]
set edge2 [torch::tensor_create {0.0 1.0 0.0} -dtype float32]
set normal [torch::cross -input $edge1 -other $edge2]
# Result: surface normal vector
```

### Rodrigues' Rotation Formula Components
```tcl
# Components for rotating vector v around axis k by angle θ
set v [torch::tensor_create {1.0 0.0 0.0} -dtype float32]
set k [torch::tensor_create {0.0 0.0 1.0} -dtype float32]  # rotation axis
set k_cross_v [torch::cross -input $k -other $v]
# Used in: v_rot = v*cos(θ) + (k×v)*sin(θ) + k(k·v)(1-cos(θ))
```

### Batch Processing
```tcl
# Process multiple vector pairs simultaneously
set vectors_a [torch::tensor_create {{1.0 0.0 0.0} {0.0 1.0 0.0} {0.0 0.0 1.0}} -dtype float32]
set vectors_b [torch::tensor_create {{0.0 1.0 0.0} {0.0 0.0 1.0} {1.0 0.0 0.0}} -dtype float32]
set batch_cross [torch::cross -input $vectors_a -other $vectors_b -dim 1]
# Efficiently compute cross products for all vector pairs
``` 