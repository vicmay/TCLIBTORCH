# torch::cond

## Description
Computes the condition number of a matrix. The condition number is a measure of how sensitive a matrix is to numerical errors and indicates how well-conditioned the matrix is for numerical computations. A lower condition number indicates better numerical stability.

## Syntax

### Original Syntax (Positional Parameters)
```tcl
torch::cond input ?p?
```

### New Syntax (Named Parameters)
```tcl
torch::cond -input tensor -p value
torch::cond -tensor tensor -norm value
```

## Parameters

### Positional Format
- **input**: Input square matrix tensor (required)
- **p**: Norm type (optional, default: 2-norm)
  - Numeric value (e.g., 1.0, 2.0)
  - String values: "fro" (Frobenius norm), "nuc" (nuclear norm)

### Named Parameter Format
- **-input/-tensor**: Input square matrix tensor (required)
- **-p/-norm**: Norm type (optional, default: 2-norm)
  - Numeric value or string ("fro", "nuc")

## Return Value
Returns a scalar tensor containing the condition number of the input matrix.

## Mathematical Background

The condition number measures the sensitivity of a linear system to changes in the input. For a matrix A, the condition number is defined as:

### 2-norm Condition Number (Default)
```
cond(A) = σ_max(A) / σ_min(A)
```
Where σ_max and σ_min are the largest and smallest singular values of A.

### Properties
- **Well-conditioned**: cond(A) ≈ 1 (good numerical stability)
- **Ill-conditioned**: cond(A) >> 1 (poor numerical stability)
- **Singular matrix**: cond(A) = ∞ (non-invertible)

### Norm Types
- **2-norm**: Based on singular value decomposition (most common)
- **1-norm**: Maximum absolute column sum
- **∞-norm**: Maximum absolute row sum
- **Frobenius norm**: Approximation using matrix norms

## Examples

### Basic Usage
```tcl
# Create a well-conditioned matrix (identity)
set matrix [torch::eye 3]
set cond_num [torch::cond $matrix]
# Result: condition number ≈ 1.0

# Create an ill-conditioned matrix
set ill_matrix [torch::tensor_create {1.0 1.0 1.0 1.000001} float32]
set ill_matrix [torch::tensor_reshape $ill_matrix {2 2}]
set cond_num [torch::cond $ill_matrix]
# Result: large condition number
```

### Specifying Norm Type
```tcl
# Using 2-norm (default)
set matrix [torch::tensor_create {4.0 1.0 1.0 3.0} float32]
set matrix [torch::tensor_reshape $matrix {2 2}]
set cond_2norm [torch::cond $matrix 2.0]

# Using Frobenius norm
set cond_fro [torch::cond $matrix "fro"]

# Using 1-norm
set cond_1norm [torch::cond $matrix 1.0]
```

### Named Parameter Syntax
```tcl
# Basic named parameter usage
set matrix [torch::tensor_create {2.0 1.0 1.0 1.0} float32]
set matrix [torch::tensor_reshape $matrix {2 2}]
set cond_num [torch::cond -input $matrix]

# Specifying norm with named parameters
set cond_num [torch::cond -input $matrix -p 2.0]
set cond_fro [torch::cond -tensor $matrix -norm "fro"]
```

### Parameter Order Independence
```tcl
# Named parameters can be in any order
set matrix [torch::tensor_create {3.0 1.0 1.0 2.0} float32]
set matrix [torch::tensor_reshape $matrix {2 2}]

set result1 [torch::cond -input $matrix -p 2.0]
set result2 [torch::cond -p 2.0 -input $matrix]
# Both produce identical results
```

## Advanced Examples

### Numerical Stability Analysis
```tcl
# Test matrix conditioning for numerical stability
proc analyze_matrix_stability {matrix_values} {
    set matrix [torch::tensor_create $matrix_values float32]
    set n [expr {int(sqrt([llength $matrix_values]))}]
    set matrix [torch::tensor_reshape $matrix [list $n $n]]
    
    set cond_num [torch::cond $matrix]
    
    if {$cond_num < 10} {
        puts "Well-conditioned matrix (cond = $cond_num)"
    } elseif {$cond_num < 1000} {
        puts "Moderately conditioned matrix (cond = $cond_num)"
    } else {
        puts "Ill-conditioned matrix (cond = $cond_num)"
    }
    
    return $cond_num
}

# Test different matrices
analyze_matrix_stability {1.0 0.0 0.0 1.0}              # Identity: well-conditioned
analyze_matrix_stability {10.0 0.0 0.0 1.0}             # Diagonal: moderate
analyze_matrix_stability {1.0 1.0 1.0 1.000001}         # Near-singular: ill-conditioned
```

### Linear System Solving Assessment
```tcl
# Assess numerical stability before solving linear systems
proc assess_system_stability {A_values b_values} {
    set A [torch::tensor_create $A_values float32]
    set n [expr {int(sqrt([llength $A_values]))}]
    set A [torch::tensor_reshape $A [list $n $n]]
    
    set b [torch::tensor_create $b_values float32]
    
    # Check condition number
    set cond_num [torch::cond $A]
    puts "System condition number: $cond_num"
    
    if {$cond_num > 1e12} {
        puts "⚠️  Warning: System is ill-conditioned!"
        puts "   Solutions may be unreliable due to numerical errors."
        return 0
    } else {
        puts "✅ System is well-conditioned for numerical solving."
        return 1
    }
}

# Example usage
set A_good {4.0 1.0 1.0 3.0}
set b_good {5.0 4.0}
assess_system_stability $A_good $b_good
```

### Matrix Regularization Guide
```tcl
# Determine if matrix needs regularization based on condition number
proc suggest_regularization {matrix_values} {
    set matrix [torch::tensor_create $matrix_values float32]
    set n [expr {int(sqrt([llength $matrix_values]))}]
    set matrix [torch::tensor_reshape $matrix [list $n $n]]
    
    set cond_num [torch::cond $matrix]
    puts "Original condition number: $cond_num"
    
    if {$cond_num > 1e6} {
        set lambda 1e-6
        puts "Suggestion: Add regularization with λ ≈ $lambda"
        puts "Regularized system: (A + λI)x = b"
        
        # Show effect of regularization
        set identity [torch::eye $n]
        set regularized [torch::tensor_add $matrix [torch::tensor_mul_scalar $identity $lambda]]
        set new_cond [torch::cond $regularized]
        puts "Regularized condition number: $new_cond"
    } else {
        puts "Matrix is well-conditioned; no regularization needed."
    }
}

# Example
suggest_regularization {1e6 1.0 1.0 1e-6}
```

### Performance Comparison of Norm Types
```tcl
# Compare computation time for different norm types
proc benchmark_condition_norms {matrix_values} {
    set matrix [torch::tensor_create $matrix_values float32]
    set n [expr {int(sqrt([llength $matrix_values]))}]
    set matrix [torch::tensor_reshape $matrix [list $n $n]]
    
    # Benchmark different norm types
    set norms {"" 1.0 2.0 "fro"}
    set names {"default" "1-norm" "2-norm" "Frobenius"}
    
    for {set i 0} {$i < [llength $norms]} {incr i} {
        set norm [lindex $norms $i]
        set name [lindex $names $i]
        
        set start [clock clicks -milliseconds]
        if {$norm eq ""} {
            set result [torch::cond $matrix]
        } else {
            set result [torch::cond $matrix $norm]
        }
        set end [clock clicks -milliseconds]
        set duration [expr {$end - $start}]
        
        puts "$name: ${duration}ms (result: $result)"
    }
}

# Create test matrix
set test_matrix {}
for {set i 0} {$i < 16} {incr i} {
    lappend test_matrix [expr {($i % 4) + 1.0 + 0.1 * $i}]
}
benchmark_condition_norms $test_matrix
```

## Practical Applications

### 1. Machine Learning - Model Stability
```tcl
# Check weight matrix conditioning in neural networks
proc check_weight_matrix_stability {weights} {
    set weight_tensor [torch::tensor_create $weights float32]
    # Assume square weight matrix for analysis
    set n [expr {int(sqrt([llength $weights]))}]
    set weight_tensor [torch::tensor_reshape $weight_tensor [list $n $n]]
    
    set cond_num [torch::cond $weight_tensor]
    
    if {$cond_num > 100} {
        puts "⚠️  Weight matrix is ill-conditioned (cond = $cond_num)"
        puts "   Consider weight initialization or regularization"
    } else {
        puts "✅ Weight matrix is well-conditioned (cond = $cond_num)"
    }
    
    return $cond_num
}
```

### 2. Scientific Computing - System Analysis
```tcl
# Analyze finite element or finite difference matrices
proc analyze_discretization_matrix {matrix_data} {
    set matrix [torch::tensor_create $matrix_data float32]
    set n [expr {int(sqrt([llength $matrix_data]))}]
    set matrix [torch::tensor_reshape $matrix [list $n $n]]
    
    set cond_num [torch::cond $matrix]
    
    puts "Discretization matrix analysis:"
    puts "  Matrix size: ${n}x${n}"
    puts "  Condition number: $cond_num"
    
    # Suggest grid refinement strategy
    if {$cond_num > 1e8} {
        puts "  ⚠️  Recommendation: Refine discretization"
        puts "     High condition number may cause convergence issues"
    }
}
```

### 3. Signal Processing - Filter Design
```tcl
# Assess digital filter stability
proc assess_filter_matrix {filter_coeffs} {
    set filter_matrix [torch::tensor_create $filter_coeffs float32]
    set n [expr {int(sqrt([llength $filter_coeffs]))}]
    set filter_matrix [torch::tensor_reshape $filter_matrix [list $n $n]]
    
    set cond_num [torch::cond $filter_matrix]
    
    if {$cond_num < 10} {
        puts "✅ Stable filter design (cond = $cond_num)"
    } else {
        puts "⚠️  Potentially unstable filter (cond = $cond_num)"
        puts "   Consider filter order reduction or coefficient adjustment"
    }
}
```

## Error Handling

### Invalid Input
```tcl
# Error: Invalid tensor
catch {torch::cond invalid_tensor} result
puts $result  # "Invalid input tensor"
```

### Missing Required Parameters
```tcl
# Error: No arguments
catch {torch::cond} result
puts $result  # Usage message
```

### Invalid Norm Parameter
```tcl
# Error: Invalid norm type
set matrix [torch::eye 2]
catch {torch::cond $matrix "invalid_norm"} result
puts $result  # "Invalid p parameter: must be a number or 'fro' or 'nuc'"
```

### Non-square Matrix
```tcl
# Condition number requires square matrices
set non_square [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
set non_square [torch::tensor_reshape $non_square {2 3}]
catch {torch::cond $non_square} result
puts $result  # Error from PyTorch about matrix dimensions
```

## Performance Considerations

### Computational Complexity
- **2-norm**: O(n³) due to SVD computation
- **1-norm and ∞-norm**: O(n³) due to matrix inversion
- **Frobenius norm**: O(n²) for norm computation + O(n³) for pseudoinverse

### Memory Usage
```tcl
# For large matrices, be aware of memory requirements
proc estimate_memory_usage {n} {
    set elements [expr {$n * $n}]
    set bytes_float32 [expr {$elements * 4}]
    set mb [expr {$bytes_float32 / 1024.0 / 1024.0}]
    
    puts "Matrix size: ${n}x${n}"
    puts "Memory requirement: ${mb:.2f} MB"
    
    if {$mb > 1000} {
        puts "⚠️  Large matrix: consider using iterative methods"
    }
}

estimate_memory_usage 1000  # 1000x1000 matrix
```

### Optimization Tips

1. **Choose appropriate norm**: Frobenius norm is often faster for approximation
2. **Batch processing**: Compute condition numbers for multiple matrices in parallel
3. **Early termination**: Use condition number thresholds to avoid expensive computations
4. **Regularization**: Add small diagonal terms to improve conditioning

## Condition Number Interpretation

### Guidelines
- **cond(A) = 1**: Perfect conditioning (orthogonal matrices)
- **cond(A) < 10**: Excellent conditioning
- **10 ≤ cond(A) < 100**: Good conditioning
- **100 ≤ cond(A) < 10⁴**: Moderate conditioning (acceptable for most applications)
- **10⁴ ≤ cond(A) < 10⁸**: Poor conditioning (use caution)
- **cond(A) ≥ 10⁸**: Very poor conditioning (numerical issues likely)

### Loss of Precision
For a matrix with condition number κ, expect to lose approximately log₁₀(κ) digits of precision in linear system solving.

## Migration Guide

### From Manual Condition Number Computation
```tcl
# Old manual approach (complex and error-prone)
# ... SVD computation, ratio calculation ...

# New approach using torch::cond
set matrix [torch::tensor_create $matrix_data float32]
set matrix [torch::tensor_reshape $matrix {n n}]
set cond_num [torch::cond $matrix]
```

### Syntax Modernization
```tcl
# Old style (still supported)
set cond_num [torch::cond $matrix 2.0]

# New style (recommended for clarity)
set cond_num [torch::cond -input $matrix -p 2.0]
```

## Related Commands
- [torch::svd](svd.md) - Singular value decomposition (used internally)
- [torch::pinverse](pinverse.md) - Moore-Penrose pseudoinverse
- [torch::matrix_rank](matrix_rank.md) - Matrix rank computation
- [torch::norm](norm.md) - Matrix and vector norms

## Mathematical References
- **Numerical Linear Algebra** by Trefethen and Bau
- **Matrix Computations** by Golub and Van Loan
- **Accuracy and Stability of Numerical Algorithms** by Higham

## See Also
- [torch::solve](solve.md) - Linear system solving
- [torch::lstsq](lstsq.md) - Least squares solutions
- [torch::matrix_norm](matrix_norm.md) - Various matrix norms 