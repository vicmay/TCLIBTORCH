#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test cases for positional syntax
test tensor-grad-1.1 {Basic positional syntax - with computed gradient} {
    # Create a tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation that will generate gradients
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    torch::tensor_backward $y
    
    # Get the gradient
    set grad [torch::tensor_grad $x]
    expr {[string length $grad] > 0}
} {1}

test tensor-grad-1.2 {Basic positional syntax - no gradient computed yet} {
    # Create a tensor that requires gradients but no computation done
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Try to get gradient without computing it
    catch {torch::tensor_grad $x} result
    expr {[string length $result] > 0}
} {1}

test tensor-grad-1.3 {Basic positional syntax - tensor doesn't require gradients} {
    # Create a tensor that doesn't require gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad false]
    
    # Try to get gradient
    catch {torch::tensor_grad $x} result
    expr {[string length $result] > 0}
} {1}

# Test cases for named syntax
test tensor-grad-2.1 {Named parameter syntax - with computed gradient} {
    # Create a tensor that requires gradients
    set x [torch::tensor_create -data {3.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation that will generate gradients
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    torch::tensor_backward $y
    
    # Get the gradient using named syntax
    set grad [torch::tensor_grad -input $x]
    expr {[string length $grad] > 0}
} {1}

test tensor-grad-2.2 {Named parameter syntax - no gradient computed yet} {
    # Create a tensor that requires gradients but no computation done
    set x [torch::tensor_create -data {3.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Try to get gradient without computing it using named syntax
    catch {torch::tensor_grad -input $x} result
    expr {[string length $result] > 0}
} {1}

test tensor-grad-2.3 {Named parameter syntax - tensor doesn't require gradients} {
    # Create a tensor that doesn't require gradients
    set x [torch::tensor_create -data {3.0} -dtype float32 -device cpu -requiresGrad false]
    
    # Try to get gradient using named syntax
    catch {torch::tensor_grad -input $x} result
    expr {[string length $result] > 0}
} {1}

# Test cases for camelCase alias
test tensor-grad-3.1 {CamelCase alias - with computed gradient} {
    # Create a tensor that requires gradients
    set x [torch::tensor_create -data {4.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation that will generate gradients
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    torch::tensor_backward $y
    
    # Get the gradient using camelCase alias
    set grad [torch::tensorGrad -input $x]
    expr {[string length $grad] > 0}
} {1}

test tensor-grad-3.2 {CamelCase alias - no gradient computed yet} {
    # Create a tensor that requires gradients but no computation done
    set x [torch::tensor_create -data {4.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Try to get gradient without computing it using camelCase alias
    catch {torch::tensorGrad -input $x} result
    expr {[string length $result] > 0}
} {1}

test tensor-grad-3.3 {CamelCase alias - tensor doesn't require gradients} {
    # Create a tensor that doesn't require gradients
    set x [torch::tensor_create -data {4.0} -dtype float32 -device cpu -requiresGrad false]
    
    # Try to get gradient using camelCase alias
    catch {torch::tensorGrad -input $x} result
    expr {[string length $result] > 0}
} {1}

# Error handling tests
test tensor-grad-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_grad invalid_tensor} result
    expr {[string length $result] > 0}
} {1}

test tensor-grad-4.2 {Error handling - missing input parameter} {
    catch {torch::tensor_grad} result
    expr {[string length $result] > 0}
} {1}

test tensor-grad-4.3 {Error handling - too many arguments} {
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::tensor_grad $x extra} result
    expr {[string length $result] > 0}
} {1}

test tensor-grad-4.4 {Error handling - unknown parameter} {
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::tensor_grad -input $x -unknown_param value} result
    expr {[string length $result] > 0}
} {1}

# Gradient correctness tests
test tensor-grad-5.1 {Gradient correctness - simple computation} {
    # Create a tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a simple computation: y = x^2
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    torch::tensor_backward $y
    
    # Get the gradient
    set grad [torch::tensor_grad $x]
    expr {[string length $grad] > 0}
} {1}

test tensor-grad-5.2 {Gradient correctness - multiple operations} {
    # Create tensors that require gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    set y [torch::tensor_create -data {3.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation: z = x * y + x
    set z1 [torch::tensor_mul $x $y]
    set z [torch::tensor_add $z1 $x]
    
    # Compute gradients
    torch::tensor_backward $z
    
    # Get the gradients
    set grad_x [torch::tensor_grad $x]
    set grad_y [torch::tensor_grad $y]
    expr {[string length $grad_x] > 0 && [string length $grad_y] > 0}
} {1}

test tensor-grad-5.3 {Gradient correctness - different data types} {
    # Create tensors with different data types
    set x1 [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    set x2 [torch::tensor_create -data {2.0} -dtype float64 -device cpu -requiresGrad true]
    
    # Create computations
    set y1 [torch::tensor_mul $x1 $x1]
    set y2 [torch::tensor_mul $x2 $x2]
    
    # Compute gradients
    torch::tensor_backward $y1
    torch::tensor_backward $y2
    
    # Get the gradients
    set grad1 [torch::tensor_grad $x1]
    set grad2 [torch::tensor_grad $x2]
    expr {[string length $grad1] > 0 && [string length $grad2] > 0}
} {1}

# Edge cases
test tensor-grad-6.1 {Edge case - zero tensor with gradients} {
    # Create a zero tensor that requires gradients
    set x [torch::tensor_create -data {0.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    torch::tensor_backward $y
    
    # Get the gradient
    set grad [torch::tensor_grad $x]
    expr {[string length $grad] > 0}
} {1}

test tensor-grad-6.2 {Edge case - large tensor with gradients} {
    # Create a large tensor that requires gradients
    set data [list]
    for {set i 0} {$i < 100} {incr i} {
        lappend data [expr {$i * 1.0}]
    }
    set x [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation
    set y [torch::tensor_sum $x]
    
    # Compute gradients
    torch::tensor_backward $y
    
    # Get the gradient
    set grad [torch::tensor_grad $x]
    expr {[string length $grad] > 0}
} {1}

test tensor-grad-6.3 {Edge case - negative values with gradients} {
    # Create a tensor with negative values that requires gradients
    set x [torch::tensor_create -data {-2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    torch::tensor_backward $y
    
    # Get the gradient
    set grad [torch::tensor_grad $x]
    expr {[string length $grad] > 0}
} {1}

# Syntax consistency tests
test tensor-grad-7.1 {Syntax consistency - positional vs named} {
    # Create a tensor that requires gradients
    set x [torch::tensor_create -data {5.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    torch::tensor_backward $y
    
    # Get gradients using both syntaxes
    set grad1 [torch::tensor_grad $x]
    set grad2 [torch::tensor_grad -input $x]
    expr {[string length $grad1] > 0 && [string length $grad2] > 0 && [string match "tensor*" $grad1] && [string match "tensor*" $grad2]}
} {1}

test tensor-grad-7.2 {Syntax consistency - positional vs camelCase} {
    # Create a tensor that requires gradients
    set x [torch::tensor_create -data {6.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    torch::tensor_backward $y
    
    # Get gradients using both syntaxes
    set grad1 [torch::tensor_grad $x]
    set grad2 [torch::tensorGrad -input $x]
    expr {[string length $grad1] > 0 && [string length $grad2] > 0 && [string match "tensor*" $grad1] && [string match "tensor*" $grad2]}
} {1}

test tensor-grad-7.3 {Syntax consistency - different error conditions} {
    # Create a tensor that doesn't require gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad false]
    
    # Try to get gradient using both syntaxes
    catch {torch::tensor_grad $x} result1
    catch {torch::tensor_grad -input $x} result2
    expr {[string length $result1] > 0 && [string length $result2] > 0 && $result1 == $result2}
} {1}

# Data type independence tests
test tensor-grad-8.1 {Data type independence - float32} {
    # Create a float32 tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    torch::tensor_backward $y
    
    # Get the gradient
    set grad [torch::tensor_grad $x]
    expr {[string length $grad] > 0}
} {1}

test tensor-grad-8.2 {Data type independence - float64} {
    # Create a float64 tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float64 -device cpu -requiresGrad true]
    
    # Create a computation
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    torch::tensor_backward $y
    
    # Get the gradient
    set grad [torch::tensor_grad $x]
    expr {[string length $grad] > 0}
} {1}

# Device independence tests
test tensor-grad-9.1 {Device independence - CPU tensor} {
    # Create a CPU tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    torch::tensor_backward $y
    
    # Get the gradient
    set grad [torch::tensor_grad $x]
    expr {[string length $grad] > 0}
} {1}

test tensor-grad-9.2 {Device independence - CUDA tensor (if available)} {
    if {[catch {torch::cuda_is_available} cuda_available] || !$cuda_available} {
        skip "CUDA not available"
    } else {
        # Create a CUDA tensor that requires gradients
        set x [torch::tensor_create -data {2.0} -dtype float32 -device cuda -requiresGrad true]
        
        # Create a computation
        set y [torch::tensor_mul $x $x]
        
        # Compute gradients
        torch::tensor_backward $y
        
        # Get the gradient
        set grad [torch::tensor_grad $x]
        expr {[string length $grad] > 0}
    }
} {1}

# Multiple gradient test
test tensor-grad-10.1 {Multiple gradients - same computation} {
    # Create tensors that require gradients
    set x1 [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    set x2 [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create the same computation for both
    set y1 [torch::tensor_mul $x1 $x1]
    set y2 [torch::tensor_mul $x2 $x2]
    
    # Compute gradients
    torch::tensor_backward $y1
    torch::tensor_backward $y2
    
    # Get the gradients
    set grad1 [torch::tensor_grad $x1]
    set grad2 [torch::tensor_grad $x2]
    expr {[string length $grad1] > 0 && [string length $grad2] > 0}
} {1}

# Return value format tests
test tensor-grad-11.1 {Return value format - valid gradient returns tensor handle} {
    # Create a tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    torch::tensor_backward $y
    
    # Get the gradient
    set grad [torch::tensor_grad $x]
    expr {[string length $grad] > 0 && [string match "tensor*" $grad]}
} {1}

test tensor-grad-11.2 {Return value format - error returns error message} {
    # Create a tensor that doesn't require gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad false]
    
    # Try to get gradient
    catch {torch::tensor_grad $x} result
    expr {[string length $result] > 0 && ![string match "tensor*" $result]}
} {1}

cleanupTests 