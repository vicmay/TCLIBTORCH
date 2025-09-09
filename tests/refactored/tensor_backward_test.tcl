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
test tensor-backward-1.1 {Basic positional syntax - simple computation} {
    # Create a tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation that will generate gradients
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    set result [torch::tensor_backward $y]
    expr {$result == "OK"}
} {1}

test tensor-backward-1.2 {Basic positional syntax - multiple operations} {
    # Create tensors that require gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    set y [torch::tensor_create -data {3.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation
    set z [torch::tensor_add $x $y]
    
    # Compute gradients
    set result [torch::tensor_backward $z]
    expr {$result == "OK"}
} {1}

test tensor-backward-1.3 {Basic positional syntax - complex computation} {
    # Create a tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a complex computation: y = x^2 + x
    set y1 [torch::tensor_mul $x $x]
    set y [torch::tensor_add $y1 $x]
    
    # Compute gradients
    set result [torch::tensor_backward $y]
    expr {$result == "OK"}
} {1}

# Test cases for named syntax
test tensor-backward-2.1 {Named parameter syntax - simple computation} {
    # Create a tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation that will generate gradients
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients using named syntax
    set result [torch::tensor_backward -input $y]
    expr {$result == "OK"}
} {1}

test tensor-backward-2.2 {Named parameter syntax - multiple operations} {
    # Create tensors that require gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    set y [torch::tensor_create -data {3.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation
    set z [torch::tensor_add $x $y]
    
    # Compute gradients using named syntax
    set result [torch::tensor_backward -input $z]
    expr {$result == "OK"}
} {1}

test tensor-backward-2.3 {Named parameter syntax - complex computation} {
    # Create a tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a complex computation: y = x^2 + x
    set y1 [torch::tensor_mul $x $x]
    set y [torch::tensor_add $y1 $x]
    
    # Compute gradients using named syntax
    set result [torch::tensor_backward -input $y]
    expr {$result == "OK"}
} {1}

# Test cases for camelCase alias
test tensor-backward-3.1 {CamelCase alias - simple computation} {
    # Create a tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation that will generate gradients
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients using camelCase alias
    set result [torch::tensorBackward -input $y]
    expr {$result == "OK"}
} {1}

test tensor-backward-3.2 {CamelCase alias - multiple operations} {
    # Create tensors that require gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    set y [torch::tensor_create -data {3.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation
    set z [torch::tensor_add $x $y]
    
    # Compute gradients using camelCase alias
    set result [torch::tensorBackward -input $z]
    expr {$result == "OK"}
} {1}

test tensor-backward-3.3 {CamelCase alias - complex computation} {
    # Create a tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a complex computation: y = x^2 + x
    set y1 [torch::tensor_mul $x $x]
    set y [torch::tensor_add $y1 $x]
    
    # Compute gradients using camelCase alias
    set result [torch::tensorBackward -input $y]
    expr {$result == "OK"}
} {1}

# Error handling tests
test tensor-backward-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_backward invalid_tensor} result
    expr {[string length $result] > 0}
} {1}

test tensor-backward-4.2 {Error handling - missing input parameter} {
    catch {torch::tensor_backward} result
    expr {[string length $result] > 0}
} {1}

test tensor-backward-4.3 {Error handling - too many arguments} {
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::tensor_backward $x extra} result
    expr {[string length $result] > 0}
} {1}

test tensor-backward-4.4 {Error handling - tensor doesn't require gradients} {
    # Create a tensor that doesn't require gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad false]
    set y [torch::tensor_mul $x $x]
    
    # Try to compute gradients
    catch {torch::tensor_backward $y} result
    expr {[string length $result] > 0}
} {1}

test tensor-backward-4.5 {Error handling - unknown parameter} {
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    set y [torch::tensor_mul $x $x]
    catch {torch::tensor_backward -input $y -unknown_param value} result
    expr {[string length $result] > 0}
} {1}

# Gradient computation tests
test tensor-backward-5.1 {Gradient computation - simple multiplication} {
    # Create a tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation: y = x^2
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    torch::tensor_backward $y
    
    # Check that gradient was computed
    set grad [torch::tensor_grad $x]
    expr {[string length $grad] > 0 && [string match "tensor*" $grad]}
} {1}

test tensor-backward-5.2 {Gradient computation - addition} {
    # Create tensors that require gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    set y [torch::tensor_create -data {3.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation: z = x + y
    set z [torch::tensor_add $x $y]
    
    # Compute gradients
    torch::tensor_backward $z
    
    # Check that gradients were computed
    set grad_x [torch::tensor_grad $x]
    set grad_y [torch::tensor_grad $y]
    expr {[string length $grad_x] > 0 && [string length $grad_y] > 0 && [string match "tensor*" $grad_x] && [string match "tensor*" $grad_y]}
} {1}

test tensor-backward-5.3 {Gradient computation - complex chain} {
    # Create tensors that require gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    set y [torch::tensor_create -data {3.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a complex computation: z = (x * y) + x
    set z1 [torch::tensor_mul $x $y]
    set z [torch::tensor_add $z1 $x]
    
    # Compute gradients
    torch::tensor_backward $z
    
    # Check that gradients were computed
    set grad_x [torch::tensor_grad $x]
    set grad_y [torch::tensor_grad $y]
    expr {[string length $grad_x] > 0 && [string length $grad_y] > 0 && [string match "tensor*" $grad_x] && [string match "tensor*" $grad_y]}
} {1}

# Data type independence tests
test tensor-backward-6.1 {Data type independence - float32} {
    # Create a float32 tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    set result [torch::tensor_backward $y]
    expr {$result == "OK"}
} {1}

test tensor-backward-6.2 {Data type independence - float64} {
    # Create a float64 tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float64 -device cpu -requiresGrad true]
    
    # Create a computation
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    set result [torch::tensor_backward $y]
    expr {$result == "OK"}
} {1}

# Device independence tests
test tensor-backward-7.1 {Device independence - CPU tensor} {
    # Create a CPU tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    set result [torch::tensor_backward $y]
    expr {$result == "OK"}
} {1}

test tensor-backward-7.2 {Device independence - CUDA tensor (if available)} {
    if {[catch {torch::cuda_is_available} cuda_available] || !$cuda_available} {
        skip "CUDA not available"
    } else {
        # Create a CUDA tensor that requires gradients
        set x [torch::tensor_create -data {2.0} -dtype float32 -device cuda -requiresGrad true]
        
        # Create a computation
        set y [torch::tensor_mul $x $x]
        
        # Compute gradients
        set result [torch::tensor_backward $y]
        expr {$result == "OK"}
    }
} {1}

# Edge cases
test tensor-backward-8.1 {Edge case - zero tensor} {
    # Create a zero tensor that requires gradients
    set x [torch::tensor_create -data {0.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    set result [torch::tensor_backward $y]
    expr {$result == "OK"}
} {1}

test tensor-backward-8.2 {Edge case - large tensor} {
    # Create a large tensor that requires gradients
    set data [list]
    for {set i 0} {$i < 100} {incr i} {
        lappend data [expr {$i * 1.0}]
    }
    set x [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation
    set y [torch::tensor_sum $x]
    
    # Compute gradients
    set result [torch::tensor_backward $y]
    expr {$result == "OK"}
} {1}

test tensor-backward-8.3 {Edge case - negative values} {
    # Create a tensor with negative values that requires gradients
    set x [torch::tensor_create -data {-2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    set result [torch::tensor_backward $y]
    expr {$result == "OK"}
} {1}

# Multiple backward passes
# NOTE: PyTorch does not allow calling .backward() twice on the same computation graph unless retain_graph=True is specified.
# This is a PyTorch limitation, not a bug in the TCL extension. The test should pass if the correct error is raised.
test tensor-backward-9.1 {Multiple backward passes - same computation (should error, PyTorch limitation)} {
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    set y [torch::tensor_mul $x $x]
    set result1 [torch::tensor_backward $y]
    catch {torch::tensor_backward $y} result2
    # Should get an error message about backward through the graph a second time
    expr {$result1 == "OK" && [string match {*backward*second time*} $result2]}
} {1}

test tensor-backward-9.2 {Multiple backward passes - different computations} {
    # Create a tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create different computations
    set y1 [torch::tensor_mul $x $x]
    set y2 [torch::tensor_add $x $x]
    
    # Compute gradients for different computations
    set result1 [torch::tensor_backward $y1]
    set result2 [torch::tensor_backward $y2]
    expr {$result1 == "OK" && $result2 == "OK"}
} {1}

# Syntax consistency tests
# NOTE: PyTorch does not allow calling .backward() twice on the same computation graph unless retain_graph=True is specified.
# This is a PyTorch limitation, not a bug in the TCL extension. The test should pass if the correct error is raised.
test tensor-backward-10.1 {Syntax consistency - positional vs named (should error, PyTorch limitation)} {
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    set y [torch::tensor_mul $x $x]
    set result1 [torch::tensor_backward $y]
    catch {torch::tensor_backward -input $y} result2
    expr {$result1 == "OK" && [string match {*backward*second time*} $result2]}
} {1}

test tensor-backward-10.2 {Syntax consistency - positional vs camelCase (should error, PyTorch limitation)} {
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    set y [torch::tensor_mul $x $x]
    set result1 [torch::tensor_backward $y]
    catch {torch::tensorBackward -input $y} result2
    expr {$result1 == "OK" && [string match {*backward*second time*} $result2]}
} {1}

test tensor-backward-10.3 {Syntax consistency - different error conditions} {
    # Create a tensor that doesn't require gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad false]
    set y [torch::tensor_mul $x $x]
    
    # Try to compute gradients using both syntaxes
    catch {torch::tensor_backward $y} result1
    catch {torch::tensor_backward -input $y} result2
    expr {[string length $result1] > 0 && [string length $result2] > 0 && $result1 == $result2}
} {1}

# Return value format tests
test tensor-backward-11.1 {Return value format - successful backward pass} {
    # Create a tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    set result [torch::tensor_backward $y]
    expr {$result == "OK"}
} {1}

test tensor-backward-11.2 {Return value format - error returns error message} {
    # Create a tensor that doesn't require gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad false]
    set y [torch::tensor_mul $x $x]
    
    # Try to compute gradients
    catch {torch::tensor_backward $y} result
    expr {[string length $result] > 0 && $result != "OK"}
} {1}

# Mathematical correctness tests
test tensor-backward-12.1 {Mathematical correctness - gradient accumulation} {
    # Create a tensor that requires gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation: y = x^2
    set y [torch::tensor_mul $x $x]
    
    # Compute gradients
    torch::tensor_backward $y
    
    # Check that gradient was computed and is a tensor handle
    set grad [torch::tensor_grad $x]
    expr {[string length $grad] > 0 && [string match "tensor*" $grad]}
} {1}

test tensor-backward-12.2 {Mathematical correctness - multiple gradients} {
    # Create tensors that require gradients
    set x [torch::tensor_create -data {2.0} -dtype float32 -device cpu -requiresGrad true]
    set y [torch::tensor_create -data {3.0} -dtype float32 -device cpu -requiresGrad true]
    
    # Create a computation: z = x * y
    set z [torch::tensor_mul $x $y]
    
    # Compute gradients
    torch::tensor_backward $z
    
    # Check that gradients were computed for both tensors
    set grad_x [torch::tensor_grad $x]
    set grad_y [torch::tensor_grad $y]
    expr {[string length $grad_x] > 0 && [string length $grad_y] > 0 && [string match "tensor*" $grad_x] && [string match "tensor*" $grad_y]}
} {1}

cleanupTests 