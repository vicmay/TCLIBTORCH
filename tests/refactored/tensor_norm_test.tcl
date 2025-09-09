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
test tensor-norm-1.1 {Basic positional syntax - L2 norm} {
    # Create test tensor
    set tensor [torch::tensor_create {3.0 4.0} float32 cpu true]
    
    # Test L2 norm (default)
    set result [torch::tensor_norm $tensor]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {5.0}

test tensor-norm-1.2 {Positional syntax with p=1} {
    # Create test tensor
    set tensor [torch::tensor_create {3.0 4.0} float32 cpu true]
    
    # Test L1 norm
    set result [torch::tensor_norm $tensor 1.0]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {7.0}

test tensor-norm-1.3 {Positional syntax with p=inf} {
    # Create test tensor
    set tensor [torch::tensor_create {3.0 4.0} float32 cpu true]
    
    # Test infinity norm
    set result [torch::tensor_norm $tensor inf]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {4.0}

test tensor-norm-1.4 {Positional syntax with dimension} {
    # Create 2D test tensor
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    
    # Test norm along dimension 0
    set result [torch::tensor_norm $tensor 2.0 0]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {3.1622776985168457 4.4721360206604}

# Test cases for named parameter syntax
test tensor-norm-2.1 {Named parameter syntax - L2 norm} {
    # Create test tensor
    set tensor [torch::tensor_create {3.0 4.0} float32 cpu true]
    
    # Test L2 norm with named parameters
    set result [torch::tensor_norm -tensor $tensor]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {5.0}

test tensor-norm-2.2 {Named parameter syntax with p=1} {
    # Create test tensor
    set tensor [torch::tensor_create {3.0 4.0} float32 cpu true]
    
    # Test L1 norm with named parameters
    set result [torch::tensor_norm -tensor $tensor -p 1.0]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {7.0}

test tensor-norm-2.3 {Named parameter syntax with p and dim} {
    # Create 2D test tensor
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    
    # Test norm with named parameters
    set result [torch::tensor_norm -tensor $tensor -p 2.0 -dim 0]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {3.1622776985168457 4.4721360206604}

# Test cases for camelCase alias
test tensor-norm-3.1 {CamelCase alias positional syntax} {
    # Create test tensor
    set tensor [torch::tensor_create {3.0 4.0} float32 cpu true]
    
    # Test using camelCase alias
    set result [torch::tensorNorm $tensor]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {5.0}

test tensor-norm-3.2 {CamelCase alias named parameter syntax} {
    # Create test tensor
    set tensor [torch::tensor_create {3.0 4.0} float32 cpu true]
    
    # Test using camelCase alias with named parameters
    set result [torch::tensorNorm -tensor $tensor -p 1.0]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {7.0}

# Error handling tests
test tensor-norm-4.1 {Error handling - missing tensor} {
    # Test with non-existent tensor
    set result [catch {torch::tensor_norm nonexistent_tensor} error]
    return [list $result $error]
} {1 {Tensor not found}}

test tensor-norm-4.2 {Error handling - insufficient arguments} {
    set result [catch {torch::tensor_norm} error]
    return [list $result $error]
} {1 {Usage: torch::tensor_norm tensor ?p? ?dim? | torch::tensor_norm -tensor tensor ?-p value? ?-dim value?}}

test tensor-norm-4.3 {Error handling - invalid p value} {
    set tensor [torch::tensor_create {1.0 2.0} float32 cpu true]
    
    # Test with invalid p value (string instead of number)
    set result [catch {torch::tensor_norm $tensor "invalid"} error]
    
    return [list $result $error]
} {1 {Invalid p value}}

test tensor-norm-4.4 {Error handling - invalid dim value} {
    set tensor [torch::tensor_create {1.0 2.0} float32 cpu true]
    
    # Test with invalid dim value (string instead of number)
    set result [catch {torch::tensor_norm $tensor 2.0 "invalid"} error]
    
    return [list $result $error]
} {1 {Invalid dim value}}

test tensor-norm-4.5 {Error handling - unknown named parameter} {
    set tensor [torch::tensor_create {1.0 2.0} float32 cpu true]
    
    # Test with unknown parameter
    set result [catch {torch::tensor_norm -tensor $tensor -unknown param} error]
    
    return [list $result $error]
} {1 {Unknown parameter: -unknown. Valid parameters are: -tensor, -p, -dim}}

# Test with different tensor shapes
test tensor-norm-5.1 {1D tensor norm} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    set result [torch::tensor_norm $tensor]
    set data [torch::tensor_to_list $result]
    return $data
} {3.7416574954986572}

test tensor-norm-5.2 {3D tensor norm} {
    # Create 2D tensors first
    set tensor1 [torch::tensor_create -data {{1.0 2.0} {3.0 4.0}} -dtype float32 -device cpu -requiresGrad true]
    set tensor2 [torch::tensor_create -data {{5.0 6.0} {7.0 8.0}} -dtype float32 -device cpu -requiresGrad true]
    
    # Stack them to create a 3D tensor
    set tensor [torch::tensor_stack -tensors [list $tensor1 $tensor2] -dim 0]
    
    # Calculate norm
    set result [torch::tensor_norm $tensor]
    set data [torch::tensor_to_list $result]
    
    # Use tolerance-based comparison
    set expected 14.2828568570857
    set tolerance 1e-5
    set diff [expr {abs($data - $expected)}]
    
    return [expr {$diff < $tolerance}]
} {1}

# Test with different p values
test tensor-norm-6.1 {p=0.5 norm} {
    set tensor [torch::tensor_create {1.0 2.0} float32 cpu true]
    set result [torch::tensor_norm $tensor 0.5]
    set data [torch::tensor_to_list $result]
    return $data
} {5.828427791595459}

test tensor-norm-6.2 {p=3.0 norm} {
    set tensor [torch::tensor_create {1.0 2.0} float32 cpu true]
    set result [torch::tensor_norm $tensor 3.0]
    set data [torch::tensor_to_list $result]
    return $data
} {2.0800838470458984}

# Test with negative dimension
test tensor-norm-7.1 {Negative dimension} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    set result [torch::tensor_norm $tensor 2.0 -1]
    set data [torch::tensor_to_list $result]
    return $data
} {2.2360680103302 5.0}

cleanupTests 