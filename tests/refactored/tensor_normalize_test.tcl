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
test tensor-normalize-1.1 {Basic positional syntax - L2 normalize} {
    # Create test tensor
    set tensor [torch::tensor_create {3.0 4.0} float32 cpu true]
    
    # Test L2 normalize (default)
    set result [torch::tensor_normalize $tensor]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {0.6000000238418579 0.800000011920929}

test tensor-normalize-1.2 {Positional syntax with p=1} {
    # Create test tensor
    set tensor [torch::tensor_create {3.0 4.0} float32 cpu true]
    
    # Test L1 normalize
    set result [torch::tensor_normalize $tensor 1.0]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {0.4285714328289032 0.5714285969734192}

test tensor-normalize-1.3 {Positional syntax with dimension} {
    # Create 2D test tensor
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    
    # Test normalize along dimension 0
    set result [torch::tensor_normalize $tensor 2.0 0]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {0.3162277638912201 0.4472135901451111 0.9486832618713379 0.8944271802902222}

# Test cases for named parameter syntax
test tensor-normalize-2.1 {Named parameter syntax - L2 normalize} {
    # Create test tensor
    set tensor [torch::tensor_create {3.0 4.0} float32 cpu true]
    
    # Test L2 normalize with named parameters
    set result [torch::tensor_normalize -tensor $tensor]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {0.6000000238418579 0.800000011920929}

test tensor-normalize-2.2 {Named parameter syntax with p=1} {
    # Create test tensor
    set tensor [torch::tensor_create {3.0 4.0} float32 cpu true]
    
    # Test L1 normalize with named parameters
    set result [torch::tensor_normalize -tensor $tensor -p 1.0]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {0.4285714328289032 0.5714285969734192}

test tensor-normalize-2.3 {Named parameter syntax with p and dim} {
    # Create 2D test tensor
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    
    # Test normalize with named parameters
    set result [torch::tensor_normalize -tensor $tensor -p 2.0 -dim 0]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {0.3162277638912201 0.4472135901451111 0.9486832618713379 0.8944271802902222}

# Test cases for camelCase alias
test tensor-normalize-3.1 {CamelCase alias positional syntax} {
    # Create test tensor
    set tensor [torch::tensor_create {3.0 4.0} float32 cpu true]
    
    # Test using camelCase alias
    set result [torch::tensorNormalize $tensor]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {0.6000000238418579 0.800000011920929}

test tensor-normalize-3.2 {CamelCase alias named parameter syntax} {
    # Create test tensor
    set tensor [torch::tensor_create {3.0 4.0} float32 cpu true]
    
    # Test using camelCase alias with named parameters
    set result [torch::tensorNormalize -tensor $tensor -p 1.0]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {0.4285714328289032 0.5714285969734192}

# Error handling tests
test tensor-normalize-4.1 {Error handling - missing tensor} {
    # Test with non-existent tensor
    set result [catch {torch::tensor_normalize nonexistent_tensor} error]
    return [list $result $error]
} {1 {Tensor not found}}

test tensor-normalize-4.2 {Error handling - insufficient arguments} {
    set result [catch {torch::tensor_normalize} error]
    return [list $result $error]
} {1 {Usage: torch::tensor_normalize tensor ?p? ?dim? | torch::tensor_normalize -tensor tensor ?-p value? ?-dim value?}}

test tensor-normalize-4.3 {Error handling - invalid p value} {
    set tensor [torch::tensor_create {1.0 2.0} float32 cpu true]
    
    # Test with invalid p value (string instead of number)
    set result [catch {torch::tensor_normalize $tensor "invalid"} error]
    
    return [list $result $error]
} {1 {Invalid p value}}

test tensor-normalize-4.4 {Error handling - invalid dim value} {
    set tensor [torch::tensor_create {1.0 2.0} float32 cpu true]
    
    # Test with invalid dim value (string instead of number)
    set result [catch {torch::tensor_normalize $tensor 2.0 "invalid"} error]
    
    return [list $result $error]
} {1 {Invalid dim value}}

test tensor-normalize-4.5 {Error handling - unknown named parameter} {
    set tensor [torch::tensor_create {1.0 2.0} float32 cpu true]
    
    # Test with unknown parameter
    set result [catch {torch::tensor_normalize -tensor $tensor -unknown param} error]
    
    return [list $result $error]
} {1 {Unknown parameter: -unknown. Valid parameters are: -tensor, -p, -dim}}

# Test with different tensor shapes
test tensor-normalize-5.1 {1D tensor normalize} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    set result [torch::tensor_normalize $tensor]
    set data [torch::tensor_to_list $result]
    return $data
} {0.26726123690605164 0.5345224738121033 0.8017836809158325}

test tensor-normalize-5.2 {2D tensor normalize} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    set result [torch::tensor_normalize $tensor]
    set data [torch::tensor_to_list $result]
    return $data
} {0.18257418274879456 0.3651483654975891 0.5477225184440613 0.7302967309951782}

# Test with different p values
test tensor-normalize-6.1 {p=0.5 normalize} {
    set tensor [torch::tensor_create {1.0 2.0} float32 cpu true]
    set result [torch::tensor_normalize $tensor 0.5]
    set data [torch::tensor_to_list $result]
    return $data
} {0.17157284915447235 0.3431456983089447}

test tensor-normalize-6.2 {p=3.0 normalize} {
    set tensor [torch::tensor_create {1.0 2.0} float32 cpu true]
    set result [torch::tensor_normalize $tensor 3.0]
    set data [torch::tensor_to_list $result]
    return $data
} {0.48074984550476074 0.9614996910095215}

# Test with negative dimension
test tensor-normalize-7.1 {Negative dimension} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    set result [torch::tensor_normalize $tensor 2.0 -1]
    set data [torch::tensor_to_list $result]
    return $data
} {0.4472135901451111 0.8944271802902222 0.6000000238418579 0.800000011920929}

cleanupTests 