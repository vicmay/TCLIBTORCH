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

# Helper function to create a test matrix
proc create_test_matrix {} {
    return [torch::tensor_create -data {{4.0 0.0 0.0} {0.0 3.0 0.0} {0.0 0.0 2.0}} -dtype "float32"]
}

# Helper function to check if tensors are approximately equal
proc tensors_approx_equal {tensor1 tensor2} {
    return [torch::allclose -input $tensor1 -other $tensor2 -rtol 1e-3 -atol 1e-3]
}

# Test cases for positional syntax
test spectral-norm-1.1 {Basic positional syntax} {
    set matrix [create_test_matrix]
    set result [torch::spectral_norm $matrix]
    
    # The spectral norm should normalize by the largest singular value (4.0)
    set expected [torch::tensor_create -data {{1.0 0.0 0.0} {0.0 0.75 0.0} {0.0 0.0 0.5}} -dtype "float32"]
    tensors_approx_equal $result $expected
} {1}

test spectral-norm-1.2 {Positional syntax with power iterations} {
    set matrix [create_test_matrix]
    set result [torch::spectral_norm $matrix 5]
    
    # Result should be the same as without power iterations for this simple case
    set expected [torch::tensor_create -data {{1.0 0.0 0.0} {0.0 0.75 0.0} {0.0 0.0 0.5}} -dtype "float32"]
    tensors_approx_equal $result $expected
} {1}

test spectral-norm-1.3 {Error handling - invalid tensor} {
    catch {torch::spectral_norm "invalid_tensor"} err
    set err
} {Error in spectral_norm: Invalid tensor}

test spectral-norm-1.4 {Error handling - 1D tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype "float32"]
    catch {torch::spectral_norm $tensor} err
    set err
} {Error in spectral_norm: Spectral norm requires at least 2D tensor}

# Test cases for camelCase alias
test spectral-norm-2.1 {CamelCase alias} {
    set matrix [create_test_matrix]
    set result [torch::spectralNorm $matrix]
    
    set expected [torch::tensor_create -data {{1.0 0.0 0.0} {0.0 0.75 0.0} {0.0 0.0 0.5}} -dtype "float32"]
    tensors_approx_equal $result $expected
} {1}

cleanupTests 