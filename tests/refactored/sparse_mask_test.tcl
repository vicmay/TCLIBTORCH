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

# Helper function to check tensor shape
proc check_tensor_shape {tensor expected_shape} {
    set shape [torch::tensor_shape $tensor]
    if {$shape != $expected_shape} {
        error "Expected shape $expected_shape but got $shape"
    }
}

# Test cases for positional syntax
test sparse_mask-1.1 {Basic positional syntax} {
    # Create a sparse tensor
    set indices [torch::tensor_create {0 1 2 0 1 2} int64 cpu]
    set indices [torch::tensor_reshape -input $indices -shape {2 3}]
    set values [torch::tensor_create {1.0 2.0 3.0} float32 cpu]
    set tensor [torch::sparse_coo_tensor $indices $values {3 3}]
    
    # Create a mask tensor
    set mask_indices [torch::tensor_create {0 1} int64 cpu]
    set mask_indices [torch::tensor_reshape -input $mask_indices -shape {2 1}]
    set mask_values [torch::tensor_create {1.0} float32 cpu]
    set mask [torch::sparse_coo_tensor $mask_indices $mask_values {3 3}]
    
    # Apply mask
    set result [torch::sparse_mask $tensor $mask]
    check_tensor_shape $result {3 3}
    expr {1}
} {1}

# Test cases for named parameter syntax
test sparse_mask-2.1 {Named parameter syntax} {
    # Create a sparse tensor
    set indices [torch::tensor_create {0 1 2 0 1 2} int64 cpu]
    set indices [torch::tensor_reshape -input $indices -shape {2 3}]
    set values [torch::tensor_create {1.0 2.0 3.0} float32 cpu]
    set tensor [torch::sparse_coo_tensor $indices $values {3 3}]
    
    # Create a mask tensor
    set mask_indices [torch::tensor_create {0 1} int64 cpu]
    set mask_indices [torch::tensor_reshape -input $mask_indices -shape {2 1}]
    set mask_values [torch::tensor_create {1.0} float32 cpu]
    set mask [torch::sparse_coo_tensor $mask_indices $mask_values {3 3}]
    
    # Apply mask
    set result [torch::sparse_mask -tensor $tensor -mask $mask]
    check_tensor_shape $result {3 3}
    expr {1}
} {1}

# Test cases for camelCase alias
test sparse_mask-3.1 {CamelCase alias} {
    # Create a sparse tensor
    set indices [torch::tensor_create {0 1 2 0 1 2} int64 cpu]
    set indices [torch::tensor_reshape -input $indices -shape {2 3}]
    set values [torch::tensor_create {1.0 2.0 3.0} float32 cpu]
    set tensor [torch::sparse_coo_tensor $indices $values {3 3}]
    
    # Create a mask tensor
    set mask_indices [torch::tensor_create {0 1} int64 cpu]
    set mask_indices [torch::tensor_reshape -input $mask_indices -shape {2 1}]
    set mask_values [torch::tensor_create {1.0} float32 cpu]
    set mask [torch::sparse_coo_tensor $mask_indices $mask_values {3 3}]
    
    # Apply mask
    set result [torch::sparseMask -tensor $tensor -mask $mask]
    check_tensor_shape $result {3 3}
    expr {1}
} {1}

# Error handling tests
test sparse_mask-4.1 {Error: Invalid tensor} {
    set mask_indices [torch::tensor_create {0 1} int64 cpu]
    set mask_indices [torch::tensor_reshape -input $mask_indices -shape {2 1}]
    set mask_values [torch::tensor_create {1.0} float32 cpu]
    set mask [torch::sparse_coo_tensor $mask_indices $mask_values {3 3}]
    catch {torch::sparse_mask invalid_tensor $mask} err
    set err
} {Invalid tensor}

test sparse_mask-4.2 {Error: Invalid mask tensor} {
    set indices [torch::tensor_create {0 1 2 0 1 2} int64 cpu]
    set indices [torch::tensor_reshape -input $indices -shape {2 3}]
    set values [torch::tensor_create {1.0 2.0 3.0} float32 cpu]
    set tensor [torch::sparse_coo_tensor $indices $values {3 3}]
    catch {torch::sparse_mask $tensor invalid_mask} err
    set err
} {Invalid mask tensor}

test sparse_mask-4.3 {Error: Missing required parameters} {
    set indices [torch::tensor_create {0 1 2 0 1 2} int64 cpu]
    set indices [torch::tensor_reshape -input $indices -shape {2 3}]
    set values [torch::tensor_create {1.0 2.0 3.0} float32 cpu]
    set tensor [torch::sparse_coo_tensor $indices $values {3 3}]
    catch {torch::sparse_mask -tensor $tensor} err
    set err
} {Required parameters missing: tensor, mask}

test sparse_mask-4.4 {Error: Unknown parameter} {
    set indices [torch::tensor_create {0 1 2 0 1 2} int64 cpu]
    set indices [torch::tensor_reshape -input $indices -shape {2 3}]
    set values [torch::tensor_create {1.0 2.0 3.0} float32 cpu]
    set tensor [torch::sparse_coo_tensor $indices $values {3 3}]
    
    set mask_indices [torch::tensor_create {0 1} int64 cpu]
    set mask_indices [torch::tensor_reshape -input $mask_indices -shape {2 1}]
    set mask_values [torch::tensor_create {1.0} float32 cpu]
    set mask [torch::sparse_coo_tensor $mask_indices $mask_values {3 3}]
    
    catch {torch::sparse_mask -tensor $tensor -mask $mask -invalid_param 0} err
    set err
} {Unknown parameter: -invalid_param}

cleanupTests 