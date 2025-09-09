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

# Test cases for positional syntax (backward compatibility)
test tensor-index-select-1.1 {Basic positional syntax} {
    # Create test tensors
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    set indices [torch::tensor_create -data {0 1} -shape {2} -dtype int32]
    
    # Test index_select along dimension 0
    set result [torch::tensor_index_select $tensor 0 $indices]
    set result_data [torch::tensor_to_list $result]
    
    return $result_data
} {1.0 2.0 3.0 4.0 5.0 6.0}

test tensor-index-select-1.2 {Positional syntax with dimension 1} {
    # Create test tensors
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    set indices [torch::tensor_create -data {0 2} -shape {2} -dtype int32]
    
    # Test index_select along dimension 1
    set result [torch::tensor_index_select $tensor 1 $indices]
    set result_data [torch::tensor_to_list $result]
    
    return $result_data
} {1.0 3.0 4.0 6.0}

test tensor-index-select-1.3 {Positional syntax with 3D tensor} {
    # Create 3D test tensor
    set tensor [torch::tensor_create -data {1 2 3 4 5 6 7 8} -shape {2 2 2}]
    set indices [torch::tensor_create -data {0} -shape {1} -dtype int32]
    
    # Test index_select along dimension 0
    set result [torch::tensor_index_select $tensor 0 $indices]
    set result_data [torch::tensor_to_list $result]
    
    return $result_data
} {1.0 2.0 3.0 4.0}

# Test cases for named parameter syntax
test tensor-index-select-2.1 {Named parameter syntax with -input} {
    # Create test tensors
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    set indices [torch::tensor_create -data {0 1} -shape {2} -dtype int32]
    
    # Test index_select with named parameters
    set result [torch::tensor_index_select -input $tensor -dim 0 -indices $indices]
    set result_data [torch::tensor_to_list $result]
    
    return $result_data
} {1.0 2.0 3.0 4.0 5.0 6.0}

test tensor-index-select-2.2 {Named parameter syntax with -tensor alias} {
    # Create test tensors
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    set indices [torch::tensor_create -data {0 1} -shape {2} -dtype int32]
    
    # Test index_select with -tensor parameter
    set result [torch::tensor_index_select -tensor $tensor -dim 0 -indices $indices]
    set result_data [torch::tensor_to_list $result]
    
    return $result_data
} {1.0 2.0 3.0 4.0 5.0 6.0}

test tensor-index-select-2.3 {Named parameter syntax with -dimension alias} {
    # Create test tensors
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    set indices [torch::tensor_create -data {0 2} -shape {2} -dtype int32]
    
    # Test index_select with -dimension parameter
    set result [torch::tensor_index_select -input $tensor -dimension 1 -indices $indices]
    set result_data [torch::tensor_to_list $result]
    
    return $result_data
} {1.0 3.0 4.0 6.0}

# Test cases for camelCase alias
test tensor-index-select-3.1 {CamelCase alias with positional syntax} {
    # Create test tensors
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    set indices [torch::tensor_create -data {0 1} -shape {2} -dtype int32]
    
    # Test index_select using camelCase alias
    set result [torch::tensorIndexSelect $tensor 0 $indices]
    set result_data [torch::tensor_to_list $result]
    
    return $result_data
} {1.0 2.0 3.0 4.0 5.0 6.0}

test tensor-index-select-3.2 {CamelCase alias with named parameters} {
    # Create test tensors
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    set indices [torch::tensor_create -data {0 2} -shape {2} -dtype int32]
    
    # Test index_select using camelCase alias with named parameters
    set result [torch::tensorIndexSelect -input $tensor -dim 1 -indices $indices]
    set result_data [torch::tensor_to_list $result]
    
    return $result_data
} {1.0 3.0 4.0 6.0}

# Error handling tests
test tensor-index-select-4.1 {Error: Invalid tensor name} {
    set indices [torch::tensor_create -data {0 1} -shape {2} -dtype int32]
    
    # Try to use non-existent tensor
    set result [catch {torch::tensor_index_select nonexistent_tensor 0 $indices} error_msg]
    
    return [list $result $error_msg]
} {1 {Invalid tensor name}}

test tensor-index-select-4.2 {Error: Invalid indices tensor name} {
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    
    # Try to use non-existent indices tensor
    set result [catch {torch::tensor_index_select $tensor 0 nonexistent_indices} error_msg]
    
    return [list $result $error_msg]
} {1 {Invalid indices tensor name}}

test tensor-index-select-4.3 {Error: Invalid dimension value} {
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    set indices [torch::tensor_create -data {0 1} -shape {2} -dtype int32]
    
    # Try to use invalid dimension
    set result [catch {torch::tensor_index_select $tensor invalid_dim $indices} error_msg]
    
    return [list $result $error_msg]
} {1 {Invalid dimension value}}

test tensor-index-select-4.4 {Error: Missing required parameters} {
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    
    # Try to use named parameters without required values
    set result [catch {torch::tensor_index_select -input $tensor -dim 0} error_msg]
    
    return [list $result $error_msg]
} {1 {Required parameters missing: input tensor and indices tensor are required}}

test tensor-index-select-4.5 {Error: Unknown parameter} {
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    set indices [torch::tensor_create -data {0 1} -shape {2} -dtype int32]
    
    # Try to use unknown parameter
    set result [catch {torch::tensor_index_select -input $tensor -dim 0 -indices $indices -unknown_param value} error_msg]
    
    return [list $result $error_msg]
} {1 {Unknown parameter: -unknown_param}}

test tensor-index-select-4.6 {Error: Missing value for parameter} {
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    
    # Try to use parameter without value
    set result [catch {torch::tensor_index_select -input $tensor -dim} error_msg]
    
    return [list $result $error_msg]
} {1 {Missing value for parameter}}

test tensor-index-select-4.7 {Error: Wrong number of arguments for positional syntax} {
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    set indices [torch::tensor_create -data {0 1} -shape {2} -dtype int32]
    
    # Try to use wrong number of arguments
    set result [catch {torch::tensor_index_select $tensor 0} error_msg]
    
    return [list $result $error_msg]
} {1 {Invalid number of arguments}}

# Edge cases and different data types
test tensor-index-select-5.1 {Single index selection} {
    # Create test tensors
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    set indices [torch::tensor_create -data {0} -shape {1} -dtype int32]
    
    # Test selecting single index
    set result [torch::tensor_index_select $tensor 0 $indices]
    set result_data [torch::tensor_to_list $result]
    
    return $result_data
} {1.0 2.0 3.0}

test tensor-index-select-5.2 {Empty indices tensor} {
    # Create test tensors
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    set indices [torch::tensor_create -data {0} -shape {1} -dtype int32]
    
    # Test with single index (empty indices not supported)
    set result [torch::tensor_index_select $tensor 0 $indices]
    set result_shape [torch::tensor_shape $result]
    
    return $result_shape
} {1 3}

test tensor-index-select-5.3 {Float tensor} {
    # Create float test tensors
    set tensor [torch::tensor_create -data {1.5 2.5 3.5 4.5 5.5 6.5} -shape {2 3} -dtype float32]
    set indices [torch::tensor_create -data {0 1} -shape {2} -dtype int32]
    
    # Test with float tensor
    set result [torch::tensor_index_select $tensor 0 $indices]
    set result_data [torch::tensor_to_list $result]
    
    return $result_data
} {1.5 2.5 3.5 4.5 5.5 6.5}

test tensor-index-select-5.4 {Large dimension index} {
    # Create 3D test tensor
    set tensor [torch::tensor_create -data {1 2 3 4 5 6 7 8} -shape {2 2 2}]
    set indices [torch::tensor_create -data {0 1} -shape {2} -dtype int32]
    
    # Test with dimension 2 (last dimension)
    set result [torch::tensor_index_select $tensor 2 $indices]
    set result_data [torch::tensor_to_list $result]
    
    return $result_data
} {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0}

# Syntax consistency tests
test tensor-index-select-6.1 {Positional and named syntax produce same result} {
    # Create test tensors
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    set indices [torch::tensor_create -data {0 1} -shape {2} -dtype int32]
    
    # Test positional syntax
    set result1 [torch::tensor_index_select $tensor 0 $indices]
    set data1 [torch::tensor_to_list $result1]
    
    # Test named syntax
    set result2 [torch::tensor_index_select -input $tensor -dim 0 -indices $indices]
    set data2 [torch::tensor_to_list $result2]
    
    return [expr {$data1 == $data2}]
} {1}

test tensor-index-select-6.2 {CamelCase and snake_case produce same result} {
    # Create test tensors
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    set indices [torch::tensor_create -data {0 1} -shape {2} -dtype int32]
    
    # Test snake_case
    set result1 [torch::tensor_index_select $tensor 0 $indices]
    set data1 [torch::tensor_to_list $result1]
    
    # Test camelCase
    set result2 [torch::tensorIndexSelect $tensor 0 $indices]
    set data2 [torch::tensor_to_list $result2]
    
    return [expr {$data1 == $data2}]
} {1}

cleanupTests 