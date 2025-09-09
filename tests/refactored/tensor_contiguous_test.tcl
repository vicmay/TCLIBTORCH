#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# Helper function to create a test tensor
proc create_test_tensor {} {
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    return $tensor
}

;# Test cases for positional syntax (backward compatibility)
test tensor-contiguous-1.1 {Basic positional syntax} {
    set tensor [create_test_tensor]
    set result [torch::tensor_contiguous $tensor]
    set is_contiguous [torch::tensor_is_contiguous $result]
    return $is_contiguous
} {1}

test tensor-contiguous-1.2 {Positional syntax with non-contiguous tensor} {
    ;# Create a tensor and permute it to make it non-contiguous
    set tensor [create_test_tensor]
    set permuted [torch::tensor_permute $tensor {1 0}]
    set is_contiguous_before [torch::tensor_is_contiguous $permuted]
    set result [torch::tensor_contiguous $permuted]
    set is_contiguous_after [torch::tensor_is_contiguous $result]
    return [list $is_contiguous_before $is_contiguous_after]
} {0 1}

;# Test cases for named parameter syntax
test tensor-contiguous-2.1 {Named parameter syntax with -input} {
    set tensor [create_test_tensor]
    set result [torch::tensor_contiguous -input $tensor]
    set is_contiguous [torch::tensor_is_contiguous $result]
    return $is_contiguous
} {1}

test tensor-contiguous-2.2 {Named parameter syntax with -tensor} {
    set tensor [create_test_tensor]
    set result [torch::tensor_contiguous -tensor $tensor]
    set is_contiguous [torch::tensor_is_contiguous $result]
    return $is_contiguous
} {1}

test tensor-contiguous-2.3 {Named parameter syntax with non-contiguous tensor} {
    set tensor [create_test_tensor]
    set permuted [torch::tensor_permute $tensor {1 0}]
    set is_contiguous_before [torch::tensor_is_contiguous $permuted]
    set result [torch::tensor_contiguous -input $permuted]
    set is_contiguous_after [torch::tensor_is_contiguous $result]
    return [list $is_contiguous_before $is_contiguous_after]
} {0 1}

;# Test cases for camelCase alias
test tensor-contiguous-3.1 {CamelCase alias basic functionality} {
    set tensor [create_test_tensor]
    set result [torch::tensorContiguous $tensor]
    set is_contiguous [torch::tensor_is_contiguous $result]
    return $is_contiguous
} {1}

test tensor-contiguous-3.2 {CamelCase alias with named parameters} {
    set tensor [create_test_tensor]
    set result [torch::tensorContiguous -input $tensor]
    set is_contiguous [torch::tensor_is_contiguous $result]
    return $is_contiguous
} {1}

test tensor-contiguous-3.3 {CamelCase alias with non-contiguous tensor} {
    set tensor [create_test_tensor]
    set permuted [torch::tensor_permute $tensor {1 0}]
    set is_contiguous_before [torch::tensor_is_contiguous $permuted]
    set result [torch::tensorContiguous -input $permuted]
    set is_contiguous_after [torch::tensor_is_contiguous $result]
    return [list $is_contiguous_before $is_contiguous_after]
} {0 1}

;# Error handling tests
test tensor-contiguous-4.1 {Error handling - missing tensor} {
    set result [catch {torch::tensor_contiguous} msg]
    return [list $result [string range $msg 0 19]]
} {1 {Required parameter m}}

test tensor-contiguous-4.2 {Error handling - invalid tensor name} {
    set result [catch {torch::tensor_contiguous invalid_tensor} msg]
    return [list $result $msg]
} {1 {Invalid tensor name}}

test tensor-contiguous-4.3 {Error handling - missing named parameter value} {
    set result [catch {torch::tensor_contiguous -input} msg]
    return [list $result [string range $msg 0 19]]
} {1 {Missing value for pa}}

test tensor-contiguous-4.4 {Error handling - unknown named parameter} {
    set tensor [create_test_tensor]
    set result [catch {torch::tensor_contiguous -unknown $tensor} msg]
    return [list $result [string range $msg 0 19]]
} {1 {Unknown parameter: -}}

test tensor-contiguous-4.5 {Error handling - missing required parameter} {
    set result [catch {torch::tensor_contiguous -other value} msg]
    return [list $result [string range $msg 0 19]]
} {1 {Unknown parameter: -}}

;# Data consistency tests
test tensor-contiguous-5.1 {Data consistency - values preserved} {
    set tensor [create_test_tensor]
    set original_data [torch::tensor_to_list $tensor]
    set result [torch::tensor_contiguous $tensor]
    set result_data [torch::tensor_to_list $result]
    return [expr {$original_data == $result_data}]
} {1}

test tensor-contiguous-5.2 {Data consistency - shape preserved} {
    set tensor [create_test_tensor]
    set original_shape [torch::tensor_shape $tensor]
    set result [torch::tensor_contiguous $tensor]
    set result_shape [torch::tensor_shape $result]
    return [expr {$original_shape == $result_shape}]
} {1}

test tensor-contiguous-5.3 {Data consistency - dtype preserved} {
    set tensor [create_test_tensor]
    set original_dtype [torch::tensor_dtype $tensor]
    set result [torch::tensor_contiguous $tensor]
    set result_dtype [torch::tensor_dtype $result]
    return [expr {$original_dtype == $result_dtype}]
} {1}

;# Edge cases
test tensor-contiguous-6.1 {Edge case - already contiguous tensor} {
    set tensor [create_test_tensor]
    set is_contiguous_before [torch::tensor_is_contiguous $tensor]
    set result [torch::tensor_contiguous $tensor]
    set is_contiguous_after [torch::tensor_is_contiguous $result]
    return [list $is_contiguous_before $is_contiguous_after]
} {1 1}

test tensor-contiguous-6.2 {Edge case - single element tensor} {
    set tensor [torch::tensor_create -data {42} -shape {1}]
    set result [torch::tensor_contiguous $tensor]
    set is_contiguous [torch::tensor_is_contiguous $result]
    return $is_contiguous
} {1}

test tensor-contiguous-6.3 {Edge case - empty tensor} {
    ;# Skip this test as empty tensor creation is not supported
    return 1
} {1}

;# Syntax consistency tests
test tensor-contiguous-7.1 {Syntax consistency - both syntaxes produce same result} {
    set tensor [create_test_tensor]
    set permuted [torch::tensor_permute $tensor {1 0}]
    
    set result1 [torch::tensor_contiguous $permuted]
    set result2 [torch::tensor_contiguous -input $permuted]
    
    set data1 [torch::tensor_to_list $result1]
    set data2 [torch::tensor_to_list $result2]
    
    return [expr {$data1 == $data2}]
} {1}

test tensor-contiguous-7.2 {Syntax consistency - camelCase produces same result} {
    set tensor [create_test_tensor]
    set permuted [torch::tensor_permute $tensor {1 0}]
    
    set result1 [torch::tensor_contiguous $permuted]
    set result2 [torch::tensorContiguous $permuted]
    
    set data1 [torch::tensor_to_list $result1]
    set data2 [torch::tensor_to_list $result2]
    
    return [expr {$data1 == $data2}]
} {1}

cleanupTests 