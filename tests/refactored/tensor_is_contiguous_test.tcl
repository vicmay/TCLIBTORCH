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

;# Create test tensors
set contiguous_tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
set non_contiguous_tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
set transposed_tensor [torch::tensor_permute $non_contiguous_tensor {1 0}]

;# Test cases for positional syntax
test tensor-is-contiguous-1.1 {Basic positional syntax - contiguous tensor} {
    set result [torch::tensor_is_contiguous $contiguous_tensor]
    return $result
} {1}

test tensor-is-contiguous-1.2 {Positional syntax - non-contiguous tensor} {
    set result [torch::tensor_is_contiguous $transposed_tensor]
    return $result
} {0}

test tensor-is-contiguous-1.3 {Positional syntax - 2D tensor} {
    set result [torch::tensor_is_contiguous $non_contiguous_tensor]
    return $result
} {1}

;# Test cases for named syntax
test tensor-is-contiguous-2.1 {Named parameter syntax with -tensor} {
    set result [torch::tensor_is_contiguous -tensor $contiguous_tensor]
    return $result
} {1}

test tensor-is-contiguous-2.2 {Named syntax with -input alias} {
    set result [torch::tensor_is_contiguous -input $contiguous_tensor]
    return $result
} {1}

test tensor-is-contiguous-2.3 {Named syntax - non-contiguous tensor} {
    set result [torch::tensor_is_contiguous -tensor $transposed_tensor]
    return $result
} {0}

;# Test cases for camelCase alias
test tensor-is-contiguous-3.1 {CamelCase alias - contiguous tensor} {
    set result [torch::tensorIsContiguous $contiguous_tensor]
    return $result
} {1}

test tensor-is-contiguous-3.2 {CamelCase alias - non-contiguous tensor} {
    set result [torch::tensorIsContiguous $transposed_tensor]
    return $result
} {0}

test tensor-is-contiguous-3.3 {CamelCase alias with named parameters} {
    set result [torch::tensorIsContiguous -tensor $contiguous_tensor]
    return $result
} {1}

;# Test cases for error handling
test tensor-is-contiguous-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_is_contiguous nonexistent} result
    string match "*Invalid tensor name*" $result
} {1}

test tensor-is-contiguous-4.2 {Error handling - missing tensor parameter} {
    catch {torch::tensor_is_contiguous} result
    string match "*Required tensor parameter missing*" $result
} {1}

test tensor-is-contiguous-4.3 {Error handling - unknown named parameter} {
    catch {torch::tensor_is_contiguous -unknown value} result
    string match "*Unknown parameter*" $result
} {1}

test tensor-is-contiguous-4.4 {Error handling - missing value for parameter} {
    catch {torch::tensor_is_contiguous -tensor} result
    string match "*Missing value for parameter*" $result
} {1}

;# Test cases for edge cases
test tensor-is-contiguous-5.1 {Edge case - single element tensor} {
    set single [torch::tensor_create {1.0} float32 cpu true]
    set result [torch::tensor_is_contiguous $single]
    return $result
} {1}

test tensor-is-contiguous-5.2 {Edge case - zero tensor} {
    set zero [torch::tensor_create {0.0 0.0 0.0 0.0} float32 cpu true]
    set result [torch::tensor_is_contiguous $zero]
    return $result
} {1}

test tensor-is-contiguous-5.3 {Edge case - empty tensor} {
    set empty_tensor [torch::tensor_create -data {} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::tensor_is_contiguous $empty_tensor]
    return $result
} {1}

test tensor-is-contiguous-5.4 {Edge case - 3D tensor} {
    set tensor_2d [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu false]
    set tensor_3d [torch::tensor_stack [list $tensor_2d $tensor_2d] 0]
    set result [torch::tensor_is_contiguous $tensor_3d]
    return $result
} {1}

;# Test cases for syntax consistency
test tensor-is-contiguous-6.1 {Syntax consistency - positional vs named} {
    set result1 [torch::tensor_is_contiguous $contiguous_tensor]
    set result2 [torch::tensor_is_contiguous -tensor $contiguous_tensor]
    list $result1 $result2
} {1 1}

test tensor-is-contiguous-6.2 {Syntax consistency - snake_case vs camelCase} {
    set result1 [torch::tensor_is_contiguous $contiguous_tensor]
    set result2 [torch::tensorIsContiguous $contiguous_tensor]
    list $result1 $result2
} {1 1}

test tensor-is-contiguous-6.3 {Syntax consistency - different tensor types} {
    set result1 [torch::tensor_is_contiguous $contiguous_tensor]
    set result2 [torch::tensor_is_contiguous $transposed_tensor]
    list $result1 $result2
} {1 0}

;# Test cases for tensor operations that affect contiguity
test tensor-is-contiguous-7.1 {Contiguity after reshape} {
    set reshaped [torch::tensor_reshape $contiguous_tensor {2 2}]
    set result [torch::tensor_is_contiguous $reshaped]
    return $result
} {1}

test tensor-is-contiguous-7.2 {Contiguity after permute} {
    set permuted [torch::tensor_permute $contiguous_tensor {0}]
    set result [torch::tensor_is_contiguous $permuted]
    return $result
} {1}

test tensor-is-contiguous-7.3 {Contiguity after transpose} {
    set transposed [torch::tensor_permute $non_contiguous_tensor {1 0}]
    set result [torch::tensor_is_contiguous $transposed]
    return $result
} {0}

;# Test cases for mathematical correctness
test tensor-is-contiguous-8.1 {Mathematical correctness - contiguous tensor} {
    set result [torch::tensor_is_contiguous $contiguous_tensor]
    return [expr {$result == 1}]
} {1}

test tensor-is-contiguous-8.2 {Mathematical correctness - non-contiguous tensor} {
    set result [torch::tensor_is_contiguous $transposed_tensor]
    return [expr {$result == 0}]
} {1}

;# Test cases for different data types
test tensor-is-contiguous-9.1 {Different data types - float32} {
    set float_tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    set result [torch::tensor_is_contiguous $float_tensor]
    return $result
} {1}

test tensor-is-contiguous-9.2 {Different data types - int32} {
    set int_tensor [torch::tensor_create {1 2 3} int32 cpu false]
    set result [torch::tensor_is_contiguous $int_tensor]
    return $result
} {1}

;# Cleanup after all tests
cleanupTests 