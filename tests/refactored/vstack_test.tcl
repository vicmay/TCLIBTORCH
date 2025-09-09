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

;# Test cases for positional syntax
test vstack-1.1 {Basic positional syntax with list} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {2 3}]
    set result [torch::vstack [list $tensor1 $tensor2]]
    set shape [torch::tensor_shape $result]
    return $shape
} {4 3}

test vstack-1.2 {Positional syntax with multiple arguments} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {2 3}]
    set result [torch::vstack $tensor1 $tensor2]
    set shape [torch::tensor_shape $result]
    return $shape
} {4 3}

test vstack-1.3 {Positional syntax with 3 tensors} {
    set tensor1 [torch::ones -shape {1 4}]
    set tensor2 [torch::ones -shape {1 4}]
    set tensor3 [torch::ones -shape {1 4}]
    set result [torch::vstack $tensor1 $tensor2 $tensor3]
    set shape [torch::tensor_shape $result]
    return $shape
} {3 4}

;# Test cases for named parameter syntax
test vstack-2.1 {Named parameter syntax with -tensors} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {2 3}]
    set result [torch::vstack -tensors [list $tensor1 $tensor2]]
    set shape [torch::tensor_shape $result]
    return $shape
} {4 3}

test vstack-2.2 {Named parameter syntax with -inputs} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {2 3}]
    set result [torch::vstack -inputs [list $tensor1 $tensor2]]
    set shape [torch::tensor_shape $result]
    return $shape
} {4 3}

test vstack-2.3 {Named parameter syntax with multiple tensors} {
    set tensor1 [torch::ones -shape {1 4}]
    set tensor2 [torch::ones -shape {1 4}]
    set tensor3 [torch::ones -shape {1 4}]
    set result [torch::vstack -tensors [list $tensor1 $tensor2 $tensor3]]
    set shape [torch::tensor_shape $result]
    return $shape
} {3 4}

;# Test cases for camelCase alias
test vstack-3.1 {CamelCase alias with list} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {2 3}]
    set result [torch::vStack [list $tensor1 $tensor2]]
    set shape [torch::tensor_shape $result]
    return $shape
} {4 3}

test vstack-3.2 {CamelCase alias with multiple arguments} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {2 3}]
    set result [torch::vStack $tensor1 $tensor2]
    set shape [torch::tensor_shape $result]
    return $shape
} {4 3}

test vstack-3.3 {CamelCase alias with named parameters} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {2 3}]
    set result [torch::vStack -tensors [list $tensor1 $tensor2]]
    set shape [torch::tensor_shape $result]
    return $shape
} {4 3}

;# Error handling tests
test vstack-4.1 {Error handling - missing tensors} {
    catch {torch::vstack} result
    return [string match "*tensor_list*" $result]
} {1}

test vstack-4.2 {Error handling - empty list} {
    catch {torch::vstack {}} result
    return [string match "*Missing required parameter*" $result]
} {1}

test vstack-4.3 {Error handling - invalid tensor handle} {
    catch {torch::vstack invalid_handle} result
    return [string match "*Error in vstack*" $result]
} {1}

test vstack-4.4 {Error handling - unknown named parameter} {
    set tensor [torch::ones -shape {2 3}]
    catch {torch::vstack -invalid $tensor} result
    return [string match "*Unknown parameter*" $result]
} {1}

;# Edge cases
test vstack-5.1 {Edge case - single tensor} {
    set tensor [torch::ones -shape {2 3}]
    set result [torch::vstack $tensor]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 3}

test vstack-5.2 {Edge case - different shapes} {
    set tensor1 [torch::ones -shape {1 3}]
    set tensor2 [torch::ones -shape {2 3}]
    set result [torch::vstack $tensor1 $tensor2]
    set shape [torch::tensor_shape $result]
    return $shape
} {3 3}

;# Verify both syntaxes produce same results
test vstack-6.1 {Consistency between positional and named syntax} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {2 3}]
    set result1 [torch::vstack $tensor1 $tensor2]
    set result2 [torch::vstack -tensors [list $tensor1 $tensor2]]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    return [expr {$shape1 eq $shape2}]
} {1}

;# Data integrity tests
test vstack-7.1 {Data integrity - values preserved} {
    set tensor1 [torch::arange -start 0 -end 6 -dtype float32]
    set reshaped1 [torch::tensor_reshape $tensor1 {2 3}]
    set tensor2 [torch::arange -start 6 -end 12 -dtype float32]
    set reshaped2 [torch::tensor_reshape $tensor2 {2 3}]
    set result [torch::vstack $reshaped1 $reshaped2]
    set shape [torch::tensor_shape $result]
    return $shape
} {4 3}

;# Different tensor types
test vstack-8.1 {Different tensor types - float32} {
    set tensor1 [torch::ones -shape {2 3} -dtype float32]
    set tensor2 [torch::ones -shape {2 3} -dtype float32]
    set result [torch::vstack -tensors [list $tensor1 $tensor2]]
    set dtype [torch::tensor_dtype $result]
    return $dtype
} {Float32}

test vstack-8.2 {Different tensor types - int64} {
    set tensor1 [torch::ones -shape {2 3} -dtype int64]
    set tensor2 [torch::ones -shape {2 3} -dtype int64]
    set result [torch::vstack -tensors [list $tensor1 $tensor2]]
    set dtype [torch::tensor_dtype $result]
    return $dtype
} {Int64}

cleanupTests 