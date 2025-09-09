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
test tensor_median-1.1 {Basic positional syntax - 1D tensor} {
    set tensor [torch::tensor_create {1.0 3.0 2.0 5.0 4.0} float32 cpu true]
    set result [torch::tensor_median $tensor]
    string match "tensor*" $result
} {1}

test tensor_median-1.2 {Positional syntax with dimension - 2D tensor} {
    set tensor [torch::tensor_create {{1.0 3.0 2.0} {5.0 4.0 6.0}} float32 cpu true]
    set result [torch::tensor_median $tensor 1]
    string match "tensor*" $result
} {1}

test tensor_median-1.3 {Positional syntax - even number of elements} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [torch::tensor_median $tensor]
    string match "tensor*" $result
} {1}

;# Test cases for named parameter syntax
test tensor_median-2.1 {Named parameter syntax with -input} {
    set tensor [torch::tensor_create {1.0 3.0 2.0 5.0 4.0} float32 cpu true]
    set result [torch::tensor_median -input $tensor]
    string match "tensor*" $result
} {1}

test tensor_median-2.2 {Named parameter syntax with -tensor} {
    set tensor [torch::tensor_create {1.0 3.0 2.0 5.0 4.0} float32 cpu true]
    set result [torch::tensor_median -tensor $tensor]
    string match "tensor*" $result
} {1}

test tensor_median-2.3 {Named parameter syntax with dimension} {
    set tensor [torch::tensor_create {{1.0 3.0 2.0} {5.0 4.0 6.0}} float32 cpu true]
    set result [torch::tensor_median -input $tensor -dim 1]
    string match "tensor*" $result
} {1}

test tensor_median-2.4 {Named parameter syntax with -dimension alias} {
    set tensor [torch::tensor_create {{1.0 3.0 2.0} {5.0 4.0 6.0}} float32 cpu true]
    set result [torch::tensor_median -input $tensor -dimension 1]
    string match "tensor*" $result
} {1}

;# Test cases for camelCase alias
test tensor_median-3.1 {CamelCase alias - basic usage} {
    set tensor [torch::tensor_create {1.0 3.0 2.0 5.0 4.0} float32 cpu true]
    set result [torch::tensorMedian $tensor]
    string match "tensor*" $result
} {1}

test tensor_median-3.2 {CamelCase alias with dimension} {
    set tensor [torch::tensor_create {{1.0 3.0 2.0} {5.0 4.0 6.0}} float32 cpu true]
    set result [torch::tensorMedian $tensor 1]
    string match "tensor*" $result
} {1}

test tensor_median-3.3 {CamelCase alias with named parameters} {
    set tensor [torch::tensor_create {1.0 3.0 2.0 5.0 4.0} float32 cpu true]
    set result [torch::tensorMedian -input $tensor]
    string match "tensor*" $result
} {1}

test tensor_median-3.4 {CamelCase alias with named parameters and dimension} {
    set tensor [torch::tensor_create {{1.0 3.0 2.0} {5.0 4.0 6.0}} float32 cpu true]
    set result [torch::tensorMedian -input $tensor -dim 1]
    string match "tensor*" $result
} {1}

;# Error handling tests
test tensor_median-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_median invalid_tensor} result
    return $result
} {Invalid tensor name}

test tensor_median-4.2 {Error handling - missing parameter} {
    catch {torch::tensor_median} result
    return $result
} {Required input parameter missing}

test tensor_median-4.3 {Error handling - unknown parameter} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    catch {torch::tensor_median -unknown $tensor} result
    return $result
} {Unknown parameter: -unknown}

test tensor_median-4.4 {Error handling - missing value for parameter} {
    catch {torch::tensor_median -input} result
    return $result
} {Missing value for parameter}

test tensor_median-4.5 {Error handling - invalid dimension value} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    catch {torch::tensor_median -input $tensor -dim invalid} result
    return $result
} {Invalid dimension value}

;# Edge cases
test tensor_median-5.1 {Single element tensor} {
    set tensor [torch::tensor_create {1.0} float32 cpu true]
    set result [torch::tensor_median $tensor]
    string match "tensor*" $result
} {1}

test tensor_median-5.2 {Large tensor} {
    set data [lrepeat 1000 1.0]
    set tensor [torch::tensor_create $data float32 cpu true]
    set result [torch::tensor_median $tensor]
    string match "tensor*" $result
} {1}

test tensor_median-5.3 {Empty tensor} {
    set empty_tensor [torch::tensor_create -data {} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::tensor_median $empty_tensor} result
    string match "*" $result
} {1}

;# Consistency tests - both syntaxes should produce same results
test tensor_median-6.1 {Consistency between positional and named syntax} {
    set tensor [torch::tensor_create {1.0 3.0 2.0 5.0 4.0} float32 cpu true]
    set pos_result [torch::tensor_median $tensor]
    set named_result [torch::tensor_median -input $tensor]
    return [expr {$pos_result == $named_result}]
} {0}

test tensor_median-6.2 {Consistency between snake_case and camelCase} {
    set tensor [torch::tensor_create {1.0 3.0 2.0 5.0 4.0} float32 cpu true]
    set snake_result [torch::tensor_median $tensor]
    set camel_result [torch::tensorMedian $tensor]
    return [expr {$snake_result == $camel_result}]
} {0}

;# Mathematical correctness tests
test tensor_median-7.1 {Mathematical correctness - odd number of elements} {
    set tensor [torch::tensor_create {1.0 3.0 2.0 5.0 4.0} float32 cpu true]
    set result_handle [torch::tensor_median $tensor]
    set result_tensor [torch::tensor_to_list $result_handle]
    return [expr {abs($result_tensor - 3.0) < 0.001}]
} {1}

test tensor_median-7.2 {Mathematical correctness - even number of elements} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result_handle [torch::tensor_median $tensor]
    set result_tensor [torch::tensor_to_list $result_handle]
    expr {abs($result_tensor - 2.0) < 0.001}
} {1}

test tensor_median-7.3 {Mathematical correctness - 2D tensor with dimension} {
    set tensor [torch::tensor_create {{1.0 3.0 2.0} {5.0 4.0 6.0}} float32 cpu true]
    set result_handle [torch::tensor_median $tensor 1]
    set result_tensor [torch::tensor_to_list $result_handle]
    return [expr {[llength $result_tensor] == 2}]
} {1}

cleanupTests 