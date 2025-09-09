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
test tensor_mode-1.1 {Basic positional syntax - 1D tensor} {
    set tensor [torch::tensor_create {1.0 2.0 1.0 3.0 1.0} float32 cpu true]
    set result [torch::tensor_mode $tensor]
    string match "tensor*" $result
} {1}

test tensor_mode-1.2 {Positional syntax with dimension - 2D tensor} {
    set tensor [torch::tensor_create {{1.0 2.0 1.0} {3.0 1.0 3.0}} float32 cpu true]
    set result [torch::tensor_mode $tensor 1]
    string match "tensor*" $result
} {1}

test tensor_mode-1.3 {Positional syntax - single most frequent value} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [torch::tensor_mode $tensor]
    string match "tensor*" $result
} {1}

;# Test cases for named parameter syntax
test tensor_mode-2.1 {Named parameter syntax with -input} {
    set tensor [torch::tensor_create {1.0 2.0 1.0 3.0 1.0} float32 cpu true]
    set result [torch::tensor_mode -input $tensor]
    string match "tensor*" $result
} {1}

test tensor_mode-2.2 {Named parameter syntax with -tensor} {
    set tensor [torch::tensor_create {1.0 2.0 1.0 3.0 1.0} float32 cpu true]
    set result [torch::tensor_mode -tensor $tensor]
    string match "tensor*" $result
} {1}

test tensor_mode-2.3 {Named parameter syntax with dimension} {
    set tensor [torch::tensor_create {{1.0 2.0 1.0} {3.0 1.0 3.0}} float32 cpu true]
    set result [torch::tensor_mode -input $tensor -dim 1]
    string match "tensor*" $result
} {1}

test tensor_mode-2.4 {Named parameter syntax with -dimension alias} {
    set tensor [torch::tensor_create {{1.0 2.0 1.0} {3.0 1.0 3.0}} float32 cpu true]
    set result [torch::tensor_mode -input $tensor -dimension 1]
    string match "tensor*" $result
} {1}

;# Test cases for camelCase alias
test tensor_mode-3.1 {CamelCase alias - basic usage} {
    set tensor [torch::tensor_create {1.0 2.0 1.0 3.0 1.0} float32 cpu true]
    set result [torch::tensorMode $tensor]
    string match "tensor*" $result
} {1}

test tensor_mode-3.2 {CamelCase alias with dimension} {
    set tensor [torch::tensor_create {{1.0 2.0 1.0} {3.0 1.0 3.0}} float32 cpu true]
    set result [torch::tensorMode $tensor 1]
    string match "tensor*" $result
} {1}

test tensor_mode-3.3 {CamelCase alias with named parameters} {
    set tensor [torch::tensor_create {1.0 2.0 1.0 3.0 1.0} float32 cpu true]
    set result [torch::tensorMode -input $tensor]
    string match "tensor*" $result
} {1}

test tensor_mode-3.4 {CamelCase alias with named parameters and dimension} {
    set tensor [torch::tensor_create {{1.0 2.0 1.0} {3.0 1.0 3.0}} float32 cpu true]
    set result [torch::tensorMode -input $tensor -dim 1]
    string match "tensor*" $result
} {1}

;# Error handling tests
test tensor_mode-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_mode invalid_tensor} result
    return $result
} {Invalid tensor name}

test tensor_mode-4.2 {Error handling - missing parameter} {
    catch {torch::tensor_mode} result
    return $result
} {Required input parameter missing}

test tensor_mode-4.3 {Error handling - unknown parameter} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    catch {torch::tensor_mode -unknown $tensor} result
    return $result
} {Unknown parameter: -unknown}

test tensor_mode-4.4 {Error handling - missing value for parameter} {
    catch {torch::tensor_mode -input} result
    return $result
} {Missing value for parameter}

test tensor_mode-4.5 {Error handling - invalid dimension value} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    catch {torch::tensor_mode -input $tensor -dim invalid} result
    return $result
} {Invalid dimension value}

;# Edge cases
test tensor_mode-5.1 {Single element tensor} {
    set tensor [torch::tensor_create {1.0} float32 cpu true]
    set result [torch::tensor_mode $tensor]
    string match "tensor*" $result
} {1}

test tensor_mode-5.2 {Large tensor} {
    set data [lrepeat 1000 1.0]
    set tensor [torch::tensor_create $data float32 cpu true]
    set result [torch::tensor_mode $tensor]
    string match "tensor*" $result
} {1}

test tensor_mode-5.3 {Empty tensor} {
    set empty_tensor [torch::tensor_create -data {} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::tensor_mode $empty_tensor} result
    string match "*" $result
} {1}

;# Consistency tests - both syntaxes should produce same results
test tensor_mode-6.1 {Consistency between positional and named syntax} {
    set tensor [torch::tensor_create {1.0 2.0 1.0 3.0 1.0} float32 cpu true]
    set pos_result [torch::tensor_mode $tensor]
    set named_result [torch::tensor_mode -input $tensor]
    return [expr {$pos_result == $named_result}]
} {0}

test tensor_mode-6.2 {Consistency between snake_case and camelCase} {
    set tensor [torch::tensor_create {1.0 2.0 1.0 3.0 1.0} float32 cpu true]
    set snake_result [torch::tensor_mode $tensor]
    set camel_result [torch::tensorMode $tensor]
    return [expr {$snake_result == $camel_result}]
} {0}

;# Mathematical correctness tests
test tensor_mode-7.1 {Mathematical correctness - most frequent value} {
    set tensor [torch::tensor_create {1.0 2.0 1.0 3.0 1.0} float32 cpu true]
    set result_handle [torch::tensor_mode $tensor]
    set result_tensor [torch::tensor_to_list $result_handle]
    expr {abs($result_tensor - 1.0) < 0.001}
} {1}

test tensor_mode-7.2 {Mathematical correctness - all values equal} {
    set tensor [torch::tensor_create {1.0 1.0 1.0 1.0} float32 cpu true]
    set result_handle [torch::tensor_mode $tensor]
    set result_tensor [torch::tensor_to_list $result_handle]
    expr {abs($result_tensor - 1.0) < 0.001}
} {1}

test tensor_mode-7.3 {Mathematical correctness - 2D tensor with dimension} {
    set tensor [torch::tensor_create {{1.0 2.0 1.0} {3.0 1.0 3.0}} float32 cpu true]
    set result_handle [torch::tensor_mode $tensor 1]
    set result_tensor [torch::tensor_to_list $result_handle]
    return [expr {[llength $result_tensor] == 2}]
} {1}

cleanupTests 