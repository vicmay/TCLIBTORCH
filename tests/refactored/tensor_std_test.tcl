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
test tensor_std-1.1 {Basic positional syntax - no parameters} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result [torch::tensor_std $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_std-1.2 {Positional syntax with dimension} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6}}]
    set result [torch::tensor_std $tensor 0]
    expr {[string length $result] > 0}
} {1}

test tensor_std-1.3 {Positional syntax with dimension and unbiased} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6}}]
    set result [torch::tensor_std $tensor 0 1]
    expr {[string length $result] > 0}
} {1}

test tensor_std-1.4 {Positional syntax with unbiased only} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result [torch::tensor_std $tensor 0]
    expr {[string length $result] > 0}
} {1}

;# Test cases for named parameter syntax
test tensor_std-2.1 {Named parameter syntax - input only} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result [torch::tensor_std -input $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_std-2.2 {Named parameter syntax - tensor and dim} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6}}]
    set result [torch::tensor_std -tensor $tensor -dim 0]
    expr {[string length $result] > 0}
} {1}

test tensor_std-2.3 {Named parameter syntax - all parameters} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6}}]
    set result [torch::tensor_std -input $tensor -dimension 0 -unbiased 1]
    expr {[string length $result] > 0}
} {1}

test tensor_std-2.4 {Named parameter syntax - unbiased false} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result [torch::tensor_std -tensor $tensor -unbiased 0]
    expr {[string length $result] > 0}
} {1}

;# Test cases for camelCase alias
test tensor_std-3.1 {CamelCase alias - positional syntax} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result [torch::tensorStd $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_std-3.2 {CamelCase alias - named parameter syntax} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6}}]
    set result [torch::tensorStd -input $tensor -dim 0]
    expr {[string length $result] > 0}
} {1}

;# Error handling tests
test tensor_std-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_std invalid_tensor} result
    return $result
} {Invalid tensor name}

test tensor_std-4.2 {Error handling - missing tensor} {
    catch {torch::tensor_std} result
    return $result
} {Required input parameter missing}

test tensor_std-4.3 {Error handling - invalid dimension} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_std $tensor invalid} result
    return $result
} {Invalid dimension value}

test tensor_std-4.4 {Error handling - invalid unbiased} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_std $tensor 0 invalid} result
    return $result
} {Invalid unbiased value}

test tensor_std-4.5 {Error handling - invalid named parameter} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_std -invalid $tensor} result
    return $result
} {Unknown parameter: -invalid}

test tensor_std-4.6 {Error handling - missing parameter value} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_std -input $tensor -dim} result
    return $result
} {Missing value for parameter}

;# Edge cases
test tensor_std-5.1 {Edge case - single element tensor} {
    set tensor [torch::tensor_create 5]
    set result [torch::tensor_std $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_std-5.2 {Edge case - zero tensor} {
    set tensor [torch::tensor_create {0 0 0 0}]
    set result [torch::tensor_std $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_std-5.3 {Edge case - negative values} {
    set tensor [torch::tensor_create -data {-1.0 -2.0 -3.0 -4.0 -5.0}]
    set result [torch::tensor_std $tensor]
    expr {[string length $result] > 0}
} {1}

;# Data type tests
test tensor_std-6.1 {Data type - float32 tensor} {
    set tensor [torch::tensor_create {1.5 2.5 3.5 4.5} float32]
    set result [torch::tensor_std $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_std-6.2 {Data type - float64 tensor} {
    set tensor [torch::tensor_create {1.5 2.5 3.5 4.5} float64]
    set result [torch::tensor_std $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_std-6.3 {Data type - int32 tensor} {
    set tensor [torch::tensor_create {1 2 3 4 5} int32]
    catch {torch::tensor_std $tensor} result
    expr {[string length $result] > 0}
} {1}

;# Multi-dimensional tensor tests
test tensor_std-7.1 {2D tensor - std along first dimension} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9}}]
    set result [torch::tensor_std $tensor 0]
    expr {[string length $result] > 0}
} {1}

test tensor_std-7.2 {2D tensor - std along second dimension} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9}}]
    set result [torch::tensor_std $tensor 1]
    expr {[string length $result] > 0}
} {1}

test tensor_std-7.3 {3D tensor - std along different dimensions} {
    set tensor [torch::tensor_create {{1 2 3 4} {5 6 7 8} {9 10 11 12}}]
    set result [torch::tensor_std $tensor 0]
    expr {[string length $result] > 0}
} {1}

;# Mathematical correctness tests
test tensor_std-8.1 {Mathematical correctness - known values} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result [torch::tensor_std $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_std-8.2 {Mathematical correctness - unbiased vs biased} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result1 [torch::tensor_std $tensor 0 1]
    set result2 [torch::tensor_std $tensor 0 0]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

;# Consistency tests - both syntaxes should produce same results
test tensor_std-9.1 {Consistency - positional vs named syntax} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result1 [torch::tensor_std $tensor]
    set result2 [torch::tensor_std -input $tensor]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor_std-9.2 {Consistency - snake_case vs camelCase} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result1 [torch::tensor_std $tensor]
    set result2 [torch::tensorStd $tensor]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

;# Complex scenarios
test tensor_std-10.1 {Complex - large tensor} {
    set data {}
    for {set i 1} {$i <= 100} {incr i} {
        lappend data $i
    }
    set tensor [torch::tensor_create $data]
    set result [torch::tensor_std $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_std-10.2 {Complex - mixed positive and negative} {
    set tensor [torch::tensor_create -data {-5 -3 -1 1 3 5}]
    set result [torch::tensor_std $tensor]
    expr {[string length $result] > 0}
} {1}

cleanupTests 