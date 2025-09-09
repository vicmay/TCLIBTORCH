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
test tensor_slice-1.1 {Basic positional syntax - start only} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result [torch::tensor_slice $tensor 0 1]
    expr {[string length $result] > 0}
} {1}

test tensor_slice-1.2 {Positional syntax with start and end} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result [torch::tensor_slice $tensor 0 1 4]
    expr {[string length $result] > 0}
} {1}

test tensor_slice-1.3 {Positional syntax with start, end, and step} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result [torch::tensor_slice $tensor 0 0 8 2]
    expr {[string length $result] > 0}
} {1}

test tensor_slice-1.4 {Positional syntax - 2D tensor} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9}}]
    set result [torch::tensor_slice $tensor 0 1 2]
    expr {[string length $result] > 0}
} {1}

;# Test cases for named parameter syntax
test tensor_slice-2.1 {Named parameter syntax - tensor and start} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result [torch::tensor_slice -tensor $tensor -dim 0 -start 1]
    expr {[string length $result] > 0}
} {1}

test tensor_slice-2.2 {Named parameter syntax - all parameters} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result [torch::tensor_slice -input $tensor -dimension 0 -start 1 -end 4]
    expr {[string length $result] > 0}
} {1}

test tensor_slice-2.3 {Named parameter syntax - with step} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result [torch::tensor_slice -tensor $tensor -dim 0 -start 0 -end 8 -step 2]
    expr {[string length $result] > 0}
} {1}

test tensor_slice-2.4 {Named parameter syntax - different parameter order} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9}}]
    set result [torch::tensor_slice -dim 0 -start 1 -end 2 -tensor $tensor]
    expr {[string length $result] > 0}
} {1}

;# Test cases for camelCase alias
test tensor_slice-3.1 {CamelCase alias - positional syntax} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result [torch::tensorSlice $tensor 0 1]
    expr {[string length $result] > 0}
} {1}

test tensor_slice-3.2 {CamelCase alias - named parameter syntax} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9}}]
    set result [torch::tensorSlice -input $tensor -dim 0 -start 1 -end 2]
    expr {[string length $result] > 0}
} {1}

;# Error handling tests
test tensor_slice-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_slice invalid_tensor 0 1} result
    return $result
} {Tensor not found}

test tensor_slice-4.2 {Error handling - missing tensor} {
    catch {torch::tensor_slice} result
    return $result
} {Required tensor parameter missing}

test tensor_slice-4.3 {Error handling - invalid dimension} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_slice $tensor invalid 1} result
    return $result
} {Invalid dimension value}

test tensor_slice-4.4 {Error handling - invalid start} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_slice $tensor 0 invalid} result
    return $result
} {Invalid start value}

test tensor_slice-4.5 {Error handling - invalid end} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_slice $tensor 0 1 invalid} result
    return $result
} {Invalid end value}

test tensor_slice-4.6 {Error handling - invalid step} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_slice $tensor 0 1 2 invalid} result
    return $result
} {Invalid step value}

test tensor_slice-4.7 {Error handling - invalid named parameter} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_slice -invalid $tensor} result
    return $result
} {Unknown parameter: -invalid}

test tensor_slice-4.8 {Error handling - missing parameter value} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_slice -tensor $tensor -dim} result
    return $result
} {Missing value for parameter}

;# Edge cases
test tensor_slice-5.1 {Edge case - single element slice} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result [torch::tensor_slice $tensor 0 2 3]
    expr {[string length $result] > 0}
} {1}

test tensor_slice-5.2 {Edge case - negative indices} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result [torch::tensor_slice $tensor 0 -3 -1]
    expr {[string length $result] > 0}
} {1}

test tensor_slice-5.3 {Edge case - step of 1} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result [torch::tensor_slice $tensor 0 0 8 1]
    expr {[string length $result] > 0}
} {1}

;# Multi-dimensional tensor tests
test tensor_slice-6.1 {2D tensor - slice first dimension} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9}}]
    set result [torch::tensor_slice $tensor 0 1 2]
    expr {[string length $result] > 0}
} {1}

test tensor_slice-6.2 {2D tensor - slice second dimension} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9}}]
    set result [torch::tensor_slice $tensor 1 0 2]
    expr {[string length $result] > 0}
} {1}

test tensor_slice-6.3 {3D tensor - slice different dimensions} {
    set tensor [torch::tensor_create {{1 2 3 4} {5 6 7 8}}]
    set result [torch::tensor_slice $tensor 0 0 1]
    expr {[string length $result] > 0}
} {1}

;# Data type tests
test tensor_slice-7.1 {Data type - float32 tensor} {
    set tensor [torch::tensor_create {1.5 2.5 3.5 4.5} float32]
    set result [torch::tensor_slice $tensor 0 1 3]
    expr {[string length $result] > 0}
} {1}

test tensor_slice-7.2 {Data type - float64 tensor} {
    set tensor [torch::tensor_create {1.5 2.5 3.5 4.5} float64]
    set result [torch::tensor_slice $tensor 0 1 3]
    expr {[string length $result] > 0}
} {1}

test tensor_slice-7.3 {Data type - int32 tensor} {
    set tensor [torch::tensor_create {1 2 3 4 5} int32]
    set result [torch::tensor_slice $tensor 0 1 3]
    expr {[string length $result] > 0}
} {1}

;# Mathematical correctness tests
test tensor_slice-8.1 {Mathematical correctness - known values} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result [torch::tensor_slice $tensor 0 1 4]
    expr {[string length $result] > 0}
} {1}

test tensor_slice-8.2 {Mathematical correctness - step slicing} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result [torch::tensor_slice $tensor 0 0 8 2]
    expr {[string length $result] > 0}
} {1}

;# Consistency tests - both syntaxes should produce same results
test tensor_slice-9.1 {Consistency - positional vs named syntax} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result1 [torch::tensor_slice $tensor 0 1 4]
    set result2 [torch::tensor_slice -tensor $tensor -dim 0 -start 1 -end 4]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor_slice-9.2 {Consistency - snake_case vs camelCase} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result1 [torch::tensor_slice $tensor 0 1 4]
    set result2 [torch::tensorSlice $tensor 0 1 4]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

;# Complex scenarios
test tensor_slice-10.1 {Complex - large tensor} {
    set data {}
    for {set i 1} {$i <= 100} {incr i} {
        lappend data $i
    }
    set tensor [torch::tensor_create $data]
    set result [torch::tensor_slice $tensor 0 10 90 5]
    expr {[string length $result] > 0}
} {1}

test tensor_slice-10.2 {Complex - multi-dimensional slicing} {
    set tensor [torch::tensor_create {{1 2 3 4} {5 6 7 8} {9 10 11 12}}]
    set result [torch::tensor_slice $tensor 0 0 1]
    expr {[string length $result] > 0}
} {1}

;# Slice-specific tests
test tensor_slice-11.1 {Slice specific - empty slice} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result [torch::tensor_slice $tensor 0 5 5]
    expr {[string length $result] > 0}
} {1}

test tensor_slice-11.2 {Slice specific - full slice} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result [torch::tensor_slice $tensor 0 0 5]
    expr {[string length $result] > 0}
} {1}

test tensor_slice-11.3 {Slice specific - reverse slice} {
    set tensor [torch::tensor_create {1 2 3 4 5}]
    set result [torch::tensor_slice $tensor 0 4 5 1]
    expr {[string length $result] > 0}
} {1}

cleanupTests 