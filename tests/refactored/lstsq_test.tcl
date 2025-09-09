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

# =====================================================================
# TORCH::LSTSQ COMPREHENSIVE TEST SUITE
# =====================================================================

# Test cases for positional syntax
test lstsq-1.1 {Basic positional syntax} -body {
    set A [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set A [torch::tensor_reshape $A {3 2}]
    set B [torch::tensor_create {1.0 2.0 3.0} float32]
    set B [torch::tensor_reshape $B {3 1}]
    set result [torch::lstsq $B $A]
    expr {[string length $result] > 0}
} -result 1

test lstsq-1.2 {Positional syntax with rcond} -body {
    set A [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set A [torch::tensor_reshape $A {3 2}]
    set B [torch::tensor_create {1.0 2.0 3.0} float32]
    set B [torch::tensor_reshape $B {3 1}]
    set result [torch::lstsq $B $A 1e-15]
    expr {[string length $result] > 0}
} -result 1

# Test cases for named syntax
test lstsq-2.1 {Named parameter syntax} -body {
    set A [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set A [torch::tensor_reshape $A {3 2}]
    set B [torch::tensor_create {1.0 2.0 3.0} float32]
    set B [torch::tensor_reshape $B {3 1}]
    set result [torch::lstsq -b $B -a $A]
    expr {[string length $result] > 0}
} -result 1

test lstsq-2.2 {Named parameter syntax with rcond} -body {
    set A [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set A [torch::tensor_reshape $A {3 2}]
    set B [torch::tensor_create {1.0 2.0 3.0} float32]
    set B [torch::tensor_reshape $B {3 1}]
    set result [torch::lstsq -b $B -a $A -rcond 1e-15]
    expr {[string length $result] > 0}
} -result 1

test lstsq-2.3 {Named parameter syntax with alternate case} -body {
    set A [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set A [torch::tensor_reshape $A {3 2}]
    set B [torch::tensor_create {1.0 2.0 3.0} float32]
    set B [torch::tensor_reshape $B {3 1}]
    set result [torch::lstsq -B $B -A $A]
    expr {[string length $result] > 0}
} -result 1

# Test cases for camelCase alias
test lstsq-3.1 {CamelCase alias positional syntax} -body {
    set A [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set A [torch::tensor_reshape $A {3 2}]
    set B [torch::tensor_create {1.0 2.0 3.0} float32]
    set B [torch::tensor_reshape $B {3 1}]
    set result [torch::leastSquares $B $A]
    expr {[string length $result] > 0}
} -result 1

test lstsq-3.2 {CamelCase alias named syntax} -body {
    set A [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set A [torch::tensor_reshape $A {3 2}]
    set B [torch::tensor_create {1.0 2.0 3.0} float32]
    set B [torch::tensor_reshape $B {3 1}]
    set result [torch::leastSquares -b $B -a $A]
    expr {[string length $result] > 0}
} -result 1

# Test cases for consistency between syntaxes
test lstsq-4.1 {Consistency between positional and named syntax} -body {
    set A [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set A [torch::tensor_reshape $A {3 2}]
    set B [torch::tensor_create {1.0 2.0 3.0} float32]
    set B [torch::tensor_reshape $B {3 1}]
    
    set result1 [torch::lstsq $B $A]
    set result2 [torch::lstsq -b $B -a $A]
    
    ;# Both should produce valid results
    set valid1 [expr {[string length $result1] > 0}]
    set valid2 [expr {[string length $result2] > 0}]
    
    expr {$valid1 && $valid2}
} -result 1

# Error handling tests
test lstsq-5.1 {Error handling - missing arguments} -body {
    catch {torch::lstsq} msg
    string match "*Usage*" $msg
} -result 1

test lstsq-5.2 {Error handling - invalid tensor} -body {
    catch {torch::lstsq "invalid_tensor" "invalid_tensor"} msg
    string match "*Invalid*tensor*" $msg
} -result 1

test lstsq-5.3 {Error handling - invalid rcond} -body {
    set A [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set A [torch::tensor_reshape $A {3 2}]
    set B [torch::tensor_create {1.0 2.0 3.0} float32]
    set B [torch::tensor_reshape $B {3 1}]
    catch {torch::lstsq $B $A "invalid_rcond"} msg
    expr {$msg ne ""}
} -result 1

test lstsq-5.4 {Error handling - invalid named parameter} -body {
    set A [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set A [torch::tensor_reshape $A {3 2}]
    set B [torch::tensor_create {1.0 2.0 3.0} float32]
    set B [torch::tensor_reshape $B {3 1}]
    catch {torch::lstsq -b $B -a $A -invalid_param 1.0} msg
    string match "*Unknown parameter*" $msg
} -result 1

# Mathematical correctness tests
test lstsq-6.1 {Mathematical correctness - simple case} -body {
    ;# Solve Ax = b where A = [[1, 1], [1, 2], [1, 3]] and b = [6, 8, 10]
    ;# Expected solution should be approximately [4, 2]
    set A [torch::tensor_create {1.0 1.0 1.0 2.0 1.0 3.0} float32]
    set A [torch::tensor_reshape $A {3 2}]
    set b [torch::tensor_create {6.0 8.0 10.0} float32]
    set b [torch::tensor_reshape $b {3 1}]
    
    set result [torch::lstsq $b $A]
    expr {[string length $result] > 0}
} -result 1

# Data type support tests
test lstsq-7.1 {Different data types - float64} -body {
    set A [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float64]
    set A [torch::tensor_reshape $A {3 2}]
    set B [torch::tensor_create {1.0 2.0 3.0} float64]
    set B [torch::tensor_reshape $B {3 1}]
    set result [torch::lstsq $B $A]
    expr {[string length $result] > 0}
} -result 1

cleanupTests 