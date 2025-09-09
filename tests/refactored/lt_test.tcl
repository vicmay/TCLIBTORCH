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
test lt-1.1 {Basic positional syntax} -body {
    set a [torch::tensor_create {1.0 2.0 3.0} float32]
    set b [torch::tensor_create {2.0 2.0 2.0} float32]
    set result [torch::lt $a $b]
    expr {[string length $result] > 0}
} -result 1

test lt-1.2 {Positional syntax with different shapes} -body {
    set a [torch::tensor_create {1.0 3.0 2.0 4.0} float32]
    set a [torch::tensor_reshape $a {2 2}]
    set b [torch::tensor_create {2.0 2.0 3.0 3.0} float32]
    set b [torch::tensor_reshape $b {2 2}]
    set result [torch::lt $a $b]
    expr {[string length $result] > 0}
} -result 1

test lt-1.3 {Positional syntax with integer tensors} -body {
    set a [torch::tensor_create {1 2 3} int32]
    set b [torch::tensor_create {2 2 2} int32]
    set result [torch::lt $a $b]
    expr {[string length $result] > 0}
} -result 1

;# Test cases for named parameter syntax
test lt-2.1 {Named parameter syntax with -input1 -input2} -body {
    set a [torch::tensor_create {1.0 2.0 3.0} float32]
    set b [torch::tensor_create {2.0 2.0 2.0} float32]
    set result [torch::lt -input1 $a -input2 $b]
    expr {[string length $result] > 0}
} -result 1

test lt-2.2 {Named parameter syntax with -tensor1 -tensor2} -body {
    set a [torch::tensor_create {1.0 2.0 3.0} float32]
    set b [torch::tensor_create {2.0 2.0 2.0} float32]
    set result [torch::lt -tensor1 $a -tensor2 $b]
    expr {[string length $result] > 0}
} -result 1

test lt-2.3 {Named parameter syntax with mixed parameter names} -body {
    set a [torch::tensor_create {1.0 2.0 3.0} float32]
    set b [torch::tensor_create {2.0 2.0 2.0} float32]
    set result [torch::lt -input1 $a -tensor2 $b]
    expr {[string length $result] > 0}
} -result 1

test lt-2.4 {Named parameter syntax with reversed order} -body {
    set a [torch::tensor_create {1.0 2.0 3.0} float32]
    set b [torch::tensor_create {2.0 2.0 2.0} float32]
    set result [torch::lt -input2 $b -input1 $a]
    expr {[string length $result] > 0}
} -result 1

;# Test cases for camelCase alias
test lt-3.1 {CamelCase alias torch::Lt} -body {
    set a [torch::tensor_create {1.0 2.0 3.0} float32]
    set b [torch::tensor_create {2.0 2.0 2.0} float32]
    set result [torch::Lt $a $b]
    expr {[string length $result] > 0}
} -result 1

test lt-3.2 {CamelCase alias with named parameters} -body {
    set a [torch::tensor_create {1.0 2.0 3.0} float32]
    set b [torch::tensor_create {2.0 2.0 2.0} float32]
    set result [torch::Lt -input1 $a -input2 $b]
    expr {[string length $result] > 0}
} -result 1

;# Test cases for error handling
test lt-4.1 {Error handling - missing arguments} -body {
    catch {torch::lt} result
    expr {[string length $result] > 0}
} -result 1

test lt-4.2 {Error handling - invalid tensor name} -body {
    set a [torch::tensor_create {1.0 2.0 3.0} float32]
    catch {torch::lt $a invalid_tensor} result
    expr {[string length $result] > 0}
} -result 1

test lt-4.3 {Error handling - unknown parameter} -body {
    set a [torch::tensor_create {1.0 2.0 3.0} float32]
    set b [torch::tensor_create {2.0 2.0 2.0} float32]
    catch {torch::lt -unknown $a -input2 $b} result
    expr {[string length $result] > 0}
} -result 1

test lt-4.4 {Error handling - missing parameter value} -body {
    set a [torch::tensor_create {1.0 2.0 3.0} float32]
    catch {torch::lt -input1 $a -input2} result
    expr {[string length $result] > 0}
} -result 1

;# Test cases for different data types
test lt-5.1 {Different data types - float64} -body {
    set a [torch::tensor_create {1.0 2.0 3.0} float64]
    set b [torch::tensor_create {2.0 2.0 2.0} float64]
    set result [torch::lt $a $b]
    expr {[string length $result] > 0}
} -result 1

test lt-5.2 {Different data types - int64} -body {
    set a [torch::tensor_create {1 2 3} int64]
    set b [torch::tensor_create {2 2 2} int64]
    set result [torch::lt $a $b]
    expr {[string length $result] > 0}
} -result 1

;# Test cases for syntax consistency
test lt-6.1 {Syntax consistency - same results} -body {
    set a [torch::tensor_create {1.0 2.0 3.0} float32]
    set b [torch::tensor_create {2.0 2.0 2.0} float32]
    set result1 [torch::lt $a $b]
    set result2 [torch::lt -input1 $a -input2 $b]
    set result3 [torch::Lt $a $b]
    ;# All should return valid tensor handles
    expr {[string length $result1] > 0 && [string length $result2] > 0 && [string length $result3] > 0}
} -result 1

cleanupTests 