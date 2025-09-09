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

# Test cases for positional syntax
test tensor_mul-1.1 {Basic positional syntax} {
    set t1 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set t2 [torch::tensorCreate -data {2.0 3.0 4.0} -dtype float32]
    set result [torch::tensor_mul $t1 $t2]
    expr {[string length $result] > 0}
} {1}

test tensor_mul-1.2 {Positional syntax with different shapes} {
    set t1 [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    set t2 [torch::tensorCreate -data {3.0} -dtype float32]
    set result [torch::tensor_mul $t1 $t2]
    expr {[string length $result] > 0}
} {1}

# Test cases for named syntax
test tensor_mul-2.1 {Named parameter syntax} {
    set t1 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set t2 [torch::tensorCreate -data {2.0 3.0 4.0} -dtype float32]
    set result [torch::tensor_mul -input $t1 -other $t2]
    expr {[string length $result] > 0}
} {1}

test tensor_mul-2.2 {Named syntax with different shapes} {
    set t1 [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    set t2 [torch::tensorCreate -data {3.0} -dtype float32]
    set result [torch::tensor_mul -input $t1 -other $t2]
    expr {[string length $result] > 0}
} {1}

# Test cases for camelCase alias
test tensor_mul-3.1 {CamelCase alias} {
    set t1 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set t2 [torch::tensorCreate -data {2.0 3.0 4.0} -dtype float32]
    set result [torch::tensorMul -input $t1 -other $t2]
    expr {[string length $result] > 0}
} {1}

test tensor_mul-3.2 {CamelCase alias positional syntax} {
    set t1 [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    set t2 [torch::tensorCreate -data {3.0} -dtype float32]
    set result [torch::tensorMul $t1 $t2]
    expr {[string length $result] > 0}
} {1}

# Test cases for different data types
test tensor_mul-4.1 {Different data types} {
    set t1 [torch::tensorCreate -data {1 2 3} -dtype int32]
    set t2 [torch::tensorCreate -data {2 3 4} -dtype int32]
    set result [torch::tensor_mul $t1 $t2]
    expr {[string length $result] > 0}
} {1}

test tensor_mul-4.2 {Mixed data types} {
    set t1 [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    set t2 [torch::tensorCreate -data {3} -dtype int32]
    set result [torch::tensor_mul $t1 $t2]
    expr {[string length $result] > 0}
} {1}

# Error handling tests
test tensor_mul-5.1 {Missing parameters} {
    set t1 [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    catch {torch::tensor_mul $t1} result
    return $result
} {Usage: torch::tensor_mul tensor1 tensor2|scalar}

test tensor_mul-5.2 {Invalid tensor name} {
    catch {torch::tensor_mul invalid_tensor1 invalid_tensor2} result
    return $result
} {Invalid first tensor name}

test tensor_mul-5.3 {Invalid named parameter} {
    set t1 [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    set t2 [torch::tensorCreate -data {3.0} -dtype float32]
    catch {torch::tensor_mul -input $t1 -invalid $t2} result
    return $result
} {Unknown parameter: -invalid}

test tensor_mul-5.4 {Missing named parameter value} {
    set t1 [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    catch {torch::tensor_mul -input $t1 -other} result
    return $result
} {Missing value for parameter}

# Edge cases
test tensor_mul-6.1 {Zero tensor multiplication} {
    set t1 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set t2 [torch::tensorCreate -data {0.0 0.0 0.0} -dtype float32]
    set result [torch::tensor_mul $t1 $t2]
    expr {[string length $result] > 0}
} {1}

test tensor_mul-6.2 {Large number multiplication} {
    set t1 [torch::tensorCreate -data {1000000.0} -dtype float32]
    set t2 [torch::tensorCreate -data {2000000.0} -dtype float32]
    set result [torch::tensor_mul $t1 $t2]
    expr {[string length $result] > 0}
} {1}

# Syntax consistency tests
test tensor_mul-7.1 {Both syntaxes produce valid results} {
    set t1 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set t2 [torch::tensorCreate -data {2.0 3.0 4.0} -dtype float32]
    set result1 [torch::tensor_mul $t1 $t2]
    set result2 [torch::tensor_mul -input $t1 -other $t2]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

cleanupTests 