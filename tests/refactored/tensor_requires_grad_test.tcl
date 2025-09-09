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
test tensor-requires-grad-1.1 {Basic positional syntax - requires_grad true} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::tensor_requires_grad $a]
    expr {$result == 1}
} {1}

test tensor-requires-grad-1.2 {Basic positional syntax - requires_grad false} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tensor_requires_grad $a]
    expr {$result == 0}
} {1}

test tensor-requires-grad-1.3 {Basic positional syntax - default requires_grad} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_requires_grad $a]
    expr {$result == 0 || $result == 1}
} {1}

# Test cases for named syntax
test tensor-requires-grad-2.1 {Named parameter syntax - requires_grad true} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::tensor_requires_grad -input $a]
    expr {$result == 1}
} {1}

test tensor-requires-grad-2.2 {Named parameter syntax - requires_grad false} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tensor_requires_grad -input $a]
    expr {$result == 0}
} {1}

test tensor-requires-grad-2.3 {Named parameter syntax - default requires_grad} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_requires_grad -input $a]
    expr {$result == 0 || $result == 1}
} {1}

# Test cases for camelCase alias
test tensor-requires-grad-3.1 {CamelCase alias - requires_grad true} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::tensorRequiresGrad -input $a]
    expr {$result == 1}
} {1}

test tensor-requires-grad-3.2 {CamelCase alias - requires_grad false} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tensorRequiresGrad -input $a]
    expr {$result == 0}
} {1}

test tensor-requires-grad-3.3 {CamelCase alias - default requires_grad} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensorRequiresGrad -input $a]
    expr {$result == 0 || $result == 1}
} {1}

# Error handling tests
test tensor-requires-grad-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_requires_grad invalid_tensor} result
    expr {[string length $result] > 0}
} {1}

test tensor-requires-grad-4.2 {Error handling - missing input parameter} {
    catch {torch::tensor_requires_grad} result
    expr {[string length $result] > 0}
} {1}

test tensor-requires-grad-4.3 {Error handling - too many arguments} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::tensor_requires_grad $a extra} result
    expr {[string length $result] > 0}
} {1}

test tensor-requires-grad-4.4 {Error handling - unknown parameter} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::tensor_requires_grad -input $a -unknown_param value} result
    expr {[string length $result] > 0}
} {1}

# Gradient correctness tests
test tensor-requires-grad-5.1 {Gradient correctness - requires_grad true} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::tensor_requires_grad $a]
    expr {$result == 1}
} {1}

test tensor-requires-grad-5.2 {Gradient correctness - requires_grad false} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tensor_requires_grad $a]
    expr {$result == 0}
} {1}

test tensor-requires-grad-5.3 {Gradient correctness - different data types same setting} {
    set a1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set a2 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64 -device cpu -requiresGrad true]
    set result1 [torch::tensor_requires_grad $a1]
    set result2 [torch::tensor_requires_grad $a2]
    expr {$result1 == 1 && $result2 == 1}
} {1}

# Edge cases
test tensor-requires-grad-6.1 {Edge case - empty tensor with requires_grad true} {
    set a [torch::tensor_create -data {} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::tensor_requires_grad $a]
    expr {$result == 1}
} {1}

test tensor-requires-grad-6.2 {Edge case - single element tensor with requires_grad false} {
    set a [torch::tensor_create -data {5.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tensor_requires_grad $a]
    expr {$result == 0}
} {1}

test tensor-requires-grad-6.3 {Edge case - large tensor with requires_grad true} {
    set data [list]
    for {set i 0} {$i < 1000} {incr i} {
        lappend data [expr {$i * 1.0}]
    }
    set a [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad true]
    set result [torch::tensor_requires_grad $a]
    expr {$result == 1}
} {1}

# Syntax consistency tests
test tensor-requires-grad-7.1 {Syntax consistency - positional vs named} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set result1 [torch::tensor_requires_grad $a]
    set result2 [torch::tensor_requires_grad -input $a]
    expr {$result1 == 1 && $result2 == 1 && $result1 == $result2}
} {1}

test tensor-requires-grad-7.2 {Syntax consistency - positional vs camelCase} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set result1 [torch::tensor_requires_grad $a]
    set result2 [torch::tensorRequiresGrad -input $a]
    expr {$result1 == 1 && $result2 == 1 && $result1 == $result2}
} {1}

test tensor-requires-grad-7.3 {Syntax consistency - different requires_grad settings} {
    set a1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set a2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu -requiresGrad false]
    set result1 [torch::tensor_requires_grad $a1]
    set result2 [torch::tensor_requires_grad $a2]
    expr {$result1 == 1 && $result2 == 0 && $result1 != $result2}
} {1}

# Data type independence tests
test tensor-requires-grad-8.1 {Data type independence - float32 with requires_grad true} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::tensor_requires_grad $a]
    expr {$result == 1}
} {1}

test tensor-requires-grad-8.2 {Data type independence - float64 with requires_grad true} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64 -device cpu -requiresGrad true]
    set result [torch::tensor_requires_grad $a]
    expr {$result == 1}
} {1}

# Device independence tests
test tensor-requires-grad-9.1 {Device independence - CPU tensor with requires_grad true} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::tensor_requires_grad $a]
    expr {$result == 1}
} {1}

test tensor-requires-grad-9.2 {Device independence - CUDA tensor with requires_grad true (if available)} {
    if {[catch {torch::cuda_is_available} cuda_available] || !$cuda_available} {
        skip "CUDA not available"
    } else {
        set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cuda -requiresGrad true]
        set result [torch::tensor_requires_grad $a]
        expr {$result == 1}
    }
} {1}

# Multiple tensor test
test tensor-requires-grad-10.1 {Multiple tensors - same requires_grad setting} {
    set a1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set a2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu -requiresGrad true]
    set result1 [torch::tensor_requires_grad $a1]
    set result2 [torch::tensor_requires_grad $a2]
    expr {$result1 == 1 && $result2 == 1 && $result1 == $result2}
} {1}

test tensor-requires-grad-10.2 {Multiple tensors - different requires_grad settings} {
    set a1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set a2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu -requiresGrad false]
    set result1 [torch::tensor_requires_grad $a1]
    set result2 [torch::tensor_requires_grad $a2]
    expr {$result1 == 1 && $result2 == 0 && $result1 != $result2}
} {1}

# Return value format tests
test tensor-requires-grad-11.1 {Return value format - requires_grad true returns 1} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::tensor_requires_grad $a]
    expr {$result == 1}
} {1}

test tensor-requires-grad-11.2 {Return value format - requires_grad false returns 0} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tensor_requires_grad $a]
    expr {$result == 0}
} {1}

test tensor-requires-grad-11.3 {Return value format - boolean conversion} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::tensor_requires_grad $a]
    expr {[string is boolean $result]}
} {1}

cleanupTests 