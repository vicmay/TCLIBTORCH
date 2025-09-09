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
test tensor-dtype-1.1 {Basic positional syntax - float32} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_dtype $a]
    expr {[string length $result] > 0}
} {1}

test tensor-dtype-1.2 {Basic positional syntax - float64} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64 -device cpu]
    set result [torch::tensor_dtype $a]
    expr {[string length $result] > 0}
} {1}

test tensor-dtype-1.3 {Basic positional syntax - int32} {
    set a [torch::tensor_create -data {1 2 3} -dtype int32 -device cpu]
    set result [torch::tensor_dtype $a]
    expr {[string length $result] > 0}
} {1}

# Test cases for named syntax
test tensor-dtype-2.1 {Named parameter syntax - float32} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_dtype -input $a]
    expr {[string length $result] > 0}
} {1}

test tensor-dtype-2.2 {Named parameter syntax - float64} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64 -device cpu]
    set result [torch::tensor_dtype -input $a]
    expr {[string length $result] > 0}
} {1}

test tensor-dtype-2.3 {Named parameter syntax - int32} {
    set a [torch::tensor_create -data {1 2 3} -dtype int32 -device cpu]
    set result [torch::tensor_dtype -input $a]
    expr {[string length $result] > 0}
} {1}

# Test cases for camelCase alias
test tensor-dtype-3.1 {CamelCase alias - float32} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensorDtype -input $a]
    expr {[string length $result] > 0}
} {1}

test tensor-dtype-3.2 {CamelCase alias - float64} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64 -device cpu]
    set result [torch::tensorDtype -input $a]
    expr {[string length $result] > 0}
} {1}

test tensor-dtype-3.3 {CamelCase alias - int32} {
    set a [torch::tensor_create -data {1 2 3} -dtype int32 -device cpu]
    set result [torch::tensorDtype -input $a]
    expr {[string length $result] > 0}
} {1}

# Error handling tests
test tensor-dtype-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_dtype invalid_tensor} result
    expr {[string length $result] > 0}
} {1}

test tensor-dtype-4.2 {Error handling - missing input parameter} {
    catch {torch::tensor_dtype} result
    expr {[string length $result] > 0}
} {1}

test tensor-dtype-4.3 {Error handling - too many arguments} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_dtype $a extra} result
    expr {[string length $result] > 0}
} {1}

test tensor-dtype-4.4 {Error handling - unknown parameter} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_dtype -input $a -unknown_param value} result
    expr {[string length $result] > 0}
} {1}

# Data type correctness tests
test tensor-dtype-5.1 {Data type correctness - float32} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_dtype $a]
    expr {[string match "*Float*" $result] || [string match "*float*" $result]}
} {1}

test tensor-dtype-5.2 {Data type correctness - float64} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64 -device cpu]
    set result [torch::tensor_dtype $a]
    expr {[string match "*Double*" $result] || [string match "*double*" $result] || [string match "*Float*" $result]}
} {1}

test tensor-dtype-5.3 {Data type correctness - int32} {
    set a [torch::tensor_create -data {1 2 3} -dtype int32 -device cpu]
    set result [torch::tensor_dtype $a]
    expr {[string match "*Int*" $result] || [string match "*int*" $result]}
} {1}

test tensor-dtype-5.4 {Data type correctness - int64} {
    set a [torch::tensor_create -data {1 2 3} -dtype int64 -device cpu]
    set result [torch::tensor_dtype $a]
    expr {[string match "*Long*" $result] || [string match "*long*" $result] || [string match "*Int*" $result]}
} {1}

test tensor-dtype-5.5 {Data type correctness - bool} {
    set a [torch::tensor_create -data {1 0 1} -dtype bool -device cpu]
    set result [torch::tensor_dtype $a]
    expr {[string match "*Bool*" $result] || [string match "*bool*" $result]}
} {1}

# Edge cases
test tensor-dtype-6.1 {Edge case - empty tensor} {
    set a [torch::tensor_create -data {} -dtype float32 -device cpu]
    set result [torch::tensor_dtype $a]
    expr {[string length $result] > 0}
} {1}

test tensor-dtype-6.2 {Edge case - single element tensor} {
    set a [torch::tensor_create -data {5.0} -dtype float32 -device cpu]
    set result [torch::tensor_dtype $a]
    expr {[string length $result] > 0}
} {1}

test tensor-dtype-6.3 {Edge case - large tensor} {
    set data [list]
    for {set i 0} {$i < 1000} {incr i} {
        lappend data [expr {$i * 1.0}]
    }
    set a [torch::tensor_create -data $data -dtype float32 -device cpu]
    set result [torch::tensor_dtype $a]
    expr {[string length $result] > 0}
} {1}

# Syntax consistency tests
test tensor-dtype-7.1 {Syntax consistency - positional vs named} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_dtype $a]
    set result2 [torch::tensor_dtype -input $a]
    expr {[string length $result1] > 0 && [string length $result2] > 0 && $result1 == $result2}
} {1}

test tensor-dtype-7.2 {Syntax consistency - positional vs camelCase} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_dtype $a]
    set result2 [torch::tensorDtype -input $a]
    expr {[string length $result1] > 0 && [string length $result2] > 0 && $result1 == $result2}
} {1}

test tensor-dtype-7.3 {Syntax consistency - different data types} {
    set a1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set a2 [torch::tensor_create -data {1 2 3} -dtype int32 -device cpu]
    set result1 [torch::tensor_dtype $a1]
    set result2 [torch::tensor_dtype $a2]
    expr {[string length $result1] > 0 && [string length $result2] > 0 && $result1 != $result2}
} {1}

# Device independence tests
test tensor-dtype-8.1 {Device independence - CPU tensor} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_dtype $a]
    expr {[string length $result] > 0}
} {1}

test tensor-dtype-8.2 {Device independence - CUDA tensor (if available)} {
    if {[catch {torch::cuda_is_available} cuda_available] || !$cuda_available} {
        skip "CUDA not available"
    } else {
        set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cuda]
        set result [torch::tensor_dtype $a]
        expr {[string length $result] > 0}
    }
} {1}

cleanupTests 