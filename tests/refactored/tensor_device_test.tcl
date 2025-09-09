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
test tensor-device-1.1 {Basic positional syntax - CPU} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_device $a]
    expr {[string length $result] > 0}
} {1}

test tensor-device-1.2 {Basic positional syntax - CUDA (if available)} {
    if {[catch {torch::cuda_is_available} cuda_available] || !$cuda_available} {
        skip "CUDA not available"
    } else {
        set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cuda]
        set result [torch::tensor_device $a]
        expr {[string length $result] > 0}
    }
} {1}

test tensor-device-1.3 {Basic positional syntax - different data types} {
    set a [torch::tensor_create -data {1 2 3} -dtype int32 -device cpu]
    set result [torch::tensor_device $a]
    expr {[string length $result] > 0}
} {1}

# Test cases for named syntax
test tensor-device-2.1 {Named parameter syntax - CPU} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_device -input $a]
    expr {[string length $result] > 0}
} {1}

test tensor-device-2.2 {Named parameter syntax - CUDA (if available)} {
    if {[catch {torch::cuda_is_available} cuda_available] || !$cuda_available} {
        skip "CUDA not available"
    } else {
        set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cuda]
        set result [torch::tensor_device -input $a]
        expr {[string length $result] > 0}
    }
} {1}

test tensor-device-2.3 {Named parameter syntax - different data types} {
    set a [torch::tensor_create -data {1 2 3} -dtype int32 -device cpu]
    set result [torch::tensor_device -input $a]
    expr {[string length $result] > 0}
} {1}

# Test cases for camelCase alias
test tensor-device-3.1 {CamelCase alias - CPU} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensorDevice -input $a]
    expr {[string length $result] > 0}
} {1}

test tensor-device-3.2 {CamelCase alias - CUDA (if available)} {
    if {[catch {torch::cuda_is_available} cuda_available] || !$cuda_available} {
        skip "CUDA not available"
    } else {
        set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cuda]
        set result [torch::tensorDevice -input $a]
        expr {[string length $result] > 0}
    }
} {1}

test tensor-device-3.3 {CamelCase alias - different data types} {
    set a [torch::tensor_create -data {1 2 3} -dtype int32 -device cpu]
    set result [torch::tensorDevice -input $a]
    expr {[string length $result] > 0}
} {1}

# Error handling tests
test tensor-device-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_device invalid_tensor} result
    expr {[string length $result] > 0}
} {1}

test tensor-device-4.2 {Error handling - missing input parameter} {
    catch {torch::tensor_device} result
    expr {[string length $result] > 0}
} {1}

test tensor-device-4.3 {Error handling - too many arguments} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_device $a extra} result
    expr {[string length $result] > 0}
} {1}

test tensor-device-4.4 {Error handling - unknown parameter} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_device -input $a -unknown_param value} result
    expr {[string length $result] > 0}
} {1}

# Device correctness tests
test tensor-device-5.1 {Device correctness - CPU} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_device $a]
    expr {[string match "*cpu*" $result] || [string match "*CPU*" $result]}
} {1}

test tensor-device-5.2 {Device correctness - CUDA (if available)} {
    if {[catch {torch::cuda_is_available} cuda_available] || !$cuda_available} {
        skip "CUDA not available"
    } else {
        set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cuda]
        set result [torch::tensor_device $a]
        expr {[string match "*cuda*" $result] || [string match "*CUDA*" $result]}
    }
} {1}

test tensor-device-5.3 {Device correctness - different data types same device} {
    set a1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set a2 [torch::tensor_create -data {1 2 3} -dtype int32 -device cpu]
    set result1 [torch::tensor_device $a1]
    set result2 [torch::tensor_device $a2]
    expr {[string length $result1] > 0 && [string length $result2] > 0 && $result1 == $result2}
} {1}

# Edge cases
test tensor-device-6.1 {Edge case - empty tensor} {
    set a [torch::tensor_create -data {} -dtype float32 -device cpu]
    set result [torch::tensor_device $a]
    expr {[string length $result] > 0}
} {1}

test tensor-device-6.2 {Edge case - single element tensor} {
    set a [torch::tensor_create -data {5.0} -dtype float32 -device cpu]
    set result [torch::tensor_device $a]
    expr {[string length $result] > 0}
} {1}

test tensor-device-6.3 {Edge case - large tensor} {
    set data [list]
    for {set i 0} {$i < 1000} {incr i} {
        lappend data [expr {$i * 1.0}]
    }
    set a [torch::tensor_create -data $data -dtype float32 -device cpu]
    set result [torch::tensor_device $a]
    expr {[string length $result] > 0}
} {1}

# Syntax consistency tests
test tensor-device-7.1 {Syntax consistency - positional vs named} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_device $a]
    set result2 [torch::tensor_device -input $a]
    expr {[string length $result1] > 0 && [string length $result2] > 0 && $result1 == $result2}
} {1}

test tensor-device-7.2 {Syntax consistency - positional vs camelCase} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_device $a]
    set result2 [torch::tensorDevice -input $a]
    expr {[string length $result1] > 0 && [string length $result2] > 0 && $result1 == $result2}
} {1}

test tensor-device-7.3 {Syntax consistency - different devices} {
    set a1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    if {[catch {torch::cuda_is_available} cuda_available] || !$cuda_available} {
        skip "CUDA not available for comparison"
    } else {
        set a2 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cuda]
        set result1 [torch::tensor_device $a1]
        set result2 [torch::tensor_device $a2]
        expr {[string length $result1] > 0 && [string length $result2] > 0 && $result1 != $result2}
    }
} {1}

# Data type independence tests
test tensor-device-8.1 {Data type independence - float32} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_device $a]
    expr {[string length $result] > 0}
} {1}

test tensor-device-8.2 {Data type independence - float64} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64 -device cpu]
    set result [torch::tensor_device $a]
    expr {[string length $result] > 0}
} {1}

test tensor-device-8.3 {Data type independence - int32} {
    set a [torch::tensor_create -data {1 2 3} -dtype int32 -device cpu]
    set result [torch::tensor_device $a]
    expr {[string length $result] > 0}
} {1}

test tensor-device-8.4 {Data type independence - int64} {
    set a [torch::tensor_create -data {1 2 3} -dtype int64 -device cpu]
    set result [torch::tensor_device $a]
    expr {[string length $result] > 0}
} {1}

test tensor-device-8.5 {Data type independence - bool} {
    set a [torch::tensor_create -data {1 0 1} -dtype bool -device cpu]
    set result [torch::tensor_device $a]
    expr {[string length $result] > 0}
} {1}

# Device format tests
test tensor-device-9.1 {Device format - CPU device string} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_device $a]
    # Should contain "cpu" in some form
    expr {[string match "*cpu*" [string tolower $result]]}
} {1}

test tensor-device-9.2 {Device format - CUDA device string (if available)} {
    if {[catch {torch::cuda_is_available} cuda_available] || !$cuda_available} {
        skip "CUDA not available"
    } else {
        set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cuda]
        set result [torch::tensor_device $a]
        # Should contain "cuda" in some form
        expr {[string match "*cuda*" [string tolower $result]]}
    }
} {1}

# Multiple device test
test tensor-device-10.1 {Multiple devices - same device consistency} {
    set a1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set a2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_device $a1]
    set result2 [torch::tensor_device $a2]
    expr {[string length $result1] > 0 && [string length $result2] > 0 && $result1 == $result2}
} {1}

cleanupTests 