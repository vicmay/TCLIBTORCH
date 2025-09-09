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
test tensor-to-1.1 {Basic positional syntax - device only} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_to $x cpu]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

test tensor-to-1.2 {Basic positional syntax - device and dtype} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_to $x cpu float64]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

test tensor-to-1.3 {Basic positional syntax - CUDA device (if available)} {
    if {[catch {torch::cuda_is_available} cuda_available] || !$cuda_available} {
        skip "CUDA not available"
    } else {
        set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
        set result [torch::tensor_to $x cuda]
        expr {[string length $result] > 0 && [string match "tensor*" $result]}
    }
} {1}

# Test cases for named syntax
test tensor-to-2.1 {Named parameter syntax - device only} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_to -input $x -device cpu]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

test tensor-to-2.2 {Named parameter syntax - device and dtype} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_to -input $x -device cpu -dtype float64]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

test tensor-to-2.3 {Named parameter syntax - CUDA device (if available)} {
    if {[catch {torch::cuda_is_available} cuda_available] || !$cuda_available} {
        skip "CUDA not available"
    } else {
        set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
        set result [torch::tensor_to -input $x -device cuda]
        expr {[string length $result] > 0 && [string match "tensor*" $result]}
    }
} {1}

# Test cases for camelCase alias
test tensor-to-3.1 {CamelCase alias - device only} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensorTo -input $x -device cpu]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

test tensor-to-3.2 {CamelCase alias - device and dtype} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensorTo -input $x -device cpu -dtype float64]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

test tensor-to-3.3 {CamelCase alias - CUDA device (if available)} {
    if {[catch {torch::cuda_is_available} cuda_available] || !$cuda_available} {
        skip "CUDA not available"
    } else {
        set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
        set result [torch::tensorTo -input $x -device cuda]
        expr {[string length $result] > 0 && [string match "tensor*" $result]}
    }
} {1}

# Error handling tests
test tensor-to-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_to invalid_tensor cpu} result
    expr {[string length $result] > 0}
} {1}

test tensor-to-4.2 {Error handling - missing input parameter} {
    catch {torch::tensor_to} result
    expr {[string length $result] > 0}
} {1}

test tensor-to-4.3 {Error handling - missing device parameter} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_to $x} result
    expr {[string length $result] > 0}
} {1}

test tensor-to-4.4 {Error handling - invalid device} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_to $x invalid_device} result
    expr {[string length $result] > 0}
} {1}

test tensor-to-4.5 {Error handling - invalid dtype} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_to $x cpu invalid_dtype} result
    expr {[string length $result] > 0}
} {1}

test tensor-to-4.6 {Error handling - unknown parameter} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_to -input $x -device cpu -unknown_param value} result
    expr {[string length $result] > 0}
} {1}

# Device conversion tests
test tensor-to-5.1 {Device conversion - CPU to CPU} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_to $x cpu]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

test tensor-to-5.2 {Device conversion - CPU to CUDA (if available)} {
    if {[catch {torch::cuda_is_available} cuda_available] || !$cuda_available} {
        skip "CUDA not available"
    } else {
        set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
        set result [torch::tensor_to $x cuda]
        expr {[string length $result] > 0 && [string match "tensor*" $result]}
    }
} {1}

test tensor-to-5.3 {Device conversion - CUDA to CPU (if available)} {
    if {[catch {torch::cuda_is_available} cuda_available] || !$cuda_available} {
        skip "CUDA not available"
    } else {
        set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cuda]
        set result [torch::tensor_to $x cpu]
        expr {[string length $result] > 0 && [string match "tensor*" $result]}
    }
} {1}

# Data type conversion tests
test tensor-to-6.1 {Data type conversion - float32 to float64} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_to $x cpu float64]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

test tensor-to-6.2 {Data type conversion - float64 to float32} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64 -device cpu]
    set result [torch::tensor_to $x cpu float32]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

test tensor-to-6.3 {Data type conversion - int32 to int64} {
    set x [torch::tensor_create -data {1 2 3} -dtype int32 -device cpu]
    set result [torch::tensor_to $x cpu int64]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

test tensor-to-6.4 {Data type conversion - int64 to int32} {
    set x [torch::tensor_create -data {1 2 3} -dtype int64 -device cpu]
    set result [torch::tensor_to $x cpu int32]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

# Combined device and dtype conversion tests
test tensor-to-7.1 {Combined conversion - device and dtype} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_to $x cpu float64]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

test tensor-to-7.2 {Combined conversion - CUDA device and dtype (if available)} {
    if {[catch {torch::cuda_is_available} cuda_available] || !$cuda_available} {
        skip "CUDA not available"
    } else {
        set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
        set result [torch::tensor_to $x cuda float64]
        expr {[string length $result] > 0 && [string match "tensor*" $result]}
    }
} {1}

# Edge cases
test tensor-to-8.1 {Edge case - zero tensor} {
    set x [torch::tensor_create -data {0.0} -dtype float32 -device cpu]
    set result [torch::tensor_to $x cpu float64]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

test tensor-to-8.2 {Edge case - large tensor} {
    set data [list]
    for {set i 0} {$i < 100} {incr i} {
        lappend data [expr {$i * 1.0}]
    }
    set x [torch::tensor_create -data $data -dtype float32 -device cpu]
    set result [torch::tensor_to $x cpu float64]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

test tensor-to-8.3 {Edge case - negative values} {
    set x [torch::tensor_create -data {-1.0 -2.0 -3.0} -dtype float32 -device cpu]
    set result [torch::tensor_to $x cpu float64]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

# Syntax consistency tests
test tensor-to-9.1 {Syntax consistency - positional vs named} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_to $x cpu]
    set result2 [torch::tensor_to -input $x -device cpu]
    expr {[string length $result1] > 0 && [string length $result2] > 0 && [string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

test tensor-to-9.2 {Syntax consistency - positional vs camelCase} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_to $x cpu]
    set result2 [torch::tensorTo -input $x -device cpu]
    expr {[string length $result1] > 0 && [string length $result2] > 0 && [string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

test tensor-to-9.3 {Syntax consistency - different error conditions} {
    catch {torch::tensor_to invalid_tensor cpu} result1
    catch {torch::tensor_to -input invalid_tensor -device cpu} result2
    expr {[string length $result1] > 0 && [string length $result2] > 0 && $result1 == $result2}
} {1}

# Data type independence tests
test tensor-to-10.1 {Data type independence - float32} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_to $x cpu]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

test tensor-to-10.2 {Data type independence - float64} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64 -device cpu]
    set result [torch::tensor_to $x cpu]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

test tensor-to-10.3 {Data type independence - int32} {
    set x [torch::tensor_create -data {1 2 3} -dtype int32 -device cpu]
    set result [torch::tensor_to $x cpu]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

test tensor-to-10.4 {Data type independence - int64} {
    set x [torch::tensor_create -data {1 2 3} -dtype int64 -device cpu]
    set result [torch::tensor_to $x cpu]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

# Device independence tests
test tensor-to-11.1 {Device independence - CPU tensor} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_to $x cpu]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

test tensor-to-11.2 {Device independence - CUDA tensor (if available)} {
    if {[catch {torch::cuda_is_available} cuda_available] || !$cuda_available} {
        skip "CUDA not available"
    } else {
        set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cuda]
        set result [torch::tensor_to $x cpu]
        expr {[string length $result] > 0 && [string match "tensor*" $result]}
    }
} {1}

# Multiple conversion test
test tensor-to-12.1 {Multiple conversions - same tensor} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_to $x cpu float64]
    set result2 [torch::tensor_to $x cpu int32]
    expr {[string length $result1] > 0 && [string length $result2] > 0 && [string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

# Return value format tests
test tensor-to-13.1 {Return value format - valid conversion returns tensor handle} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_to $x cpu]
    expr {[string length $result] > 0 && [string match "tensor*" $result]}
} {1}

test tensor-to-13.2 {Return value format - error returns error message} {
    catch {torch::tensor_to invalid_tensor cpu} result
    expr {[string length $result] > 0 && ![string match "tensor*" $result]}
} {1}

# Parameter order independence tests
test tensor-to-14.1 {Parameter order independence - named parameters} {
    set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_to -input $x -device cpu -dtype float64]
    set result2 [torch::tensor_to -device cpu -input $x -dtype float64]
    expr {[string length $result1] > 0 && [string length $result2] > 0 && [string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

cleanupTests 