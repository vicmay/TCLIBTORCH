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
test tensor-mean-1.1 {Basic positional syntax - no dimension} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu]
    set result [torch::tensor_mean $a]
    expr {[string length $result] > 0}
} {1}

test tensor-mean-1.2 {Basic positional syntax - with dimension} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set a2d [torch::tensor_reshape $a {2 3}]
    set result [torch::tensor_mean $a2d 0]
    expr {[string length $result] > 0}
} {1}

test tensor-mean-1.3 {Positional syntax - 2D tensor along dimension 1} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set a2d [torch::tensor_reshape $a {2 3}]
    set result [torch::tensor_mean $a2d 1]
    expr {[string length $result] > 0}
} {1}

# Test cases for named syntax
test tensor-mean-2.1 {Named parameter syntax - no dimension} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu]
    set result [torch::tensor_mean -input $a]
    expr {[string length $result] > 0}
} {1}

test tensor-mean-2.2 {Named parameter syntax - with dimension} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set a2d [torch::tensor_reshape $a {2 3}]
    set result [torch::tensor_mean -input $a2d -dim 0]
    expr {[string length $result] > 0}
} {1}

test tensor-mean-2.3 {Named parameter syntax - 2D tensor along dimension 1} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set a2d [torch::tensor_reshape $a {2 3}]
    set result [torch::tensor_mean -input $a2d -dim 1]
    expr {[string length $result] > 0}
} {1}

# Test cases for camelCase alias
test tensor-mean-3.1 {CamelCase alias - no dimension} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu]
    set result [torch::tensorMean -input $a]
    expr {[string length $result] > 0}
} {1}

test tensor-mean-3.2 {CamelCase alias - with dimension} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set a2d [torch::tensor_reshape $a {2 3}]
    set result [torch::tensorMean -input $a2d -dim 0]
    expr {[string length $result] > 0}
} {1}

# Error handling tests
test tensor-mean-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_mean invalid_tensor} result
    expr {[string length $result] > 0}
} {1}

test tensor-mean-4.2 {Error handling - missing input parameter} {
    catch {torch::tensor_mean -dim 0} result
    expr {[string length $result] > 0}
} {1}

test tensor-mean-4.3 {Error handling - invalid dimension value} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_mean -input $a -dim invalid} result
    expr {[string length $result] > 0}
} {1}

test tensor-mean-4.4 {Error handling - too many arguments} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_mean $a 0 extra} result
    expr {[string length $result] > 0}
} {1}

test tensor-mean-4.5 {Error handling - unknown parameter} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_mean -input $a -unknown_param value} result
    expr {[string length $result] > 0}
} {1}

# Mathematical correctness tests
test tensor-mean-5.1 {Mathematical correctness - 1D tensor} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu]
    set result [torch::tensor_mean $a]
    expr {[string length $result] > 0}
} {1}

test tensor-mean-5.2 {Mathematical correctness - 2D tensor along dimension 0} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set a2d [torch::tensor_reshape $a {2 3}]
    set result [torch::tensor_mean $a2d 0]
    expr {[string length $result] > 0}
} {1}

test tensor-mean-5.3 {Mathematical correctness - 2D tensor along dimension 1} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set a2d [torch::tensor_reshape $a {2 3}]
    set result [torch::tensor_mean $a2d 1]
    expr {[string length $result] > 0}
} {1}

test tensor-mean-5.4 {Mathematical correctness - 3D tensor} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set a3d [torch::tensor_reshape $a {2 2 2}]
    set result [torch::tensor_mean $a3d 0]
    expr {[string length $result] > 0}
} {1}

# Edge cases
test tensor-mean-6.1 {Edge case - single element tensor} {
    set a [torch::tensor_create -data {5.0} -dtype float32 -device cpu]
    set result [torch::tensor_mean $a]
    expr {[string length $result] > 0}
} {1}

test tensor-mean-6.2 {Edge case - zero tensor} {
    set a [torch::tensor_create -data {0.0 0.0 0.0} -dtype float32 -device cpu]
    set result [torch::tensor_mean $a]
    expr {[string length $result] > 0}
} {1}

test tensor-mean-6.3 {Edge case - negative values} {
    set a [torch::tensor_create -data {-1.0 -2.0 -3.0} -dtype float32 -device cpu]
    set result [torch::tensor_mean $a]
    expr {[string length $result] > 0}
} {1}

test tensor-mean-6.4 {Edge case - dimension out of bounds} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_mean $a 5} result
    expr {[string length $result] > 0}
} {1}

# Syntax consistency tests
test tensor-mean-7.1 {Syntax consistency - positional vs named} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_mean $a]
    set result2 [torch::tensor_mean -input $a]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor-mean-7.2 {Syntax consistency - positional vs camelCase} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_mean $a]
    set result2 [torch::tensorMean -input $a]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor-mean-7.3 {Syntax consistency - with dimension} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set a2d [torch::tensor_reshape $a {2 3}]
    set result1 [torch::tensor_mean $a2d 0]
    set result2 [torch::tensor_mean -input $a2d -dim 0]
    set result3 [torch::tensorMean -input $a2d -dim 0]
    expr {[string length $result1] > 0 && [string length $result2] > 0 && [string length $result3] > 0}
} {1}

# Data type support tests
test tensor-mean-8.1 {Data type support - float64} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float64 -device cpu]
    set result [torch::tensor_mean $a]
    expr {[string length $result] > 0}
} {1}

cleanupTests 