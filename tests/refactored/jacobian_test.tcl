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

# Test 1: Basic positional syntax (backward compatibility)
test jacobian-1.1 {Basic positional syntax} {
    set inputs [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result [torch::jacobian "dummy_func" $inputs]
    set shape [torch::tensor_shape $result]
    ;# Should return identity matrix of size 3x3
    expr {$shape eq "3 3"}
} {1}

test jacobian-1.2 {Positional syntax with different input sizes} {
    set inputs [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    set result [torch::jacobian "test_func" $inputs]
    set shape [torch::tensor_shape $result]
    ;# Should return identity matrix of size 2x2
    expr {$shape eq "2 2"}
} {1}

# Test 2: Named parameter syntax
test jacobian-2.1 {Named parameter syntax with -func and -inputs} {
    set inputs [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result [torch::jacobian -func "dummy_func" -inputs $inputs]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 3"}
} {1}

test jacobian-2.2 {Named parameter syntax with -function and -input aliases} {
    set inputs [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    set result [torch::jacobian -function "test_func" -input $inputs]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} {1}

# Test 3: camelCase alias (Jacobian)
test jacobian-3.1 {camelCase alias with positional syntax} {
    set inputs [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result [torch::Jacobian "dummy_func" $inputs]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 3"}
} {1}

test jacobian-3.2 {camelCase alias with named parameters} {
    set inputs [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    set result [torch::Jacobian -func "test_func" -inputs $inputs]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} {1}

# Test 4: Syntax consistency - both syntaxes produce same results
test jacobian-4.1 {Syntax consistency check} {
    set inputs [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result1 [torch::jacobian "func" $inputs]           ;# Positional
    set result2 [torch::jacobian -func "func" -inputs $inputs]  ;# Named
    set result3 [torch::Jacobian "func" $inputs]           ;# camelCase positional
    set result4 [torch::Jacobian -func "func" -inputs $inputs]  ;# camelCase named
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set shape3 [torch::tensor_shape $result3]
    set shape4 [torch::tensor_shape $result4]
    
    ;# All should have identical shapes
    expr {$shape1 eq $shape2 && $shape2 eq $shape3 && $shape3 eq $shape4}
} {1}

# Test 5: Error handling
test jacobian-5.1 {Error: missing arguments} {
    catch {torch::jacobian} result
    regexp {Usage:} $result
} {1}

test jacobian-5.2 {Error: insufficient arguments in positional} {
    catch {torch::jacobian "func"} result
    regexp {Usage:} $result
} {1}

test jacobian-5.3 {Error: invalid tensor name positional} {
    catch {torch::jacobian "func" invalid_tensor} result
    regexp {Error in jacobian:} $result
} {1}

test jacobian-5.4 {Error: invalid tensor name named} {
    catch {torch::jacobian -func "func" -inputs invalid_tensor} result
    regexp {Error in jacobian:} $result
} {1}

test jacobian-5.5 {Error: unknown parameter} {
    set inputs [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    catch {torch::jacobian -invalid "func" -inputs $inputs} result
    regexp {Unknown parameter} $result
} {1}

test jacobian-5.6 {Error: missing parameter value} {
    catch {torch::jacobian -func} result
    regexp {Named parameters must come in pairs} $result
} {1}

# Test 6: Different tensor sizes and data types
test jacobian-6.1 {Single element tensor} {
    set inputs [torch::tensorCreate -data {42.0} -dtype float32]
    set result [torch::jacobian "func" $inputs]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "1 1"}  ;# 1x1 identity matrix
} {1}

test jacobian-6.2 {Double precision tensor} {
    set inputs [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float64]
    set result [torch::jacobian "func" $inputs]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 3"}
} {1}

test jacobian-6.3 {Larger tensor} {
    set inputs [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32]
    set result [torch::jacobian "func" $inputs]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "5 5"}  ;# 5x5 identity matrix
} {1}

# Test 7: Verify identity matrix properties
test jacobian-7.1 {Identity matrix verification - check it's square and non-empty} {
    set inputs [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result [torch::jacobian "func" $inputs]
    set shape [torch::tensor_shape $result]
    ;# Should be 3x3 square matrix (identity matrix)
    expr {$shape eq "3 3"}
} {1}

test jacobian-7.2 {Identity matrix verification - check different sizes} {
    set inputs1 [torch::tensorCreate -data {1.0} -dtype float32]
    set inputs2 [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    set inputs4 [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -dtype float32]
    
    set result1 [torch::jacobian "func" $inputs1]
    set result2 [torch::jacobian "func" $inputs2]
    set result4 [torch::jacobian "func" $inputs4]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set shape4 [torch::tensor_shape $result4]
    
    ;# All should be square matrices of appropriate size
    expr {$shape1 eq "1 1" && $shape2 eq "2 2" && $shape4 eq "4 4"}
} {1}

# Test 8: Different function names
test jacobian-8.1 {Different function names} {
    set inputs [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    set result1 [torch::jacobian "function_a" $inputs]
    set result2 [torch::jacobian "function_b" $inputs]
    set result3 [torch::jacobian "my_func" $inputs]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set shape3 [torch::tensor_shape $result3]
    
    ;# All should produce same-sized identity matrices
    expr {$shape1 eq "2 2" && $shape2 eq "2 2" && $shape3 eq "2 2"}
} {1}

# Test 9: Edge cases
test jacobian-9.1 {Empty function name should be error} {
    set inputs [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    catch {torch::jacobian "" $inputs} result
    regexp {Required parameters missing} $result
} {1}

test jacobian-9.2 {Function name with spaces} {
    set inputs [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    set result [torch::jacobian "my function" $inputs]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}  ;# Should still work
} {1}

# Test 10: Multi-dimensional input tensors
test jacobian-10.1 {Multi-dimensional input tensor} {
    set inputs [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::jacobian "func" $inputs]
    set shape [torch::tensor_shape $result]
    ;# Should flatten to 4 elements, so 4x4 identity matrix
    expr {$shape eq "4 4"}
} {1}

test jacobian-10.2 {3D input tensor} {
    set inputs [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3 1} -dtype float32]
    set result [torch::jacobian "func" $inputs]
    set shape [torch::tensor_shape $result]
    ;# Should flatten to 6 elements, so 6x6 identity matrix
    expr {$shape eq "6 6"}
} {1}

# Test 11: Parameter validation
test jacobian-11.1 {Mixed parameter order} {
    set inputs [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result [torch::jacobian -inputs $inputs -func "my_func"]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 3"}
} {1}

test jacobian-11.2 {All parameter aliases} {
    set inputs [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    set result1 [torch::jacobian -func "f" -inputs $inputs]
    set result2 [torch::jacobian -function "f" -input $inputs]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$shape1 eq $shape2 && $shape1 eq "2 2"}
} {1}

cleanupTests 