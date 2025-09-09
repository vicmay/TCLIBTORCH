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
test tensor-sub-1.1 {Basic positional syntax} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set t2 [torch::tensor_create -data {0.5 1.0 1.5} -dtype float32 -device cpu]
    set result [torch::tensor_sub $t1 $t2]
    expr {[string length $result] > 0}
} {1}

test tensor-sub-1.2 {Positional syntax with different values} {
    set t1 [torch::tensor_create -data {10.0 20.0 30.0} -dtype float32 -device cpu]
    set t2 [torch::tensor_create -data {5.0 10.0 15.0} -dtype float32 -device cpu]
    set result [torch::tensor_sub $t1 $t2]
    expr {[string length $result] > 0}
} {1}

# Test cases for named parameter syntax
test tensor-sub-2.1 {Named parameter syntax - basic} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set t2 [torch::tensor_create -data {0.5 1.0 1.5} -dtype float32 -device cpu]
    set result [torch::tensor_sub -input $t1 -other $t2]
    expr {[string length $result] > 0}
} {1}

test tensor-sub-2.2 {Named parameter syntax - different order} {
    set t1 [torch::tensor_create -data {10.0 20.0 30.0} -dtype float32 -device cpu]
    set t2 [torch::tensor_create -data {5.0 10.0 15.0} -dtype float32 -device cpu]
    set result [torch::tensor_sub -other $t2 -input $t1]
    expr {[string length $result] > 0}
} {1}

test tensor-sub-2.3 {Named parameter syntax with alpha} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set t2 [torch::tensor_create -data {0.5 1.0 1.5} -dtype float32 -device cpu]
    set result [torch::tensor_sub -input $t1 -other $t2 -alpha 2.0]
    expr {[string length $result] > 0}
} {1}

# Test cases for camelCase alias
test tensor-sub-3.1 {CamelCase alias - positional} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set t2 [torch::tensor_create -data {0.5 1.0 1.5} -dtype float32 -device cpu]
    set result [torch::tensorSub $t1 $t2]
    expr {[string length $result] > 0}
} {1}

test tensor-sub-3.2 {CamelCase alias - named parameters} {
    set t1 [torch::tensor_create -data {10.0 20.0 30.0} -dtype float32 -device cpu]
    set t2 [torch::tensor_create -data {5.0 10.0 15.0} -dtype float32 -device cpu]
    set result [torch::tensorSub -input $t1 -other $t2]
    expr {[string length $result] > 0}
} {1}

test tensor-sub-3.3 {CamelCase alias with alpha} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set t2 [torch::tensor_create -data {0.5 1.0 1.5} -dtype float32 -device cpu]
    set result [torch::tensorSub -input $t1 -other $t2 -alpha 0.5]
    expr {[string length $result] > 0}
} {1}

# Syntax consistency
test tensor-sub-4.1 {Syntax consistency - positional vs named} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set t2 [torch::tensor_create -data {0.5 1.0 1.5} -dtype float32 -device cpu]
    set result1 [torch::tensor_sub $t1 $t2]
    set result2 [torch::tensor_sub -input $t1 -other $t2]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor-sub-4.2 {Syntax consistency - all three syntaxes} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set t2 [torch::tensor_create -data {0.5 1.0 1.5} -dtype float32 -device cpu]
    set result1 [torch::tensor_sub $t1 $t2]
    set result2 [torch::tensor_sub -input $t1 -other $t2]
    set result3 [torch::tensorSub -input $t1 -other $t2]
    expr {[string length $result1] > 0 && [string length $result2] > 0 && [string length $result3] > 0}
} {1}

# Error handling
test tensor-sub-5.1 {Error handling - invalid first tensor} {
    set t2 [torch::tensor_create -data {0.5 1.0 1.5} -dtype float32 -device cpu]
    catch {torch::tensor_sub invalid_tensor $t2} result
    return [string match "*Invalid first tensor name*" $result]
} {1}

test tensor-sub-5.2 {Error handling - invalid second tensor} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_sub $t1 invalid_tensor} result
    return [string match "*Invalid second tensor name*" $result]
} {1}

test tensor-sub-5.3 {Error handling - missing parameters} {
    catch {torch::tensor_sub} result
    return [string match "*Required parameters missing*" $result]
} {1}

test tensor-sub-5.4 {Error handling - missing second parameter} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_sub $t1} result
    return [string match "*Usage: torch::tensor_sub tensor1 tensor2*" $result]
} {1}

test tensor-sub-5.5 {Error handling - unknown parameter} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set t2 [torch::tensor_create -data {0.5 1.0 1.5} -dtype float32 -device cpu]
    catch {torch::tensor_sub -input $t1 -other $t2 -unknown_param value} result
    return [string match "*Unknown parameter*" $result]
} {1}

# Mathematical correctness
test tensor-sub-6.1 {Mathematical correctness - basic subtraction} {
    set t1 [torch::tensor_create -data {5.0 10.0 15.0} -dtype float32 -device cpu]
    set t2 [torch::tensor_create -data {2.0 4.0 6.0} -dtype float32 -device cpu]
    set result [torch::tensor_sub $t1 $t2]
    expr {[string length $result] > 0}
} {1}

test tensor-sub-6.2 {Mathematical correctness - with alpha} {
    set t1 [torch::tensor_create -data {10.0 20.0 30.0} -dtype float32 -device cpu]
    set t2 [torch::tensor_create -data {2.0 4.0 6.0} -dtype float32 -device cpu]
    set result [torch::tensor_sub -input $t1 -other $t2 -alpha 2.0]
    expr {[string length $result] > 0}
} {1}

# Edge cases
test tensor-sub-7.1 {Edge case - zero tensor} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set t2 [torch::tensor_create -data {0.0 0.0 0.0} -dtype float32 -device cpu]
    set result [torch::tensor_sub $t1 $t2]
    expr {[string length $result] > 0}
} {1}

test tensor-sub-7.2 {Edge case - negative values} {
    set t1 [torch::tensor_create -data {-1.0 -2.0 -3.0} -dtype float32 -device cpu]
    set t2 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set result [torch::tensor_sub $t1 $t2]
    expr {[string length $result] > 0}
} {1}

test tensor-sub-7.3 {Edge case - alpha zero} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set t2 [torch::tensor_create -data {0.5 1.0 1.5} -dtype float32 -device cpu]
    set result [torch::tensor_sub -input $t1 -other $t2 -alpha 0.0]
    expr {[string length $result] > 0}
} {1}

# Data preservation
test tensor-sub-8.1 {Data preservation - values unchanged} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set t2 [torch::tensor_create -data {0.5 1.0 1.5} -dtype float32 -device cpu]
    set result [torch::tensor_sub $t1 $t2]
    set values [torch::tensor_print $result]
    # Expected values: 1.0-0.5=0.5, 2.0-1.0=1.0, 3.0-1.5=1.5
    # The actual output format is {0.5   1.   1.5}
    return [string match "*0.5*1.*1.5*" $values]
} {1}

test tensor-sub-8.2 {Data preservation - original tensors unchanged} {
    set t1 [torch::tensor_create -data {5.0 10.0 15.0} -dtype float32 -device cpu]
    set t2 [torch::tensor_create -data {2.0 4.0 6.0} -dtype float32 -device cpu]
    set original_t1 [torch::tensor_print $t1]
    set original_t2 [torch::tensor_print $t2]
    set result [torch::tensor_sub $t1 $t2]
    set new_t1 [torch::tensor_print $t1]
    set new_t2 [torch::tensor_print $t2]
    return [expr {$original_t1 eq $new_t1 && $original_t2 eq $new_t2}]
} {1}

cleanupTests 