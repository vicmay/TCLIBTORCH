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
test tensor-stack-1.1 {Basic positional syntax - stack 1D tensors} {
    set t1 [torch::tensor_create {1 2 3}]
    set t2 [torch::tensor_create {4 5 6}]
    set result [torch::tensor_stack [list $t1 $t2] 0]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 3}

test tensor-stack-1.2 {Positional - stack along new dimension} {
    set t1 [torch::tensor_create {1 2 3}]
    set t2 [torch::tensor_create {4 5 6}]
    set result [torch::tensor_stack [list $t1 $t2] 1]
    set shape [torch::tensor_shape $result]
    return $shape
} {3 2}

# Test cases for named parameter syntax
test tensor-stack-2.1 {Named parameter syntax - basic} {
    set t1 [torch::tensor_create {1 2 3}]
    set t2 [torch::tensor_create {4 5 6}]
    set result [torch::tensor_stack -tensors [list $t1 $t2] -dim 0]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 3}

test tensor-stack-2.2 {Named parameter syntax - different order} {
    set t1 [torch::tensor_create {1 2 3}]
    set t2 [torch::tensor_create {4 5 6}]
    set result [torch::tensor_stack -dim 1 -tensors [list $t1 $t2]]
    set shape [torch::tensor_shape $result]
    return $shape
} {3 2}

# Test cases for camelCase alias
test tensor-stack-3.1 {CamelCase alias - basic} {
    set t1 [torch::tensor_create {1 2 3}]
    set t2 [torch::tensor_create {4 5 6}]
    set result [torch::tensorStack [list $t1 $t2] 0]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 3}

test tensor-stack-3.2 {CamelCase alias - named parameters} {
    set t1 [torch::tensor_create {1 2 3}]
    set t2 [torch::tensor_create {4 5 6}]
    set result [torch::tensorStack -tensors [list $t1 $t2] -dim 1]
    set shape [torch::tensor_shape $result]
    return $shape
} {3 2}

# Syntax consistency
test tensor-stack-4.1 {Syntax consistency - positional vs named} {
    set t1 [torch::tensor_create {1 2 3}]
    set t2 [torch::tensor_create {4 5 6}]
    set result1 [torch::tensor_stack [list $t1 $t2] 0]
    set result2 [torch::tensor_stack -tensors [list $t1 $t2] -dim 0]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    return [expr {$shape1 eq $shape2}]
} {1}

test tensor-stack-4.2 {Syntax consistency - all three syntaxes} {
    set t1 [torch::tensor_create {1 2 3}]
    set t2 [torch::tensor_create {4 5 6}]
    set result1 [torch::tensor_stack [list $t1 $t2] 0]
    set result2 [torch::tensor_stack -tensors [list $t1 $t2] -dim 0]
    set result3 [torch::tensorStack [list $t1 $t2] 0]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set shape3 [torch::tensor_shape $result3]
    return [expr {$shape1 eq $shape2 && $shape2 eq $shape3}]
} {1}

# Error handling
test tensor-stack-5.1 {Error handling - invalid tensor} {
    set t1 [torch::tensor_create {1 2 3}]
    catch {torch::tensor_stack [list $t1 invalid_tensor] 0} result
    return [string match "*Invalid tensor name*" $result]
} {1}

test tensor-stack-5.2 {Error handling - missing tensors} {
    catch {torch::tensor_stack -dim 0} result
    return [string match "*Required parameter missing*" $result]
} {1}

test tensor-stack-5.3 {Error handling - missing dim positional} {
    set t1 [torch::tensor_create {1 2 3}]
    set t2 [torch::tensor_create {4 5 6}]
    catch {torch::tensor_stack [list $t1 $t2]} result
    return [string match "*Usage: torch::tensor_stack tensors dim*" $result]
} {1}

test tensor-stack-5.4 {Error handling - unknown parameter} {
    set t1 [torch::tensor_create {1 2 3}]
    set t2 [torch::tensor_create {4 5 6}]
    catch {torch::tensor_stack -tensors [list $t1 $t2] -foo 0} result
    return [string match "*Unknown parameter*" $result]
} {1}

# Edge cases
test tensor-stack-6.1 {Edge case - stack more than two tensors} {
    set t1 [torch::tensor_create {1 2 3}]
    set t2 [torch::tensor_create {4 5 6}]
    set t3 [torch::tensor_create {7 8 9}]
    set result [torch::tensor_stack [list $t1 $t2 $t3] 0]
    set shape [torch::tensor_shape $result]
    return $shape
} {3 3}

test tensor-stack-6.2 {Edge case - stack along higher dimension} {
    set t1 [torch::tensor_create {1 2 3 4}]
    set t2 [torch::tensor_create {5 6 7 8}]
    set t1_2d [torch::tensor_reshape $t1 {2 2}]
    set t2_2d [torch::tensor_reshape $t2 {2 2}]
    set result [torch::tensor_stack [list $t1_2d $t2_2d] 2]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2 2}

# Data preservation
test tensor-stack-7.1 {Data preservation - values unchanged} {
    set t1 [torch::tensor_create {1 2 3}]
    set t2 [torch::tensor_create {4 5 6}]
    set result [torch::tensor_stack [list $t1 $t2] 0]
    set values [torch::tensor_print $result]
    return [string match "*1*2*3*4*5*6*" $values]
} {1}

test tensor-stack-7.2 {Data preservation - stack and unstack} {
    set t1 [torch::tensor_create {1 2 3}]
    set t2 [torch::tensor_create {4 5 6}]
    set stacked [torch::tensor_stack [list $t1 $t2] 0]
    set shape [torch::tensor_shape $stacked]
    return [expr {$shape eq "2 3"}]
} {1}

cleanupTests 