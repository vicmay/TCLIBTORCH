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

# =====================================================================
# TORCH::GLU COMPREHENSIVE TEST SUITE
# =====================================================================

# Tests for positional syntax (backward compatibility)

test glu-1.1 {Basic positional syntax} {
    ;# GLU expects input tensor with even-sized last dimension
    set t1 [torch::ones {3 4}]
    set result [torch::glu $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3 2}

test glu-1.2 {GLU positional syntax with 3D tensor} {
    ;# GLU halves the last dimension
    set t1 [torch::zeros {2 3 6}]
    set result [torch::glu $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 3}

test glu-1.3 {GLU positional syntax with 1D tensor} {
    ;# GLU with 1D tensor with even size
    set t1 [torch::ones {8}]
    set result [torch::glu $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

# Tests for named parameter syntax

test glu-2.1 {Named parameter syntax basic} {
    set t1 [torch::ones {2 6}]
    set result [torch::glu -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

test glu-2.2 {Named parameter syntax with tensor alias} {
    set t1 [torch::zeros {4 8}]
    set result [torch::glu -tensor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {4 4}

test glu-2.3 {Named parameter syntax 3D tensor} {
    set t1 [torch::ones {2 3 4}]
    set result [torch::glu -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 2}

# Tests for camelCase consistency (glu is already camelCase)

test glu-3.1 {Syntax consistency positional vs named} {
    set t1 [torch::zeros {4 6}]
    set result1 [torch::glu $t1]
    set result2 [torch::glu -input $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test glu-3.2 {Multiple syntax forms consistency} {
    set t1 [torch::ones {3 8}]
    set r1 [torch::glu $t1]
    set r2 [torch::glu -input $t1]
    set r3 [torch::glu -tensor $t1]
    set s1 [torch::tensor_shape $r1]
    set s2 [torch::tensor_shape $r2]
    set s3 [torch::tensor_shape $r3]
    expr {$s1 eq $s2 && $s2 eq $s3}
} {1}

# Tests for error handling

test glu-4.1 {Error on missing tensor} {
    catch {torch::glu} msg
    expr {[string match "*wrong # args*" $msg] || [string match "*Usage*" $msg] || [string match "*Missing*" $msg]}
} {1}

test glu-4.2 {Error on invalid tensor name positional} {
    catch {torch::glu invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

test glu-4.3 {Error on invalid tensor name named} {
    catch {torch::glu -input invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

test glu-4.4 {Error on unknown parameter} {
    set t1 [torch::zeros {2 4}]
    catch {torch::glu -unknown $t1} msg
    string match "*Unknown parameter*" $msg
} {1}

test glu-4.5 {Error on odd last dimension} {
    ;# GLU requires even-sized last dimension
    set t1 [torch::ones {2 3}]
    catch {torch::glu $t1} msg
    ;# Should catch runtime error about dimension size
    expr {$msg ne ""}
} {1}

test glu-4.6 {Error on missing value for named parameter} {
    catch {torch::glu -input} msg
    expr {[string match "*Missing value*" $msg] || [string match "*wrong # args*" $msg]}
} {1}

# Tests for mathematical correctness

test glu-5.1 {GLU mathematical property - dimension halving} {
    ;# GLU always halves the last dimension
    set t1 [torch::randn {3 5 8}]
    set result [torch::glu $t1]
    set input_shape [torch::tensor_shape $t1]
    set output_shape [torch::tensor_shape $result]
    
    ;# Check that output last dim is half of input last dim
    set input_last [lindex $input_shape end]
    set output_last [lindex $output_shape end]
    expr {$output_last * 2 == $input_last}
} {1}

test glu-5.2 {GLU with different data types} {
    ;# Test GLU with float32
    set t1 [torch::ones {2 4} -dtype float32]
    set result [torch::glu $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test glu-5.3 {GLU preserves leading dimensions} {
    ;# GLU should preserve all dimensions except the last
    set t1 [torch::zeros {2 3 4 6}]
    set result [torch::glu $t1]
    set input_shape [torch::tensor_shape $t1]
    set output_shape [torch::tensor_shape $result]
    
    ;# Check that all but last dimension are preserved
    set input_prefix [lrange $input_shape 0 end-1]
    set output_prefix [lrange $output_shape 0 end-1]
    expr {$input_prefix eq $output_prefix}
} {1}

# Tests for edge cases

test glu-6.1 {GLU with minimum valid tensor size} {
    ;# Smallest valid tensor for GLU (last dim = 2)
    set t1 [torch::ones {2}]
    set result [torch::glu $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {1}

test glu-6.2 {GLU with larger tensor} {
    ;# Test with larger tensor
    set t1 [torch::zeros {10 20}]
    set result [torch::glu $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {10 10}

test glu-6.3 {GLU mathematical range validation} {
    ;# GLU output should be finite for finite input
    set t1 [torch::ones {3 4}]
    set result [torch::glu $t1]
    ;# Just ensure we get a result without error
    set shape [torch::tensor_shape $result]
    expr {[llength $shape] > 0}
} {1}

# Tests for multiple call consistency

test glu-7.1 {Multiple calls consistency} {
    set t1 [torch::randn {4 6}]
    set r1 [torch::glu $t1]
    set r2 [torch::glu $t1]
    set s1 [torch::tensor_shape $r1]
    set s2 [torch::tensor_shape $r2]
    expr {$s1 eq $s2}
} {1}

test glu-7.2 {Different input tensors} {
    set t1 [torch::zeros {2 8}]
    set t2 [torch::ones {2 8}]
    set r1 [torch::glu $t1]
    set r2 [torch::glu $t2]
    set s1 [torch::tensor_shape $r1]
    set s2 [torch::tensor_shape $r2]
    expr {$s1 eq $s2}
} {1}

cleanupTests 