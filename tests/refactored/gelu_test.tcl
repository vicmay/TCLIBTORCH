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
# TORCH::GELU COMPREHENSIVE TEST SUITE
# =====================================================================

# Tests for positional syntax (backward compatibility)

test gelu-1.1 {Basic positional syntax} {
    set t1 [torch::zeros {3 3}]
    set result [torch::gelu $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

# Tests for named parameter syntax

test gelu-2.1 {Named parameter syntax basic} {
    set t1 [torch::ones {2 3}]
    set result [torch::gelu -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

# Tests for camelCase alias (gelu is already lowercase, but ensure consistency)

test gelu-3.1 {Alias consistency positional} {
    set t1 [torch::zeros {4 4}]
    set result1 [torch::gelu $t1]
    set result2 [torch::gelu -input $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for error handling

test gelu-4.1 {Error on missing tensor} {
    catch {torch::gelu} msg
    expr {[string match "*wrong # args*" $msg] || [string match "*Usage*" $msg] || [string match "*Missing*" $msg]}
} {1}

test gelu-4.2 {Error on invalid tensor name} {
    catch {torch::gelu invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

test gelu-4.3 {Error on unknown parameter} {
    set t1 [torch::zeros {2 2}]
    catch {torch::gelu -unknown $t1} msg
    string match "*Unknown parameter*" $msg
} {1}

# Tests for syntax consistency

test gelu-5.1 {Syntax consistency - same shape} {
    set t1 [torch::zeros {5 5}]
    set r1 [torch::gelu $t1]
    set r2 [torch::gelu -input $t1]
    set s1 [torch::tensor_shape $r1]
    set s2 [torch::tensor_shape $r2]
    expr {$s1 eq $s2}
} {1}

cleanupTests 