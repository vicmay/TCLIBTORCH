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

# Helper to create a 3x3 tensor
proc create_3x3 {} {
    return [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9}} float32 cpu true]
}

# Test cases for positional syntax
test triu-1.1 {Basic positional syntax} {
    set t [create_3x3]
    set triu [torch::triu $t]
    set vals [torch::tensor_to_list $triu]
    return $vals
} {1.0 2.0 3.0 0.0 5.0 6.0 0.0 0.0 9.0}

test triu-1.2 {Positional syntax with diagonal} {
    set t [create_3x3]
    set triu [torch::triu $t 1]
    set vals [torch::tensor_to_list $triu]
    return $vals
} {0.0 2.0 3.0 0.0 0.0 6.0 0.0 0.0 0.0}

# Test cases for named parameter syntax
test triu-2.1 {Named parameter syntax} {
    set t [create_3x3]
    set triu [torch::triu -input $t]
    set vals [torch::tensor_to_list $triu]
    return $vals
} {1.0 2.0 3.0 0.0 5.0 6.0 0.0 0.0 9.0}

test triu-2.2 {Named parameter syntax with diagonal} {
    set t [create_3x3]
    set triu [torch::triu -input $t -diagonal 1]
    set vals [torch::tensor_to_list $triu]
    return $vals
} {0.0 2.0 3.0 0.0 0.0 6.0 0.0 0.0 0.0}

# Error handling tests
test triu-3.1 {Error handling - missing input} {
    set result [catch {torch::triu} error]
    return [list $result [string range $error 0 50]]
} {1 {Usage: torch::triu input ?diagonal?}}

test triu-3.2 {Error handling - invalid diagonal} {
    set t [create_3x3]
    set result [catch {torch::triu $t foo} error]
    return $result
} {1}

test triu-3.3 {Error handling - unknown named parameter} {
    set t [create_3x3]
    set result [catch {torch::triu -input $t -foo 1} error]
    return $result
} {1}

test triu-3.4 {Error handling - missing value for parameter} {
    set t [create_3x3]
    set result [catch {torch::triu -input $t -diagonal} error]
    return $result
} {1}

# Mathematical consistency
test triu-4.1 {Mathematical consistency between syntaxes} {
    set t [create_3x3]
    set triu1 [torch::triu $t 1]
    set triu2 [torch::triu -input $t -diagonal 1]
    set vals1 [torch::tensor_to_list $triu1]
    set vals2 [torch::tensor_to_list $triu2]
    return [list $vals1 $vals2]
} {{0.0 2.0 3.0 0.0 0.0 6.0 0.0 0.0 0.0} {0.0 2.0 3.0 0.0 0.0 6.0 0.0 0.0 0.0}}

cleanupTests 