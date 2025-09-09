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
test tril-1.1 {Basic positional syntax} {
    set t [create_3x3]
    set tril [torch::tril $t]
    set vals [torch::tensor_to_list $tril]
    return $vals
} {1.0 0.0 0.0 4.0 5.0 0.0 7.0 8.0 9.0}

test tril-1.2 {Positional syntax with diagonal} {
    set t [create_3x3]
    set tril [torch::tril $t 1]
    set vals [torch::tensor_to_list $tril]
    return $vals
} {1.0 2.0 0.0 4.0 5.0 6.0 7.0 8.0 9.0}

# Test cases for named parameter syntax
test tril-2.1 {Named parameter syntax} {
    set t [create_3x3]
    set tril [torch::tril -input $t]
    set vals [torch::tensor_to_list $tril]
    return $vals
} {1.0 0.0 0.0 4.0 5.0 0.0 7.0 8.0 9.0}

test tril-2.2 {Named parameter syntax with diagonal} {
    set t [create_3x3]
    set tril [torch::tril -input $t -diagonal 1]
    set vals [torch::tensor_to_list $tril]
    return $vals
} {1.0 2.0 0.0 4.0 5.0 6.0 7.0 8.0 9.0}

# Error handling tests
test tril-3.1 {Error handling - missing input} {
    set result [catch {torch::tril} error]
    return [list $result [string range $error 0 50]]
} {1 {Usage: torch::tril input ?diagonal?}}

test tril-3.2 {Error handling - invalid diagonal} {
    set t [create_3x3]
    set result [catch {torch::tril $t foo} error]
    return $result
} {1}

test tril-3.3 {Error handling - unknown named parameter} {
    set t [create_3x3]
    set result [catch {torch::tril -input $t -foo 1} error]
    return $result
} {1}

test tril-3.4 {Error handling - missing value for parameter} {
    set t [create_3x3]
    set result [catch {torch::tril -input $t -diagonal} error]
    return $result
} {1}

# Mathematical consistency
test tril-4.1 {Mathematical consistency between syntaxes} {
    set t [create_3x3]
    set tril1 [torch::tril $t 1]
    set tril2 [torch::tril -input $t -diagonal 1]
    set vals1 [torch::tensor_to_list $tril1]
    set vals2 [torch::tensor_to_list $tril2]
    return [list $vals1 $vals2]
} {{1.0 2.0 0.0 4.0 5.0 6.0 7.0 8.0 9.0} {1.0 2.0 0.0 4.0 5.0 6.0 7.0 8.0 9.0}}

cleanupTests 