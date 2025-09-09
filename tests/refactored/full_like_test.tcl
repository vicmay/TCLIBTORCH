#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test setup
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Helper to compare shapes
proc _same_shape {a b} {
    set sa [torch::tensor_shape $a]
    set sb [torch::tensor_shape $b]
    return [string equal $sa $sb]
}

# -------------------------- Tests --------------------------

# Test 1: Positional syntax
set const_val 3.14

test full_like-1.1 {Positional syntax} {
    set src [torch::zeros {2 3} float32 cpu false]
    set res [torch::full_like $src $const_val]
    _same_shape $src $res
} {1}

# Test 2: Named parameter syntax
set const2 7.0

test full_like-2.1 {Named parameter syntax} {
    set src [torch::zeros {4 5} float32 cpu false]
    set res [torch::full_like -input $src -value $const2 -dtype float32 -device cpu]
    expr {[string length $res] > 0}
} {1}

# Test 3: requiresGrad flag

test full_like-2.2 {Named syntax with requiresGrad} {
    set src [torch::ones {3 3} float32 cpu false]
    set res [torch::full_like -input $src -value 0.0 -requiresGrad true]
    set rg [torch::tensor_requires_grad $res]
    string equal $rg "1"
} {1}

# Test 4: camelCase alias

test full_like-3.1 {camelCase alias} {
    set src [torch::zeros {2 2} float32 cpu false]
    set res [torch::fullLike -input $src -value 5.5]
    expr {[string length $res] > 0}
} {1}

# Test 5: Error handling - missing input parameter

test full_like-4.1 {Error handling - missing input} {
    set rc [catch {torch::full_like -value 1.0} err]
    expr {$rc == 1 && [string match "*Missing required parameter*" $err]}
} {1}

# Test 6: Error handling - unknown parameter

test full_like-4.2 {Error handling - unknown parameter} {
    set src [torch::zeros {1 1} float32 cpu false]
    set rc [catch {torch::full_like -input $src -value 1.0 -bad 1} err]
    expr {$rc == 1 && [string match "*Unknown parameter*" $err]}
} {1}

# Test 7: Error handling - wrong positional count

test full_like-4.3 {Error handling - wrong positional count} {
    set src [torch::zeros {1 1} float32 cpu false]
    set rc [catch {torch::full_like $src} err]
    expr {$rc == 1}
} {1}

cleanupTests 