#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

if {[catch {load ../../build/libtorchtcl.so} err]} {
    puts "Failed to load libtorchtcl.so: $err"
    exit 1
}

configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Prepare simple tensors
set x [torch::tensor_randn {1 3 10}]
set w [torch::tensor_randn {6 3 3}]
set b [torch::tensor_randn {6}]

# Positional syntax
test conv1d-1.1 {Positional conv1d} {
    set y [torch::conv1d $x $w $b 1 0 1 1]
    expr {[string length $y] > 0}
} 1

# Named parameter syntax
test conv1d-2.1 {Named parameter conv1d} {
    set y [torch::conv1d -input $x -weight $w -bias $b -stride 1 -padding 0 -dilation 1 -groups 1]
    expr {[string length $y] > 0}
} 1

# Bias none
test conv1d-3.1 {No bias specified} {
    set y [torch::conv1d -input $x -weight $w -bias none]
    expr {[string length $y] > 0}
} 1

# Error handling missing required
test conv1d-4.1 {Missing weight} -body {
    torch::conv1d -input $x -stride 1
} -returnCodes error -match glob -result *

cleanupTests 