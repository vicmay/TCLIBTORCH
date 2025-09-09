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
test positional_encoding-1.1 {Basic positional syntax} {
    set result [torch::positional_encoding 4 8 0.1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq {4 8}}
} {1}

test positional_encoding-1.2 {Positional syntax - larger dimensions} {
    set result [torch::positional_encoding 10 16 0.2]
    set shape [torch::tensor_shape $result]
    expr {$shape eq {10 16}}
} {1}

test positional_encoding-1.3 {Positional syntax - zero dropout} {
    set result [torch::positional_encoding 6 12 0.0]
    set shape [torch::tensor_shape $result]
    expr {$shape eq {6 12}}
} {1}

# Test cases for named parameter syntax
test positional_encoding-2.1 {Named parameter syntax} {
    set result [torch::positional_encoding -seqLen 4 -dModel 8 -dropout 0.1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq {4 8}}
} {1}

test positional_encoding-2.2 {Named parameter syntax - larger dimensions} {
    set result [torch::positional_encoding -seqLen 10 -dModel 16 -dropout 0.2]
    set shape [torch::tensor_shape $result]
    expr {$shape eq {10 16}}
} {1}

test positional_encoding-2.3 {Named parameter syntax - zero dropout} {
    set result [torch::positional_encoding -seqLen 6 -dModel 12 -dropout 0.0]
    set shape [torch::tensor_shape $result]
    expr {$shape eq {6 12}}
} {1}

# Test cases for camelCase alias
test positional_encoding-3.1 {CamelCase alias with named parameters} {
    set result [torch::positionalEncoding -seqLen 4 -dModel 8 -dropout 0.1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq {4 8}}
} {1}

test positional_encoding-3.2 {CamelCase alias with larger dimensions} {
    set result [torch::positionalEncoding -seqLen 10 -dModel 16 -dropout 0.2]
    set shape [torch::tensor_shape $result]
    expr {$shape eq {10 16}}
} {1}

# Error cases
test positional_encoding-4.1 {Error - missing required parameters} -body {
    torch::positional_encoding
} -returnCodes error -result {Usage: torch::positional_encoding seq_len d_model dropout}

test positional_encoding-4.2 {Error - invalid sequence length} -body {
    torch::positional_encoding -seqLen -1 -dModel 8 -dropout 0.1
} -returnCodes error -match glob -result {*}

test positional_encoding-4.3 {Error - invalid d_model} -body {
    torch::positional_encoding -seqLen 4 -dModel 0 -dropout 0.1
} -returnCodes error -match glob -result {*}

cleanupTests 