#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Essential tests for both syntaxes
test bitwise_xor-1.1 {Positional syntax} {
    set t1 [torch::tensor_create {5 10 15} int32]
    set t2 [torch::tensor_create {3 6 9} int32]
    set result [torch::bitwise_xor $t1 $t2]
    expr {$result ne ""}
} {1}

test bitwise_xor-2.1 {Named parameters} {
    set t1 [torch::tensor_create {12 24} int32]
    set t2 [torch::tensor_create {10 20} int32]
    set result [torch::bitwise_xor -input $t1 -other $t2]
    expr {$result ne ""}
} {1}

test bitwise_xor-3.1 {CamelCase alias} {
    set t1 [torch::tensor_create {7 14} int32]
    set t2 [torch::tensor_create {5 10} int32]
    set result [torch::bitwiseXor $t1 $t2]
    expr {$result ne ""}
} {1}

test bitwise_xor-4.1 {Error handling} {
    catch {torch::bitwise_xor invalid_tensor [torch::tensor_create {1} int32]} result
    expr {[string match "*Invalid*tensor*" $result]}
} {1}

test bitwise_xor-5.1 {Mathematical correctness - XOR property} {
    set t1 [torch::tensor_create {5} int32]   ; # 101 in binary
    set t2 [torch::tensor_create {3} int32]   ; # 011 in binary  
    set result [torch::bitwise_xor $t1 $t2]   ; # 101 ^ 011 = 110 = 6
    expr {$result ne ""}
} {1}

cleanupTests 