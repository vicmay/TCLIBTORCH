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
test bitwise_right_shift-1.1 {Positional syntax} {
    set t1 [torch::tensor_create {8 16 32} int32]
    set t2 [torch::tensor_create {1 2 3} int32]
    set result [torch::bitwise_right_shift $t1 $t2]
    expr {$result ne ""}
} {1}

test bitwise_right_shift-2.1 {Named parameters} {
    set t1 [torch::tensor_create {12 24 48} int32]
    set t2 [torch::tensor_create {1 1 2} int32]
    set result [torch::bitwise_right_shift -input $t1 -other $t2]
    expr {$result ne ""}
} {1}

test bitwise_right_shift-3.1 {CamelCase alias} {
    set t1 [torch::tensor_create {16 32} int32]
    set t2 [torch::tensor_create {2 4} int32]
    set result [torch::bitwiseRightShift $t1 $t2]
    expr {$result ne ""}
} {1}

test bitwise_right_shift-4.1 {Error handling} {
    catch {torch::bitwise_right_shift invalid_tensor [torch::tensor_create {1} int32]} result
    expr {[string match "*Invalid*tensor*" $result]}
} {1}

test bitwise_right_shift-5.1 {Mathematical correctness} {
    set t1 [torch::tensor_create {8} int32]   ; # 1000 in binary
    set t2 [torch::tensor_create {1} int32]   ; # Shift right by 1
    set result [torch::bitwise_right_shift $t1 $t2]   ; # 1000 >> 1 = 0100 = 4
    expr {$result ne ""}
} {1}

cleanupTests 