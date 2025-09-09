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

# Helper to create sample 3-D volume tensor of shape (N, C, D, H, W)
proc createVolume {d h w} {
    # Simple range data
    set total [expr {$d * $h * $w}]
    set data {}
    for {set i 1} {$i <= $total} {incr i} {
        lappend data $i.0
    }
    set t [torch::tensor_create $data float32 cpu false]
    return [torch::tensor_reshape $t [list 1 1 $d $h $w]]
}

# 1. Positional syntax
test adaptive_maxpool3d-1.1 {Positional syntax} {
    set vol [createVolume 4 4 4]
    set result [torch::adaptive_maxpool3d $vol 2]
    expr {[string length $result] > 0}
} {1}

# 2. Named parameter syntax (-output_size)
test adaptive_maxpool3d-2.1 {Named syntax output_size int} {
    set vol [createVolume 4 4 4]
    set result [torch::adaptive_maxpool3d -input $vol -output_size 2]
    expr {[string length $result] > 0}
} {1}

# 3. Named parameter syntax with list
test adaptive_maxpool3d-2.2 {Named syntax outputSize list} {
    set vol [createVolume 6 6 6]
    set result [torch::adaptive_maxpool3d -input $vol -outputSize {3 2 1}]
    expr {[string length $result] > 0}
} {1}

# 4. camelCase alias
test adaptive_maxpool3d-3.1 {camelCase alias} {
    set vol [createVolume 4 4 4]
    set result [torch::adaptiveMaxpool3d -input $vol -output_size 2]
    expr {[string length $result] > 0}
} {1}

# 5. Equivalence of syntaxes (shape compare)
test adaptive_maxpool3d-4.1 {Equivalence positional vs named} {
    set vol1 [createVolume 5 5 5]
    set vol2 [createVolume 5 5 5]
    set r1 [torch::adaptive_maxpool3d $vol1 2]
    set r2 [torch::adaptive_maxpool3d -input $vol2 -output_size 2]
    set s1 [torch::tensor_shape $r1]
    set s2 [torch::tensor_shape $r2]
    string equal $s1 $s2
} {1}

# 6. Error handling ‑ missing input
test adaptive_maxpool3d-5.1 {Missing input param} {
    set code [catch {torch::adaptive_maxpool3d -output_size 2} err]
    expr {$code == 1 && [string match "*Required parameters*" $err]}
} {1}

# 7. Error handling ‑ bad tensor name
test adaptive_maxpool3d-5.2 {Invalid tensor name} {
    set code [catch {torch::adaptive_maxpool3d -input badTensor -output_size 2} err]
    expr {$code == 1 && [string match "*Invalid*" $err]}
} {1}

cleanupTests 