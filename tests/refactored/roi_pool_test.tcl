#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

if {[catch {load ../../build/libtorchtcl.so} err]} {
    puts "Failed to load libtorchtcl.so: $err"
    exit 1
}

configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

proc make_feats {} { return [torch::ones {1 1 16 16}] }
proc make_boxes {} {
    # Create 1×5 boxes tensor using positional syntax; shape not critical for simplified ROI pool
    return [torch::tensor_create {0 0 0 15 15} {5} float32]
}

# Positional

test roi_pool-1.1 {positional syntax} {
    set feats [make_feats]
    set boxes [make_boxes]
    set out [torch::roi_pool $feats $boxes {4 4}]
    string match "tensor*" $out
} {1}

# Named

test roi_pool-2.1 {named syntax} {
    set feats [make_feats]
    set boxes [make_boxes]
    set out [torch::roi_pool -input $feats -boxes $boxes -outputSize {4 4}]
    string match "tensor*" $out
} {1}

# CamelCase alias

test roi_pool-3.1 {camelCase alias} {
    set feats [make_feats]
    set boxes [make_boxes]
    set out [torch::roiPool -input $feats -boxes $boxes -outputSize {4 4}]
    string match "tensor*" $out
} {1}

# Error handling

test roi_pool-4.1 {missing required param} -body {
    torch::roi_pool -boxes invalid -outputSize {4 4}
} -returnCodes error -match glob -result *

cleanupTests 