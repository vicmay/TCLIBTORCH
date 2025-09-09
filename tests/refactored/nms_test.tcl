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

# Helper function to create test boxes and scores
proc create_test_boxes {} {
    # Create boxes in format [x1, y1, x2, y2]
    set boxes [torch::tensor_create -data {0.0 0.0 1.0 1.0 0.1 0.1 1.1 1.1 0.9 0.9 1.9 1.9 0.2 0.2 0.8 0.8} -shape {4 4} -dtype float32]
    
    # Create corresponding scores
    set scores [torch::tensor_create -data {0.9 0.8 0.7 0.6} -dtype float32]
    
    return [list $boxes $scores]
}

# Test cases for positional syntax
test nms-1.1 {Basic positional syntax} {
    lassign [create_test_boxes] boxes scores
    set indices [torch::nms $boxes $scores 0.5]
    expr {[string match "tensor*" $indices]}
} {1}

test nms-1.2 {Positional syntax with score threshold} {
    lassign [create_test_boxes] boxes scores
    set indices [torch::nms $boxes $scores 0.5 0.7]
    expr {[string match "tensor*" $indices]}
} {1}

test nms-1.3 {Error on missing arguments} {
    catch {torch::nms} msg
    set msg
} {Usage: torch::nms boxes scores iou_threshold ?score_threshold? | torch::nms -boxes boxes -scores scores -iouThreshold value ?-scoreThreshold value?}

# Test cases for named syntax
test nms-2.1 {Basic named syntax} {
    lassign [create_test_boxes] boxes scores
    set indices [torch::nms -boxes $boxes -scores $scores -iouThreshold 0.5]
    expr {[string match "tensor*" $indices]}
} {1}

test nms-2.2 {Named syntax with score threshold} {
    lassign [create_test_boxes] boxes scores
    set indices [torch::nms -boxes $boxes -scores $scores -iouThreshold 0.5 -scoreThreshold 0.7]
    expr {[string match "tensor*" $indices]}
} {1}

test nms-2.3 {Error on missing required parameters} {
    lassign [create_test_boxes] boxes scores
    catch {torch::nms -boxes $boxes} msg
    set msg
} {Named syntax requires at least -boxes, -scores, and -iouThreshold parameters}

test nms-2.4 {Error on invalid parameter} {
    lassign [create_test_boxes] boxes scores
    catch {torch::nms -boxes $boxes -scores $scores -invalid 0.5} msg
    set msg
} {Unknown parameter: -invalid. Valid parameters are: -boxes, -scores, -iouThreshold, -scoreThreshold}

# Test cases for camelCase alias
test nms-3.1 {CamelCase alias} {
    lassign [create_test_boxes] boxes scores
    set indices [torch::Nms -boxes $boxes -scores $scores -iouThreshold 0.5]
    expr {[string match "tensor*" $indices]}
} {1}

# Test cases for validation
test nms-4.1 {Invalid boxes tensor} {
    lassign [create_test_boxes] boxes scores
    catch {torch::nms -boxes "invalid" -scores $scores -iouThreshold 0.5} msg
    set msg
} {Invalid boxes tensor}

test nms-4.2 {Invalid scores tensor} {
    lassign [create_test_boxes] boxes scores
    catch {torch::nms -boxes $boxes -scores "invalid" -iouThreshold 0.5} msg
    set msg
} {Invalid scores tensor}

test nms-4.3 {Invalid iou threshold} {
    lassign [create_test_boxes] boxes scores
    catch {torch::nms -boxes $boxes -scores $scores -iouThreshold 1.5} msg
    set msg
} {iouThreshold must be between 0.0 and 1.0}

cleanupTests
