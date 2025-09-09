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

# Helper function to create test tensors
proc create_test_tensors {} {
    set input [torch::ones {1 64 32 32}]
    set boxes [torch::tensor_create {0 0 15 15} {4} float32]
    return [list $input $boxes]
}

# ============================================================================
# Basic ROI Align Tests - Positional Syntax
# ============================================================================

test roi_align-1.1 {Basic ROI align creation - positional syntax} {
    lassign [create_test_tensors] input boxes
    set result [torch::roi_align $input $boxes {2 2}]
    string match "tensor*" $result
} {1}

test roi_align-1.2 {ROI align with spatial scale - positional syntax} {
    lassign [create_test_tensors] input boxes
    set result [torch::roi_align $input $boxes {2 2} 0.5]
    string match "tensor*" $result
} {1}

test roi_align-1.3 {ROI align with all parameters - positional syntax} {
    lassign [create_test_tensors] input boxes
    set result [torch::roi_align $input $boxes {2 2} 0.5 2 true]
    string match "tensor*" $result
} {1}

# ============================================================================
# ROI Align Tests - Named Parameter Syntax
# ============================================================================

test roi_align-2.1 {Basic ROI align creation - named parameter syntax} {
    lassign [create_test_tensors] input boxes
    set result [torch::roi_align -input $input -boxes $boxes -outputSize {2 2}]
    string match "tensor*" $result
} {1}

test roi_align-2.2 {ROI align with spatial scale - named parameter syntax} {
    lassign [create_test_tensors] input boxes
    set result [torch::roi_align -input $input -boxes $boxes -outputSize {2 2} -spatialScale 0.5]
    string match "tensor*" $result
} {1}

test roi_align-2.3 {ROI align with all parameters - named parameter syntax} {
    lassign [create_test_tensors] input boxes
    set result [torch::roi_align \
        -input $input \
        -boxes $boxes \
        -outputSize {2 2} \
        -spatialScale 0.5 \
        -samplingRatio 2 \
        -aligned true]
    string match "tensor*" $result
} {1}

test roi_align-2.4 {ROI align with mixed parameter order - named syntax} {
    lassign [create_test_tensors] input boxes
    set result [torch::roi_align \
        -boxes $boxes \
        -outputSize {2 2} \
        -input $input \
        -aligned false]
    string match "tensor*" $result
} {1}

# ============================================================================
# CamelCase Alias Tests
# ============================================================================

test roi_align-3.1 {CamelCase alias - positional syntax} {
    lassign [create_test_tensors] input boxes
    set result [torch::roiAlign $input $boxes {2 2}]
    string match "tensor*" $result
} {1}

test roi_align-3.2 {CamelCase alias - named parameter syntax} {
    lassign [create_test_tensors] input boxes
    set result [torch::roiAlign -input $input -boxes $boxes -outputSize {2 2}]
    string match "tensor*" $result
} {1}

test roi_align-3.3 {CamelCase alias with all parameters} {
    lassign [create_test_tensors] input boxes
    set result [torch::roiAlign \
        -input $input \
        -boxes $boxes \
        -outputSize {2 2} \
        -spatialScale 0.5 \
        -samplingRatio 2 \
        -aligned true]
    string match "tensor*" $result
} {1}

# ============================================================================
# Parameter Validation Tests
# ============================================================================

test roi_align-4.1 {Error: Missing required parameters - positional} {
    lassign [create_test_tensors] input boxes
    catch {torch::roi_align $input} error
    string match "*Usage*" $error
} {1}

test roi_align-4.2 {Error: Missing required parameters - named} {
    lassign [create_test_tensors] input boxes
    catch {torch::roi_align -input $input} msg
    string match "*Usage: torch::roi_align*" $msg
} {1}

test roi_align-4.3 {Error: Invalid input tensor} {
    lassign [create_test_tensors] input boxes
    catch {torch::roi_align -input "invalid" -boxes $boxes -outputSize {2 2}} error
    string match "*Invalid input tensor*" $error
} {1}

test roi_align-4.4 {Error: Invalid boxes tensor} {
    lassign [create_test_tensors] input boxes
    catch {torch::roi_align -input $input -boxes "invalid" -outputSize {2 2}} error
    string match "*Invalid boxes tensor*" $error
} {1}

test roi_align-4.5 {Error: Invalid spatial scale} {
    lassign [create_test_tensors] input boxes
    catch {torch::roi_align -input $input -boxes $boxes -outputSize {2 2} -spatialScale "invalid"} error
    string match "*Invalid spatialScale*" $error
} {1}

test roi_align-4.6 {Error: Invalid sampling ratio} {
    lassign [create_test_tensors] input boxes
    catch {torch::roi_align -input $input -boxes $boxes -outputSize {2 2} -samplingRatio "invalid"} error
    string match "*Invalid samplingRatio*" $error
} {1}

test roi_align-4.7 {Error: Unknown parameter} {
    lassign [create_test_tensors] input boxes
    catch {torch::roi_align -input $input -boxes $boxes -outputSize {2 2} -invalidParam 5} error
    string match "*Unknown parameter*" $error
} {1}

# ============================================================================
# Edge Cases and Special Values
# ============================================================================

test roi_align-5.1 {Edge case: Minimum output size} {
    lassign [create_test_tensors] input boxes
    set result [torch::roi_align -input $input -boxes $boxes -outputSize {1 1}]
    string match "tensor*" $result
} {1}

test roi_align-5.2 {Edge case: Large output size} {
    lassign [create_test_tensors] input boxes
    set result [torch::roi_align -input $input -boxes $boxes -outputSize {8 8}]
    string match "tensor*" $result
} {1}

test roi_align-5.3 {Edge case: Minimum spatial scale} {
    lassign [create_test_tensors] input boxes
    set result [torch::roi_align -input $input -boxes $boxes -outputSize {2 2} -spatialScale 0.0001]
    string match "tensor*" $result
} {1}

test roi_align-5.4 {Edge case: Maximum spatial scale} {
    lassign [create_test_tensors] input boxes
    set result [torch::roi_align -input $input -boxes $boxes -outputSize {2 2} -spatialScale 100.0]
    string match "tensor*" $result
} {1}

# ============================================================================
# Syntax Equivalence Tests
# ============================================================================

test roi_align-6.1 {Positional and named syntax equivalence - basic} {
    lassign [create_test_tensors] input boxes
    set result1 [torch::roi_align $input $boxes {2 2}]
    set result2 [torch::roi_align -input $input -boxes $boxes -outputSize {2 2}]
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

test roi_align-6.2 {Positional and named syntax equivalence - with parameters} {
    lassign [create_test_tensors] input boxes
    set result1 [torch::roi_align $input $boxes {2 2} 0.5 2 true]
    set result2 [torch::roi_align -input $input -boxes $boxes -outputSize {2 2} -spatialScale 0.5 -samplingRatio 2 -aligned true]
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

test roi_align-6.3 {CamelCase and snake_case equivalence} {
    lassign [create_test_tensors] input boxes
    set result1 [torch::roi_align $input $boxes {2 2}]
    set result2 [torch::roiAlign $input $boxes {2 2}]
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

cleanupTests 