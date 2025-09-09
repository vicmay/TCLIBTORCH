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

# =====================================================================
# TORCH::LR_SCHEDULER_STEP_ADVANCED COMPREHENSIVE TEST SUITE
# =====================================================================

# Helper to create a dummy optimizer and scheduler
proc createScheduler {} {
    # Create a dummy tensor parameter and simple SGD optimizer (matches helper in lambda tests)
    set param [torch::tensor_create {1.0 2.0 3.0} float32]
    set optimizer [torch::optimizer_sgd $param 0.01]
    # Use lambda scheduler which is already dual-syntax compliant
    set scheduler [torch::lr_scheduler_lambda $optimizer]
    return $scheduler
}

# Tests for positional syntax (backward compatibility)
set sched [createScheduler]

test lr_scheduler_step_advanced-1.1 {Positional syntax without metric} {
    set result [torch::lr_scheduler_step_advanced $sched]
    set result
} {OK}

test lr_scheduler_step_advanced-1.2 {Positional syntax with metric} {
    set result [torch::lr_scheduler_step_advanced $sched 0.95]
    set result
} {OK}

# Tests for named parameter syntax
set sched2 [createScheduler]

test lr_scheduler_step_advanced-2.1 {Named parameter syntax basic} {
    set result [torch::lr_scheduler_step_advanced -scheduler $sched2]
    set result
} {OK}

test lr_scheduler_step_advanced-2.2 {Named parameter syntax with metric} {
    set result [torch::lr_scheduler_step_advanced -scheduler $sched2 -metric 0.9]
    set result
} {OK}

# Tests for camelCase alias
set sched3 [createScheduler]

test lr_scheduler_step_advanced-3.1 {CamelCase alias basic} {
    set result [torch::lrSchedulerStepAdvanced $sched3]
    set result
} {OK}

test lr_scheduler_step_advanced-3.2 {CamelCase alias named parameters} {
    set result [torch::lrSchedulerStepAdvanced -scheduler $sched3 -metric 1.1]
    set result
} {OK}

# Error handling tests

test lr_scheduler_step_advanced-4.1 {Error on missing scheduler} {
    catch {torch::lr_scheduler_step_advanced} msg
    string match "*scheduler*" $msg
} {1}

test lr_scheduler_step_advanced-4.2 {Error on invalid scheduler handle} {
    catch {torch::lr_scheduler_step_advanced invalid_handle} msg
    string match "*Invalid scheduler handle*" $msg
} {1}

test lr_scheduler_step_advanced-4.3 {Error on unknown named parameter} {
    catch {torch::lr_scheduler_step_advanced -scheduler $sched -foo 1} msg
    string match "*Unknown parameter*" $msg
} {1}

test lr_scheduler_step_advanced-4.4 {Error on missing value for parameter} {
    catch {torch::lr_scheduler_step_advanced -scheduler} msg
    string match "*Missing value for parameter*" $msg
} {1}

cleanupTests 