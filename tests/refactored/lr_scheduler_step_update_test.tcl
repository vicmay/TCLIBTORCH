#!/usr/bin/env tclsh

package require tcltest
namespace import tcltest::*

if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

proc makeScheduler {} {
    set p [torch::tensor_create {1.0 2.0} float32]
    set opt [torch::optimizer_sgd $p 0.1]
    set sch [torch::lr_scheduler_lambda $opt 0.9]
    return $sch
}

set sch1 [makeScheduler]

test step_update-1.1 {Positional syntax} {
    set res [torch::lr_scheduler_step_update $sch1]
    set res
} {OK}

test step_update-1.2 {Named parameter syntax} {
    set res [torch::lr_scheduler_step_update -scheduler $sch1]
    set res
} {OK}

test step_update-1.3 {CamelCase alias positional} {
    set res [torch::lrSchedulerStepUpdate $sch1]
    set res
} {OK}

test step_update-1.4 {CamelCase alias named} {
    set res [torch::lrSchedulerStepUpdate -scheduler $sch1]
    set res
} {OK}

# Error handling

test step_update-2.1 {Missing scheduler} {
    catch {torch::lr_scheduler_step_update} msg
    string match "*scheduler*" $msg
} {1}

test step_update-2.2 {Invalid scheduler} {
    catch {torch::lr_scheduler_step_update invalid} msg
    string match "*Invalid scheduler name*" $msg
} {1}

test step_update-2.3 {Unknown parameter} {
    catch {torch::lr_scheduler_step_update -foo bar} msg
    string match "*Unknown parameter*" $msg
} {1}

test step_update-2.4 {Missing value after parameter} {
    catch {torch::lr_scheduler_step_update -scheduler} msg
    string match "*Missing value for parameter*" $msg
} {1}

cleanupTests 