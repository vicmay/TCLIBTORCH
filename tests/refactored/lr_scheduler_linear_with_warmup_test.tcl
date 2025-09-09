#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so} err]} {
    puts "Failed to load libtorchtcl.so: $err"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Helper to create a simple optimizer (SGD) for testing
proc create_test_optimizer {} {
    set p1 [torch::zeros {3 3}]
    set p2 [torch::zeros {5}]
    # Positional syntax for optimizer_sgd: parameters lr ?momentum? ?dampening? ?weightDecay? ?nesterov?
    set opt [torch::optimizer_sgd [list $p1 $p2] 0.1]
    return $opt
}

# 1. Positional syntax (legacy)

test lr_scheduler_linear_warmup-1.1 {Positional syntax basic} {
    set opt [create_test_optimizer]
    set sched [torch::lr_scheduler_linear_with_warmup $opt 10 100]
    string match "linear_warmup_scheduler*" $sched
} {1}

# 2. Named parameter syntax (modern)

test lr_scheduler_linear_warmup-2.1 {Named parameter syntax} {
    set opt [create_test_optimizer]
    set sched [torch::lr_scheduler_linear_with_warmup -optimizer $opt -numWarmupSteps 20 -numTrainingSteps 200]
    string match "linear_warmup_scheduler*" $sched
} {1}

test lr_scheduler_linear_warmup-2.2 {Named syntax with alternative aliases} {
    set opt [create_test_optimizer]
    set sched [torch::lr_scheduler_linear_with_warmup -optimizer $opt -num_warmup_steps 5 -num_training_steps 50 -last_epoch 9]
    string match "linear_warmup_scheduler*" $sched
} {1}

# 3. camelCase alias

test lr_scheduler_linear_warmup-3.1 {CamelCase alias positional} {
    set opt [create_test_optimizer]
    set sched [torch::lrSchedulerLinearWithWarmup $opt 15 150]
    string match "linear_warmup_scheduler*" $sched
} {1}

# 4. Error handling

test lr_scheduler_linear_warmup-4.1 {Error: missing required parameters} -body {
    torch::lr_scheduler_linear_with_warmup -optimizer invalid
} -returnCodes error -match glob -result *

test lr_scheduler_linear_warmup-4.2 {Error: invalid optimizer} -body {
    torch::lr_scheduler_linear_with_warmup invalid 10 100
} -returnCodes error -match glob -result *

cleanupTests 