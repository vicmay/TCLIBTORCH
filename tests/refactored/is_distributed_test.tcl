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

# Test cases for basic functionality
test is_distributed-1.1 {Basic functionality - returns boolean} {
    set result [torch::is_distributed]
    expr {$result == 0 || $result == 1}
} {1}

test is_distributed-1.2 {Returns boolean type} {
    set result [torch::is_distributed]
    string is boolean $result
} {1}

test is_distributed-1.3 {Consistent results across calls} {
    set result1 [torch::is_distributed]
    set result2 [torch::is_distributed]
    expr {$result1 == $result2}
} {1}

test is_distributed-1.4 {Default is false in non-distributed mode} {
    ;# In single-process mode, should return false
    set result [torch::is_distributed]
    expr {$result == 0}
} {1}

# Test cases for camelCase alias
test is_distributed-2.1 {CamelCase alias functionality} {
    set result [torch::isDistributed]
    expr {$result == 0 || $result == 1}
} {1}

test is_distributed-2.2 {CamelCase alias returns boolean type} {
    set result [torch::isDistributed]
    string is boolean $result
} {1}

test is_distributed-2.3 {Both syntaxes return same result} {
    set result1 [torch::is_distributed]
    set result2 [torch::isDistributed]
    expr {$result1 == $result2}
} {1}

# Test cases for error handling
test is_distributed-3.1 {No arguments accepted - snake_case} {
    set result [catch {torch::is_distributed extra_arg} msg]
    list $result [string match "*wrong # args*" $msg]
} {1 1}

test is_distributed-3.2 {No arguments accepted - camelCase} {
    set result [catch {torch::isDistributed extra_arg} msg]
    list $result [string match "*wrong # args*" $msg]
} {1 1}

test is_distributed-3.3 {Multiple arguments rejected - snake_case} {
    set result [catch {torch::is_distributed arg1 arg2} msg]
    list $result [string match "*wrong # args*" $msg]
} {1 1}

test is_distributed-3.4 {Multiple arguments rejected - camelCase} {
    set result [catch {torch::isDistributed arg1 arg2} msg]
    list $result [string match "*wrong # args*" $msg]
} {1 1}

# Test cases for single GPU initialization
test is_distributed-4.1 {Single GPU initialization - not distributed} {
    torch::distributed_init 0 1 "gloo"
    set result [torch::is_distributed]
    expr {$result == 0}
} {1}

test is_distributed-4.2 {Single GPU - camelCase alias} {
    set result [torch::isDistributed]
    expr {$result == 0}
} {1}

test is_distributed-4.3 {Single GPU - consistency between syntaxes} {
    set result1 [torch::is_distributed]
    set result2 [torch::isDistributed]
    expr {$result1 == $result2}
} {1}

# Test cases for multi-GPU initialization
test is_distributed-5.1 {Multi-GPU initialization - is distributed} {
    torch::distributed_init 0 4 "nccl"
    set result [torch::is_distributed]
    expr {$result == 1}
} {1}

test is_distributed-5.2 {Multi-GPU - camelCase alias} {
    set result [torch::isDistributed]
    expr {$result == 1}
} {1}

test is_distributed-5.3 {Multi-GPU - consistency between syntaxes} {
    set result1 [torch::is_distributed]
    set result2 [torch::isDistributed]
    expr {$result1 == $result2}
} {1}

# Test cases for integration with other distributed functions
test is_distributed-6.1 {Integration with get_rank} {
    set is_distributed [torch::is_distributed]
    set rank [torch::get_rank]
    
    ;# Both should return valid values
    expr {($is_distributed == 0 || $is_distributed == 1) && $rank >= 0}
} {1}

test is_distributed-6.2 {Integration with get_world_size} {
    set is_distributed [torch::is_distributed]
    set world_size [torch::get_world_size]
    
    ;# If distributed, world_size should be > 1
    ;# If not distributed, world_size should be 1
    if {$is_distributed} {
        expr {$world_size > 1}
    } else {
        expr {$world_size == 1}
    }
} {1}

test is_distributed-6.3 {Logical consistency with world_size} {
    set is_distributed [torch::is_distributed]
    set world_size [torch::get_world_size]
    
    ;# is_distributed should be true if and only if world_size > 1
    expr {($is_distributed == 1) == ($world_size > 1)}
} {1}

# Test cases for state transitions
test is_distributed-7.1 {State change from single to multi-GPU} {
    ;# Initialize single GPU
    torch::distributed_init 0 1 "gloo"
    set single_result [torch::is_distributed]
    
    ;# Initialize multi-GPU
    torch::distributed_init 0 4 "nccl"
    set multi_result [torch::is_distributed]
    
    ;# Single should be false, multi should be true
    expr {$single_result == 0 && $multi_result == 1}
} {1}

test is_distributed-7.2 {State change - camelCase consistency} {
    ;# Test with camelCase during state change
    torch::distributed_init 0 1 "gloo"
    set single_result [torch::isDistributed]
    
    torch::distributed_init 0 2 "nccl"
    set multi_result [torch::isDistributed]
    
    expr {$single_result == 0 && $multi_result == 1}
} {1}

# Test cases for multiple calls consistency
test is_distributed-8.1 {Multiple calls are consistent} {
    set results {}
    for {set i 0} {$i < 5} {incr i} {
        lappend results [torch::is_distributed]
    }
    
    ;# Check all results are the same
    set first [lindex $results 0]
    set all_same 1
    foreach result $results {
        if {$result != $first} {
            set all_same 0
            break
        }
    }
    set all_same
} {1}

test is_distributed-8.2 {Multiple calls with camelCase are consistent} {
    set results {}
    for {set i 0} {$i < 5} {incr i} {
        lappend results [torch::isDistributed]
    }
    
    ;# Check all results are the same
    set first [lindex $results 0]
    set all_same 1
    foreach result $results {
        if {$result != $first} {
            set all_same 0
            break
        }
    }
    set all_same
} {1}

# Test cases for different backend types
test is_distributed-9.1 {Different backends - gloo single GPU} {
    torch::distributed_init 0 1 "gloo"
    set result [torch::is_distributed]
    expr {$result == 0}
} {1}

test is_distributed-9.2 {Different backends - nccl multi-GPU} {
    torch::distributed_init 0 4 "nccl"
    set result [torch::is_distributed]
    expr {$result == 1}
} {1}

test is_distributed-9.3 {Different backends - mpi multi-GPU} {
    torch::distributed_init 0 3 "mpi"
    set result [torch::is_distributed]
    expr {$result == 1}
} {1}

cleanupTests 