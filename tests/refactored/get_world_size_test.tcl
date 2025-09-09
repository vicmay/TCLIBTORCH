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
test get_world_size-1.1 {Basic functionality - returns positive integer} {
    set world_size [torch::get_world_size]
    expr {$world_size > 0}
} {1}

test get_world_size-1.2 {Returns integer type} {
    set world_size [torch::get_world_size]
    string is integer $world_size
} {1}

test get_world_size-1.3 {Consistent results across calls} {
    set world_size1 [torch::get_world_size]
    set world_size2 [torch::get_world_size]
    expr {$world_size1 == $world_size2}
} {1}

test get_world_size-1.4 {Default world size is 1 in non-distributed mode} {
    ;# In single-process mode, world size should be 1
    set world_size [torch::get_world_size]
    expr {$world_size == 1}
} {1}

# Test cases for camelCase alias
test get_world_size-2.1 {CamelCase alias functionality} {
    set world_size [torch::getWorldSize]
    expr {$world_size > 0}
} {1}

test get_world_size-2.2 {CamelCase alias returns integer type} {
    set world_size [torch::getWorldSize]
    string is integer $world_size
} {1}

test get_world_size-2.3 {Both syntaxes return same result} {
    set world_size1 [torch::get_world_size]
    set world_size2 [torch::getWorldSize]
    expr {$world_size1 == $world_size2}
} {1}

# Test cases for error handling
test get_world_size-3.1 {No arguments accepted - snake_case} {
    set result [catch {torch::get_world_size extra_arg} msg]
    list $result [string match "*wrong # args*" $msg]
} {1 1}

test get_world_size-3.2 {No arguments accepted - camelCase} {
    set result [catch {torch::getWorldSize extra_arg} msg]
    list $result [string match "*wrong # args*" $msg]
} {1 1}

# Test cases for reasonable values
test get_world_size-4.1 {Returns reasonable world size value} {
    set world_size [torch::get_world_size]
    ;# World size should be positive and reasonable (1-1000 for most cases)
    expr {$world_size > 0 && $world_size <= 1000}
} {1}

test get_world_size-4.2 {World size is valid for single process} {
    ;# In non-distributed mode, world size should be 1
    set world_size [torch::get_world_size]
    expr {$world_size == 1}
} {1}

# Test cases for distributed training integration
test get_world_size-5.1 {Integration with distributed initialization} {
    ;# Initialize distributed training (simulated)
    set init_result [catch {torch::distributed_init -rank 0 -worldSize 2 -masterAddr "127.0.0.1"} msg]
    
    ;# Get world size after initialization
    set world_size [torch::get_world_size]
    
    ;# World size should be 2 (as set in initialization)
    expr {$world_size == 2}
} {1}

test get_world_size-5.2 {Integration with rank} {
    ;# Get rank and world size
    set rank [torch::get_rank]
    set world_size [torch::get_world_size]
    
    ;# Rank should be less than world size
    expr {$rank < $world_size}
} {1}

test get_world_size-5.3 {World size consistency with camelCase in distributed context} {
    set world_size1 [torch::get_world_size]
    set world_size2 [torch::getWorldSize]
    
    ;# Both should return the same value
    expr {$world_size1 == $world_size2}
} {1}

# Test cases for multiple calls consistency
test get_world_size-6.1 {Multiple calls are consistent} {
    set results {}
    for {set i 0} {$i < 5} {incr i} {
        lappend results [torch::get_world_size]
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

test get_world_size-6.2 {Multiple calls with camelCase are consistent} {
    set results {}
    for {set i 0} {$i < 5} {incr i} {
        lappend results [torch::getWorldSize]
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

# Test cases for distributed properties
test get_world_size-7.1 {World size is valid relative to rank} {
    set rank [torch::get_rank]
    set world_size [torch::get_world_size]
    
    ;# Rank should be in range [0, world_size)
    expr {$rank >= 0 && $rank < $world_size}
} {1}

test get_world_size-7.2 {Distributed state consistency} {
    set world_size [torch::get_world_size]
    set is_distributed [torch::is_distributed]
    
    ;# In non-distributed mode, world size should be 1
    if {!$is_distributed} {
        expr {$world_size == 1}
    } else {
        expr {$world_size > 1}
    }
} {1}

test get_world_size-7.3 {World size is always positive} {
    set world_size [torch::get_world_size]
    expr {$world_size > 0}
} {1}

cleanupTests 