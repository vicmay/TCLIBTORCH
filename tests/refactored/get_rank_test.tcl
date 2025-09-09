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
test get_rank-1.1 {Basic functionality - returns non-negative integer} {
    set rank [torch::get_rank]
    expr {$rank >= 0}
} {1}

test get_rank-1.2 {Returns integer type} {
    set rank [torch::get_rank]
    string is integer $rank
} {1}

test get_rank-1.3 {Consistent results across calls} {
    set rank1 [torch::get_rank]
    set rank2 [torch::get_rank]
    expr {$rank1 == $rank2}
} {1}

test get_rank-1.4 {Default rank is zero in non-distributed mode} {
    ;# In single-process mode, rank should be 0
    set rank [torch::get_rank]
    expr {$rank == 0}
} {1}

# Test cases for camelCase alias
test get_rank-2.1 {CamelCase alias functionality} {
    set rank [torch::getRank]
    expr {$rank >= 0}
} {1}

test get_rank-2.2 {CamelCase alias returns integer type} {
    set rank [torch::getRank]
    string is integer $rank
} {1}

test get_rank-2.3 {Both syntaxes return same result} {
    set rank1 [torch::get_rank]
    set rank2 [torch::getRank]
    expr {$rank1 == $rank2}
} {1}

# Test cases for error handling
test get_rank-3.1 {No arguments accepted - snake_case} {
    set result [catch {torch::get_rank extra_arg} msg]
    list $result [string match "*wrong # args*" $msg]
} {1 1}

test get_rank-3.2 {No arguments accepted - camelCase} {
    set result [catch {torch::getRank extra_arg} msg]
    list $result [string match "*wrong # args*" $msg]
} {1 1}

# Test cases for reasonable values
test get_rank-4.1 {Returns reasonable rank value} {
    set rank [torch::get_rank]
    ;# Rank should be non-negative and reasonable (0-1000 for most cases)
    expr {$rank >= 0 && $rank <= 1000}
} {1}

test get_rank-4.2 {Rank is valid for single process} {
    ;# In non-distributed mode, rank should be 0
    set rank [torch::get_rank]
    expr {$rank == 0}
} {1}

# Test cases for distributed training integration
test get_rank-5.1 {Integration with distributed initialization} {
    ;# Initialize distributed training (simulated)
    set init_result [catch {torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1"} msg]
    
    ;# Get rank after initialization
    set rank [torch::get_rank]
    
    ;# Rank should be 0 (as set in initialization)
    expr {$rank == 0}
} {1}

test get_rank-5.2 {Integration with world size} {
    ;# Get rank and world size
    set rank [torch::get_rank]
    set world_size [torch::get_world_size]
    
    ;# Rank should be less than world size
    expr {$rank < $world_size}
} {1}

test get_rank-5.3 {Rank consistency with camelCase in distributed context} {
    set rank1 [torch::get_rank]
    set rank2 [torch::getRank]
    
    ;# Both should return the same value
    expr {$rank1 == $rank2}
} {1}

# Test cases for multiple calls consistency
test get_rank-6.1 {Multiple calls are consistent} {
    set results {}
    for {set i 0} {$i < 5} {incr i} {
        lappend results [torch::get_rank]
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

test get_rank-6.2 {Multiple calls with camelCase are consistent} {
    set results {}
    for {set i 0} {$i < 5} {incr i} {
        lappend results [torch::getRank]
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
test get_rank-7.1 {Rank is valid relative to world size} {
    set rank [torch::get_rank]
    set world_size [torch::get_world_size]
    
    ;# Rank should be in range [0, world_size)
    expr {$rank >= 0 && $rank < $world_size}
} {1}

test get_rank-7.2 {Distributed state consistency} {
    set rank [torch::get_rank]
    set is_distributed [torch::is_distributed]
    
    ;# In non-distributed mode, rank should be 0
    if {!$is_distributed} {
        expr {$rank == 0}
    } else {
        expr {$rank >= 0}
    }
} {1}

cleanupTests 