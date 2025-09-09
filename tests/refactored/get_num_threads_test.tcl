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
test get_num_threads-1.1 {Basic functionality - returns positive integer} {
    set num_threads [torch::get_num_threads]
    expr {$num_threads > 0}
} {1}

test get_num_threads-1.2 {Returns integer type} {
    set num_threads [torch::get_num_threads]
    string is integer $num_threads
} {1}

test get_num_threads-1.3 {Consistent results across calls} {
    set num1 [torch::get_num_threads]
    set num2 [torch::get_num_threads]
    expr {$num1 == $num2}
} {1}

# Test cases for camelCase alias
test get_num_threads-2.1 {CamelCase alias functionality} {
    set num_threads [torch::getNumThreads]
    expr {$num_threads > 0}
} {1}

test get_num_threads-2.2 {CamelCase alias returns integer type} {
    set num_threads [torch::getNumThreads]
    string is integer $num_threads
} {1}

test get_num_threads-2.3 {Both syntaxes return same result} {
    set num1 [torch::get_num_threads]
    set num2 [torch::getNumThreads]
    expr {$num1 == $num2}
} {1}

# Test cases for error handling
test get_num_threads-3.1 {No arguments accepted - snake_case} {
    set result [catch {torch::get_num_threads extra_arg} msg]
    list $result [string match "*wrong # args*" $msg]
} {1 1}

test get_num_threads-3.2 {No arguments accepted - camelCase} {
    set result [catch {torch::getNumThreads extra_arg} msg]
    list $result [string match "*wrong # args*" $msg]
} {1 1}

# Test cases for reasonable values
test get_num_threads-4.1 {Returns reasonable number of threads} {
    set num_threads [torch::get_num_threads]
    expr {$num_threads >= 1 && $num_threads <= 1024}
} {1}

test get_num_threads-4.2 {Number of threads matches system capabilities} {
    set num_threads [torch::get_num_threads]
    ;# Should be reasonable for most systems (1-128 cores typical)
    expr {$num_threads >= 1 && $num_threads <= 128}
} {1}

# Test cases for integration with set_num_threads
test get_num_threads-5.1 {Integration with set_num_threads} {
    ;# Get original value
    set original_threads [torch::get_num_threads]
    
    ;# Set to a specific value
    torch::set_num_threads 4
    set new_threads [torch::get_num_threads]
    
    ;# Restore original value
    torch::set_num_threads $original_threads
    set restored_threads [torch::get_num_threads]
    
    ;# Check that setting worked and restoration worked
    expr {$new_threads == 4 && $restored_threads == $original_threads}
} {1}

test get_num_threads-5.2 {Integration with set_num_threads using camelCase} {
    ;# Get original value using camelCase
    set original_threads [torch::getNumThreads]
    
    ;# Set to a specific value
    torch::set_num_threads 2
    set new_threads [torch::getNumThreads]
    
    ;# Restore original value
    torch::set_num_threads $original_threads
    set restored_threads [torch::getNumThreads]
    
    ;# Check that setting worked and restoration worked
    expr {$new_threads == 2 && $restored_threads == $original_threads}
} {1}

# Test cases for multiple calls consistency
test get_num_threads-6.1 {Multiple calls are consistent} {
    set results {}
    for {set i 0} {$i < 5} {incr i} {
        lappend results [torch::get_num_threads]
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

test get_num_threads-6.2 {Multiple calls with camelCase are consistent} {
    set results {}
    for {set i 0} {$i < 5} {incr i} {
        lappend results [torch::getNumThreads]
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

cleanupTests 