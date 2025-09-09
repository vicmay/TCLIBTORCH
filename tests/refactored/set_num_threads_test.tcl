#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load the extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# Test cases for positional syntax
test set_num_threads-1.1 {Basic positional syntax} {
    torch::set_num_threads 4
    torch::get_num_threads
} {4}

test set_num_threads-1.2 {Minimum threads} {
    torch::set_num_threads 1
    torch::get_num_threads
} {1}

;# Test cases for named syntax
test set_num_threads-2.1 {Named parameter syntax with -numThreads} {
    torch::set_num_threads -numThreads 2
    torch::get_num_threads
} {2}

test set_num_threads-2.2 {Named parameter syntax with -num_threads} {
    torch::set_num_threads -num_threads 3
    torch::get_num_threads
} {3}

;# Test cases for camelCase alias
test set_num_threads-3.1 {CamelCase alias} {
    torch::setNumThreads -numThreads 4
    torch::get_num_threads
} {4}

;# Error handling tests
test set_num_threads-4.1 {Error handling - missing argument} -body {
    torch::set_num_threads
} -returnCodes error -result {Error in set_num_threads: Usage: torch::set_num_threads num_threads | torch::set_num_threads -numThreads value}

test set_num_threads-4.2 {Error handling - invalid argument type} -body {
    torch::set_num_threads invalid
} -returnCodes error -result {Error in set_num_threads: Invalid num_threads value (must be a positive integer)}

test set_num_threads-4.3 {Error handling - negative threads} -body {
    torch::set_num_threads -1
} -returnCodes error -result {Error in set_num_threads: Named parameters must come in pairs}

test set_num_threads-4.4 {Error handling - zero threads} -body {
    torch::set_num_threads 0
} -returnCodes error -result {Error in set_num_threads: Number of threads must be positive}

test set_num_threads-4.5 {Error handling - too many arguments} -body {
    torch::set_num_threads 1 2
} -returnCodes error -result {Error in set_num_threads: Usage: torch::set_num_threads num_threads}

;# Verify state is preserved
test set_num_threads-5.1 {State preservation} {
    torch::set_num_threads 2
    set result1 [torch::get_num_threads]
    torch::set_num_threads 4
    set result2 [torch::get_num_threads]
    torch::set_num_threads 1
    set result3 [torch::get_num_threads]
    list $result1 $result2 $result3
} {2 4 1}

cleanupTests 