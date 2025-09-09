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

# Test cases for profiler_start
test profiler_start-1.1 {Basic profiler_start} {
    torch::profiler_start
} {profiler_started}

test profiler_start-1.2 {Basic profiler_start with config} {
    torch::profiler_start "verbose=1"
} {profiler_started}

test profiler_start-2.1 {CamelCase alias} {
    torch::profilerStart
} {profiler_started}

test profiler_start-2.2 {CamelCase alias with config} {
    torch::profilerStart "verbose=1"
} {profiler_started}

# Test cases for profiler_stop
test profiler_stop-1.1 {Basic profiler_stop} {
    torch::profiler_stop
} {profiler_stopped}

test profiler_stop-2.1 {CamelCase alias} {
    torch::profilerStop
} {profiler_stopped}

# Test profiling workflow
test profiler-3.1 {Complete profiling workflow} {
    torch::profiler_start
    set t1 [torch::ones {2 2}]
    set t2 [torch::ones {2 2}]
    set t3 [torch::tensor_add $t1 $t2]
    torch::profiler_stop
} {profiler_stopped}

test profiler-3.2 {Complete profiling workflow with camelCase} {
    torch::profilerStart
    set t1 [torch::ones {2 2}]
    set t2 [torch::ones {2 2}]
    set t3 [torch::tensor_add $t1 $t2]
    torch::profilerStop
} {profiler_stopped}

cleanupTests 