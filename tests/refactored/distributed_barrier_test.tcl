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

# Test basic functionality (without distributed initialization)
test distributed_barrier-1.1 {Barrier without initialization} {
    set result [torch::distributed_barrier]
    string equal $result "Distributed not initialized"
} {1}

test distributed_barrier-1.2 {CamelCase alias without initialization} {
    set result [torch::distributedBarrier]
    string equal $result "Distributed not initialized"
} {1}

test distributed_barrier-1.3 {Consistency between syntaxes without init} {
    set result1 [torch::distributed_barrier]
    set result2 [torch::distributedBarrier]
    string equal $result1 $result2
} {1}

# Test with distributed initialization (single GPU)
test distributed_barrier-2.1 {Barrier with single GPU initialization} {
    torch::distributed_init 0 1 "gloo"
    set result [torch::distributed_barrier]
    string equal $result "Barrier synchronized (single GPU)"
} {1}

test distributed_barrier-2.2 {CamelCase alias with initialization} {
    set result [torch::distributedBarrier]
    string equal $result "Barrier synchronized (single GPU)"
} {1}

test distributed_barrier-2.3 {Multiple barrier calls} {
    set result1 [torch::distributed_barrier]
    set result2 [torch::distributedBarrier]
    set result3 [torch::distributed_barrier]
    
    set valid1 [string equal $result1 "Barrier synchronized (single GPU)"]
    set valid2 [string equal $result2 "Barrier synchronized (single GPU)"]
    set valid3 [string equal $result3 "Barrier synchronized (single GPU)"]
    
    expr {$valid1 && $valid2 && $valid3}
} {1}

# Test with multi-GPU initialization
test distributed_barrier-3.1 {Barrier with multi-GPU initialization} {
    torch::distributed_init 0 4 "nccl"
    set result [torch::distributed_barrier]
    string equal $result "Barrier synchronized (simulated multi-GPU)"
} {1}

test distributed_barrier-3.2 {Multi-GPU camelCase alias} {
    set result [torch::distributedBarrier]
    string equal $result "Barrier synchronized (simulated multi-GPU)"
} {1}

# Test error handling
test distributed_barrier-4.1 {Too many arguments} {
    catch {torch::distributed_barrier extra_arg} msg
    expr {[string match "*Wrong number of arguments*" $msg]}
} {1}

test distributed_barrier-4.2 {CamelCase alias - too many arguments} {
    catch {torch::distributedBarrier extra_arg} msg
    expr {[string match "*Wrong number of arguments*" $msg]}
} {1}

test distributed_barrier-4.3 {Multiple extra arguments} {
    catch {torch::distributed_barrier arg1 arg2} msg
    expr {[string match "*Wrong number of arguments*" $msg]}
} {1}

# Test consistency across modes
test distributed_barrier-5.1 {Consistency between init modes} {
    # Test single GPU first
    torch::distributed_init 0 1 "gloo"
    set single_result [torch::distributed_barrier]
    
    # Test multi-GPU
    torch::distributed_init 0 4 "nccl"
    set multi_result [torch::distributed_barrier]
    
    # Both should contain "Barrier synchronized"
    set valid_single [string match "*Barrier synchronized*" $single_result]
    set valid_multi [string match "*Barrier synchronized*" $multi_result]
    
    expr {$valid_single && $valid_multi}
} {1}

cleanupTests 