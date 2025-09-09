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

# Helper function to validate memory stats output
proc validateMemoryStats {stats} {
    # Check for CUDA availability info
    if {![regexp {cuda_available: (true|false)} $stats]} {
        return 0
    }
    
    # If CUDA is available, check for device count
    if {[regexp {cuda_available: true} $stats]} {
        if {![regexp {device_count: \d+} $stats]} {
            return 0
        }
    }
    
    return 1
}

# Test original command
test memory_stats-1.1 {Basic functionality} {
    set stats [torch::memory_stats]
    validateMemoryStats $stats
} {1}

# Test camelCase alias
test memory_stats-2.1 {CamelCase alias} {
    set stats [torch::memoryStats]
    validateMemoryStats $stats
} {1}

# Test consistency between original and camelCase alias
test memory_stats-3.1 {Consistency between original and camelCase alias} {
    set stats1 [torch::memory_stats]
    set stats2 [torch::memoryStats]
    
    expr {$stats1 eq $stats2}
} {1}

# Test error handling - too many arguments
test memory_stats-4.1 {Error: too many arguments} {
    catch {torch::memory_stats arg1 arg2} err
    string match "*wrong # args*" $err
} {1}

test memory_stats-4.2 {Error: too many arguments with camelCase alias} {
    catch {torch::memoryStats arg1 arg2} err
    string match "*wrong # args*" $err
} {1}

cleanupTests
