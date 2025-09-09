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

# Helper function to validate memory snapshot output
proc validateMemorySnapshot {snapshot} {
    # Check for timestamp
    if {![regexp {timestamp: \d+} $snapshot]} {
        return 0
    }
    
    # Check for CUDA availability info
    if {![regexp {cuda_available: (true|false)} $snapshot]} {
        return 0
    }
    
    # If CUDA is available, check for device count
    if {[regexp {cuda_available: true} $snapshot]} {
        if {![regexp {device_count: \d+} $snapshot]} {
            return 0
        }
    }
    
    return 1
}

# Test original command
test memory_snapshot-1.1 {Basic functionality} {
    set snapshot [torch::memory_snapshot]
    validateMemorySnapshot $snapshot
} {1}

# Test camelCase alias
test memory_snapshot-2.1 {CamelCase alias} {
    set snapshot [torch::memorySnapshot]
    validateMemorySnapshot $snapshot
} {1}

# Test consistency between original and camelCase alias
test memory_snapshot-3.1 {Consistency between original and camelCase alias} {
    # Note: We can't directly compare the outputs as timestamps will differ
    # Instead, we verify that both return valid snapshots
    set snapshot1 [torch::memory_snapshot]
    set snapshot2 [torch::memorySnapshot]
    
    expr {[validateMemorySnapshot $snapshot1] && [validateMemorySnapshot $snapshot2]}
} {1}

# Test error handling - no arguments expected
test memory_snapshot-4.1 {Error: invalid parameter} {
    catch {torch::memory_snapshot -invalid_param value} err
    string match "*Unknown parameter*" $err
} {1}

test memory_snapshot-4.2 {Error: invalid parameter with camelCase alias} {
    catch {torch::memorySnapshot -invalid_param value} err
    string match "*Unknown parameter*" $err
} {1}

cleanupTests
