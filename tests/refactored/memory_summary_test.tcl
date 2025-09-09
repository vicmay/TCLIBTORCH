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

# Helper function to validate memory summary output
proc validateMemorySummary {summary} {
    # Check for CUDA information
    if {[regexp {CUDA Memory Summary} $summary]} {
        if {![regexp {Device Count: \d+} $summary]} {
            return 0
        }
    } elseif {![regexp {CUDA not available} $summary]} {
        return 0
    }
    
    return 1
}

# Test original command
test memory_summary-1.1 {Basic functionality} {
    set summary [torch::memory_summary]
    validateMemorySummary $summary
} {1}

# Test camelCase alias
test memory_summary-2.1 {CamelCase alias} {
    set summary [torch::memorySummary]
    validateMemorySummary $summary
} {1}

# Test consistency between original and camelCase alias
test memory_summary-3.1 {Consistency between original and camelCase alias} {
    set summary1 [torch::memory_summary]
    set summary2 [torch::memorySummary]
    
    expr {$summary1 eq $summary2}
} {1}

# Test error handling - too many arguments
test memory_summary-4.1 {Error: too many arguments} {
    catch {torch::memory_summary arg1 arg2} err
    string match "*wrong # args*" $err
} {1}

test memory_summary-4.2 {Error: too many arguments with camelCase alias} {
    catch {torch::memorySummary arg1 arg2} err
    string match "*wrong # args*" $err
} {1}

cleanupTests
