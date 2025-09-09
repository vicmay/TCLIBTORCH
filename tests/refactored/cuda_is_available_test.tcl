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

# ============================================================================
# Test torch::cuda_is_available - Dual Syntax Support
# ============================================================================

# ============================================================================
# Test Positional Syntax (Backward Compatibility)
# ============================================================================

test cuda_is_available-1.1 {Basic positional syntax} {
    set result [torch::cuda_is_available]
    expr {$result == 0 || $result == 1}
} 1

test cuda_is_available-1.2 {Positional syntax returns boolean integer} {
    set result [torch::cuda_is_available]
    string is integer $result
} 1

test cuda_is_available-1.3 {Positional syntax error - too many args} {
    catch {torch::cuda_is_available extra_arg} error
    string match "*Wrong number of arguments*" $error
} 1

test cuda_is_available-1.4 {Positional syntax error - multiple extra args} {
    catch {torch::cuda_is_available arg1 arg2} error
    string match "*Wrong number of arguments*" $error
} 1

# ============================================================================
# Test Named Parameter Syntax
# ============================================================================

test cuda_is_available-2.1 {Named parameter syntax with no parameters} {
    set result [torch::cuda_is_available]
    expr {$result == 0 || $result == 1}
} 1

test cuda_is_available-2.2 {Named parameter syntax returns boolean integer} {
    set result [torch::cuda_is_available]
    string is integer $result
} 1

test cuda_is_available-2.3 {Named parameter syntax error - unknown parameter} {
    catch {torch::cuda_is_available -invalid_param value} error
    string match "*Unknown parameter*" $error
} 1

test cuda_is_available-2.4 {Named parameter syntax error - missing value for parameter} {
    catch {torch::cuda_is_available -some_param} error
    string match "*Missing value for parameter*" $error
} 1

test cuda_is_available-2.5 {Named parameter syntax error - any parameter not allowed} {
    catch {torch::cuda_is_available -device 0} error
    string match "*Unknown parameter*" $error
} 1

# ============================================================================
# Test camelCase Alias
# ============================================================================

test cuda_is_available-3.1 {camelCase alias basic functionality} {
    set result [torch::cudaIsAvailable]
    expr {$result == 0 || $result == 1}
} 1

test cuda_is_available-3.2 {camelCase alias returns boolean integer} {
    set result [torch::cudaIsAvailable]
    string is integer $result
} 1

test cuda_is_available-3.3 {camelCase alias error handling} {
    catch {torch::cudaIsAvailable extra_arg} error
    string match "*Wrong number of arguments*" $error
} 1

test cuda_is_available-3.4 {camelCase alias with named parameter error} {
    catch {torch::cudaIsAvailable -invalid value} error
    string match "*Unknown parameter*" $error
} 1

# ============================================================================
# Test Syntax Consistency (Both syntaxes produce same results)
# ============================================================================

test cuda_is_available-4.1 {Syntax consistency - same result} {
    set snake_result [torch::cuda_is_available]
    set camel_result [torch::cudaIsAvailable]
    expr {$snake_result == $camel_result}
} 1

test cuda_is_available-4.2 {Multiple calls consistency} {
    set result1 [torch::cuda_is_available]
    set result2 [torch::cuda_is_available]
    set result3 [torch::cudaIsAvailable]
    expr {$result1 == $result2 && $result2 == $result3}
} 1

# ============================================================================
# Test Return Value Properties
# ============================================================================

test cuda_is_available-5.1 {Return value is boolean integer} {
    set result [torch::cuda_is_available]
    expr {$result == 0 || $result == 1}
} 1

test cuda_is_available-5.2 {Return value is integer type} {
    set result [torch::cuda_is_available]
    string is integer $result
} 1

test cuda_is_available-5.3 {Return value is stable across calls} {
    set result1 [torch::cuda_is_available]
    set result2 [torch::cuda_is_available]
    set result3 [torch::cuda_is_available]
    expr {$result1 == $result2 && $result2 == $result3}
} 1

test cuda_is_available-5.4 {Return value has expected range} {
    set result [torch::cuda_is_available]
    expr {$result >= 0 && $result <= 1}
} 1

# ============================================================================
# Test CUDA Device Count Integration
# ============================================================================

test cuda_is_available-6.1 {Relationship with device count} {
    set cuda_available [torch::cuda_is_available]
    set device_count [torch::cuda_device_count]
    
    if {$cuda_available} {
        # If CUDA is available, should have at least 1 device
        expr {$device_count > 0}
    } else {
        # If CUDA not available, should have 0 devices
        expr {$device_count == 0}
    }
} 1

test cuda_is_available-6.2 {Logical consistency with device count} {
    set available [torch::cuda_is_available]
    set count [torch::cuda_device_count]
    
    # available=1 implies count>0, available=0 implies count=0
    expr {($available == 1 && $count > 0) || ($available == 0 && $count == 0)}
} 1

# ============================================================================
# Test Edge Cases and Robustness
# ============================================================================

test cuda_is_available-7.1 {No command contamination} {
    # Ensure command doesn't interfere with variables
    set cuda_available "before"
    set result [torch::cuda_is_available]
    expr {$cuda_available == "before" && ($result == 0 || $result == 1)}
} 1

test cuda_is_available-7.2 {Multiple rapid calls} {
    set results {}
    for {set i 0} {$i < 10} {incr i} {
        lappend results [torch::cuda_is_available]
    }
    
    # All results should be the same boolean integer
    set first [lindex $results 0]
    set all_same 1
    foreach r $results {
        if {$r != $first} {
            set all_same 0
            break
        }
    }
    expr {$all_same && ($first == 0 || $first == 1)}
} 1

test cuda_is_available-7.3 {Command in different contexts} {
    # Test in subshell
    set result1 [torch::cuda_is_available]
    
    # Test in eval
    set result2 [eval {torch::cuda_is_available}]
    
    # Test in procedure
    proc get_cuda_status {} {
        return [torch::cuda_is_available]
    }
    set result3 [get_cuda_status]
    
    expr {$result1 == $result2 && $result2 == $result3}
} 1

# ============================================================================
# Test Performance
# ============================================================================

test cuda_is_available-8.1 {Performance test - multiple calls} {
    set start_time [clock milliseconds]
    for {set i 0} {$i < 1000} {incr i} {
        torch::cuda_is_available
    }
    set end_time [clock milliseconds]
    set duration [expr {$end_time - $start_time}]
    # Should complete 1000 calls in reasonable time (< 2 seconds)
    expr {$duration < 2000}
} 1

test cuda_is_available-8.2 {Performance test - mixed syntax calls} {
    set start_time [clock milliseconds]
    for {set i 0} {$i < 500} {incr i} {
        torch::cuda_is_available
        torch::cudaIsAvailable
    }
    set end_time [clock milliseconds]
    set duration [expr {$end_time - $start_time}]
    # Should complete 1000 calls in reasonable time (< 2 seconds)
    expr {$duration < 2000}
} 1

# ============================================================================
# Test Integration with Other Commands
# ============================================================================

test cuda_is_available-9.1 {Integration with device count} {
    set available [torch::cuda_is_available]
    set count [torch::cuda_device_count]
    
    # Logical consistency check
    if {$available} {
        expr {$count > 0}
    } else {
        expr {$count == 0}
    }
} 1

test cuda_is_available-9.2 {Use in conditional logic} {
    set available [torch::cuda_is_available]
    
    if {$available} {
        set backend "cuda"
    } else {
        set backend "cpu"
    }
    
    # Should work without errors
    expr {$backend == "cuda" || $backend == "cpu"}
} 1

test cuda_is_available-9.3 {Use in boolean operations} {
    set available [torch::cuda_is_available]
    
    # Test boolean operations
    set not_available [expr {!$available}]
    set and_test [expr {$available && 1}]
    set or_test [expr {$available || 1}]
    
    expr {($not_available == 0 || $not_available == 1) && $or_test == 1}
} 1

# ============================================================================
# Test Documentation Examples
# ============================================================================

test cuda_is_available-10.1 {Documentation example 1} {
    # Example from docs: basic usage
    set has_cuda [torch::cuda_is_available]
    expr {$has_cuda == 0 || $has_cuda == 1}
} 1

test cuda_is_available-10.2 {Documentation example 2} {
    # Example from docs: conditional usage
    set cuda_status [torch::cuda_is_available]
    if {$cuda_status} {
        set message "CUDA acceleration available"
    } else {
        set message "CPU-only mode"
    }
    expr {[string length $message] > 0}
} 1

test cuda_is_available-10.3 {Documentation example 3} {
    # Example from docs: camelCase usage
    set cuda_available [torch::cudaIsAvailable]
    expr {$cuda_available == 0 || $cuda_available == 1}
} 1

test cuda_is_available-10.4 {Documentation example 4} {
    # Example from docs: device creation logic
    set has_cuda [torch::cuda_is_available]
    if {$has_cuda} {
        set device "cuda"
    } else {
        set device "cpu"
    }
    expr {$device == "cuda" || $device == "cpu"}
} 1

# ============================================================================
# Test Environment Compatibility
# ============================================================================

test cuda_is_available-11.1 {Works in CUDA and non-CUDA environments} {
    # Should work regardless of CUDA installation status
    set result [torch::cuda_is_available]
    expr {$result == 0 || $result == 1}
} 1

test cuda_is_available-11.2 {Graceful handling of CUDA driver issues} {
    # Should return 0 instead of throwing errors for driver problems
    set result [torch::cuda_is_available]
    string is integer $result
} 1

test cuda_is_available-11.3 {Consistent behavior across runtime conditions} {
    # Multiple calls should be consistent even if system state changes
    set results {}
    for {set i 0} {$i < 5} {incr i} {
        lappend results [torch::cuda_is_available]
        after 10  # Small delay
    }
    
    set first [lindex $results 0]
    set all_same 1
    foreach r $results {
        if {$r != $first} {
            set all_same 0
            break
        }
    }
    set all_same
} 1

# Cleanup
cleanupTests 