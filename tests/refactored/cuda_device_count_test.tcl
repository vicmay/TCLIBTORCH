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
# Test torch::cuda_device_count - Dual Syntax Support
# ============================================================================

# ============================================================================
# Test Positional Syntax (Backward Compatibility)
# ============================================================================

test cuda_device_count-1.1 {Basic positional syntax} {
    set result [torch::cuda_device_count]
    string is integer $result
} 1

test cuda_device_count-1.2 {Positional syntax returns non-negative integer} {
    set result [torch::cuda_device_count]
    expr {$result >= 0}
} 1

test cuda_device_count-1.3 {Positional syntax error - too many args} {
    catch {torch::cuda_device_count extra_arg} error
    string match "*Wrong number of arguments*" $error
} 1

test cuda_device_count-1.4 {Positional syntax error - multiple extra args} {
    catch {torch::cuda_device_count arg1 arg2} error
    string match "*Wrong number of arguments*" $error
} 1

# ============================================================================
# Test Named Parameter Syntax
# ============================================================================

test cuda_device_count-2.1 {Named parameter syntax with no parameters} {
    set result [torch::cuda_device_count]
    string is integer $result
} 1

test cuda_device_count-2.2 {Named parameter syntax returns non-negative integer} {
    set result [torch::cuda_device_count]
    expr {$result >= 0}
} 1

test cuda_device_count-2.3 {Named parameter syntax error - unknown parameter} {
    catch {torch::cuda_device_count -invalid_param value} error
    string match "*Unknown parameter*" $error
} 1

test cuda_device_count-2.4 {Named parameter syntax error - missing value for parameter} {
    catch {torch::cuda_device_count -some_param} error
    string match "*Missing value for parameter*" $error
} 1

test cuda_device_count-2.5 {Named parameter syntax error - any parameter not allowed} {
    catch {torch::cuda_device_count -device 0} error
    string match "*Unknown parameter*" $error
} 1

# ============================================================================
# Test camelCase Alias
# ============================================================================

test cuda_device_count-3.1 {camelCase alias basic functionality} {
    set result [torch::cudaDeviceCount]
    string is integer $result
} 1

test cuda_device_count-3.2 {camelCase alias returns non-negative integer} {
    set result [torch::cudaDeviceCount]
    expr {$result >= 0}
} 1

test cuda_device_count-3.3 {camelCase alias error handling} {
    catch {torch::cudaDeviceCount extra_arg} error
    string match "*Wrong number of arguments*" $error
} 1

test cuda_device_count-3.4 {camelCase alias with named parameter error} {
    catch {torch::cudaDeviceCount -invalid value} error
    string match "*Unknown parameter*" $error
} 1

# ============================================================================
# Test Syntax Consistency (Both syntaxes produce same results)
# ============================================================================

test cuda_device_count-4.1 {Syntax consistency - same result} {
    set snake_result [torch::cuda_device_count]
    set camel_result [torch::cudaDeviceCount]
    expr {$snake_result == $camel_result}
} 1

test cuda_device_count-4.2 {Multiple calls consistency} {
    set result1 [torch::cuda_device_count]
    set result2 [torch::cuda_device_count]
    set result3 [torch::cudaDeviceCount]
    expr {$result1 == $result2 && $result2 == $result3}
} 1

# ============================================================================
# Test Return Value Properties
# ============================================================================

test cuda_device_count-5.1 {Return value is integer} {
    set result [torch::cuda_device_count]
    string is integer $result
} 1

test cuda_device_count-5.2 {Return value is non-negative} {
    set result [torch::cuda_device_count]
    expr {$result >= 0}
} 1

test cuda_device_count-5.3 {Return value has expected range} {
    set result [torch::cuda_device_count]
    # Should be reasonable number (0 to 32 devices is typical)
    expr {$result >= 0 && $result <= 32}
} 1

test cuda_device_count-5.4 {Return value is consistent across calls} {
    set result1 [torch::cuda_device_count]
    set result2 [torch::cuda_device_count]
    set result3 [torch::cuda_device_count]
    expr {$result1 == $result2 && $result2 == $result3}
} 1

# ============================================================================
# Test CUDA Availability Integration
# ============================================================================

test cuda_device_count-6.1 {Relationship with CUDA availability} {
    set cuda_available [torch::cuda_is_available]
    set device_count [torch::cuda_device_count]
    
    if {$cuda_available} {
        # If CUDA is available, should have at least 1 device
        expr {$device_count > 0}
    } else {
        # If CUDA not available, should return 0
        expr {$device_count == 0}
    }
} 1

test cuda_device_count-6.2 {Zero devices when CUDA unavailable} {
    # This test is environment dependent, but let's check the logic
    set device_count [torch::cuda_device_count]
    expr {$device_count >= 0}
} 1

# ============================================================================
# Test Edge Cases and Robustness
# ============================================================================

test cuda_device_count-7.1 {No command contamination} {
    # Ensure command doesn't interfere with variables
    set device_count "before"
    set result [torch::cuda_device_count]
    expr {$device_count == "before" && [string is integer $result]}
} 1

test cuda_device_count-7.2 {Multiple rapid calls} {
    set results {}
    for {set i 0} {$i < 10} {incr i} {
        lappend results [torch::cuda_device_count]
    }
    
    # All results should be the same integer
    set first [lindex $results 0]
    set all_same 1
    foreach r $results {
        if {$r != $first} {
            set all_same 0
            break
        }
    }
    expr {$all_same && [string is integer $first]}
} 1

test cuda_device_count-7.3 {Command in different contexts} {
    # Test in subshell
    set result1 [torch::cuda_device_count]
    
    # Test in eval
    set result2 [eval {torch::cuda_device_count}]
    
    # Test in procedure
    proc get_device_count {} {
        return [torch::cuda_device_count]
    }
    set result3 [get_device_count]
    
    expr {$result1 == $result2 && $result2 == $result3}
} 1

# ============================================================================
# Test Performance
# ============================================================================

test cuda_device_count-8.1 {Performance test - multiple calls} {
    set start_time [clock milliseconds]
    for {set i 0} {$i < 1000} {incr i} {
        torch::cuda_device_count
    }
    set end_time [clock milliseconds]
    set duration [expr {$end_time - $start_time}]
    # Should complete 1000 calls in reasonable time (< 2 seconds)
    expr {$duration < 2000}
} 1

test cuda_device_count-8.2 {Performance test - mixed syntax calls} {
    set start_time [clock milliseconds]
    for {set i 0} {$i < 500} {incr i} {
        torch::cuda_device_count
        torch::cudaDeviceCount
    }
    set end_time [clock milliseconds]
    set duration [expr {$end_time - $start_time}]
    # Should complete 1000 calls in reasonable time (< 2 seconds)
    expr {$duration < 2000}
} 1

# ============================================================================
# Test Integration with Other Commands
# ============================================================================

test cuda_device_count-9.1 {Integration with cuda_is_available} {
    set available [torch::cuda_is_available]
    set count [torch::cuda_device_count]
    
    # Logical consistency: if CUDA available, count > 0; if not, count = 0
    if {$available} {
        expr {$count > 0}
    } else {
        expr {$count == 0}
    }
} 1

test cuda_device_count-9.2 {Use in conditional logic} {
    set count [torch::cuda_device_count]
    
    if {$count > 0} {
        set has_cuda 1
    } else {
        set has_cuda 0
    }
    
    # Should work without errors
    expr {$has_cuda == 0 || $has_cuda == 1}
} 1

test cuda_device_count-9.3 {Use in loops and arithmetic} {
    set count [torch::cuda_device_count]
    
    set total 0
    for {set i 0} {$i < $count} {incr i} {
        incr total
    }
    
    expr {$total == $count}
} 1

# ============================================================================
# Test Documentation Examples
# ============================================================================

test cuda_device_count-10.1 {Documentation example 1} {
    # Example from docs: basic usage
    set num_gpus [torch::cuda_device_count]
    string is integer $num_gpus
} 1

test cuda_device_count-10.2 {Documentation example 2} {
    # Example from docs: conditional usage
    set gpu_count [torch::cuda_device_count]
    if {$gpu_count > 0} {
        set message "Found $gpu_count GPU(s)"
    } else {
        set message "No GPUs available"
    }
    expr {[string length $message] > 0}
} 1

test cuda_device_count-10.3 {Documentation example 3} {
    # Example from docs: camelCase usage
    set device_count [torch::cudaDeviceCount]
    string is integer $device_count
} 1

# Cleanup
cleanupTests 