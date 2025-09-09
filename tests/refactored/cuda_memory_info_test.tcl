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

# Helper function to check if CUDA is available
proc is_cuda_available {} {
    return [catch {echo "torch::cuda_is_available" | tclsh} result] == 0 && $result == "1"
}

# Test cases for positional syntax (backward compatibility)
test cuda_memory_info-1.1 {Basic positional syntax - default device} {
    # This test will work whether CUDA is available or not
    set result [catch {torch::cuda_memory_info} output]
    if {[is_cuda_available]} {
        # If CUDA is available, should return memory info
        expr {$result == 0 && [string length $output] > 0 && [string match "*Memory:*" $output]}
    } else {
        # If CUDA is not available, should return error
        expr {$result == 1 && [string match "*CUDA not available*" $output]}
    }
} {1}

test cuda_memory_info-1.2 {Positional syntax with device_id 0} {
    set result [catch {torch::cuda_memory_info 0} output]
    if {[is_cuda_available]} {
        # Should return device 0 memory info
        expr {$result == 0 && [string match "*Device 0 Memory:*" $output]}
    } else {
        # Should return CUDA not available error
        expr {$result == 1 && [string match "*CUDA not available*" $output]}
    }
} {1}

test cuda_memory_info-1.3 {Positional syntax error - too many arguments} {
    set result [catch {torch::cuda_memory_info 0 1} output]
    expr {$result == 1}
} {1}

test cuda_memory_info-1.4 {Positional syntax error - invalid device_id type} {
    set result [catch {torch::cuda_memory_info "invalid"} output]
    expr {$result == 1 && [string match "*Invalid device_id value*" $output]}
} {1}

test cuda_memory_info-1.5 {Positional syntax error - negative device_id} {
    set result [catch {torch::cuda_memory_info -1} output]
    # Should return invalid device_id error (validation happens before CUDA check)
    expr {$result == 1 && [string match "*Invalid device_id: must be non-negative*" $output]}
} {1}

# Test cases for named parameter syntax
test cuda_memory_info-2.1 {Named parameter syntax - default device} {
    set result [catch {torch::cuda_memory_info} output]
    if {[is_cuda_available]} {
        # Should return device 0 memory info by default
        expr {$result == 0 && [string match "*Device 0 Memory:*" $output]}
    } else {
        # Should return CUDA not available error
        expr {$result == 1 && [string match "*CUDA not available*" $output]}
    }
} {1}

test cuda_memory_info-2.2 {Named parameter syntax with -device_id} {
    set result [catch {torch::cuda_memory_info -device_id 0} output]
    if {[is_cuda_available]} {
        # Should return device 0 memory info
        expr {$result == 0 && [string match "*Device 0 Memory:*" $output]}
    } else {
        # Should return CUDA not available error
        expr {$result == 1 && [string match "*CUDA not available*" $output]}
    }
} {1}

test cuda_memory_info-2.3 {Named parameter syntax error - unknown parameter} {
    set result [catch {torch::cuda_memory_info -unknown_param 1} output]
    expr {$result == 1 && [string match "*Unknown parameter*" $output]}
} {1}

test cuda_memory_info-2.4 {Named parameter syntax error - missing value} {
    set result [catch {torch::cuda_memory_info -device_id} output]
    expr {$result == 1 && [string match "*Missing value for parameter*" $output]}
} {1}

test cuda_memory_info-2.5 {Named parameter syntax error - invalid device_id value} {
    set result [catch {torch::cuda_memory_info -device_id "invalid"} output]
    expr {$result == 1 && [string match "*Invalid device_id value*" $output]}
} {1}

test cuda_memory_info-2.6 {Named parameter syntax error - negative device_id} {
    set result [catch {torch::cuda_memory_info -device_id -1} output]
    # Should return invalid device_id error (validation happens before CUDA check)
    expr {$result == 1 && [string match "*Invalid device_id: must be non-negative*" $output]}
} {1}

# Test cases for camelCase alias
test cuda_memory_info-3.1 {CamelCase alias - basic functionality} {
    set result [catch {torch::cudaMemoryInfo} output]
    if {[is_cuda_available]} {
        # Should return device 0 memory info by default
        expr {$result == 0 && [string match "*Device 0 Memory:*" $output]}
    } else {
        # Should return CUDA not available error
        expr {$result == 1 && [string match "*CUDA not available*" $output]}
    }
} {1}

test cuda_memory_info-3.2 {CamelCase alias with positional syntax} {
    set result [catch {torch::cudaMemoryInfo 0} output]
    if {[is_cuda_available]} {
        # Should return device 0 memory info
        expr {$result == 0 && [string match "*Device 0 Memory:*" $output]}
    } else {
        # Should return CUDA not available error
        expr {$result == 1 && [string match "*CUDA not available*" $output]}
    }
} {1}

test cuda_memory_info-3.3 {CamelCase alias with named parameters} {
    set result [catch {torch::cudaMemoryInfo -device_id 0} output]
    if {[is_cuda_available]} {
        # Should return device 0 memory info
        expr {$result == 0 && [string match "*Device 0 Memory:*" $output]}
    } else {
        # Should return CUDA not available error
        expr {$result == 1 && [string match "*CUDA not available*" $output]}
    }
} {1}

# Test cases for CUDA-specific functionality (when CUDA is available)
test cuda_memory_info-4.1 {Check memory info format when CUDA available} {
    if {[is_cuda_available]} {
        set result [catch {torch::cuda_memory_info} output]
        # Should contain Used, Free, and Total memory information
        expr {$result == 0 && [string match "*Device 0 Memory:*" $output] && 
              [string match "*Used=*MB*" $output] &&
              [string match "*Free=*MB*" $output] &&
              [string match "*Total=*MB*" $output]}
    } else {
        # Skip test if CUDA not available
        return 1
    }
} {1}

test cuda_memory_info-4.2 {Memory info consistency between syntaxes} {
    if {[is_cuda_available]} {
        set result1 [catch {torch::cuda_memory_info 0} output1]
        set result2 [catch {torch::cuda_memory_info -device_id 0} output2]
        set result3 [catch {torch::cudaMemoryInfo 0} output3]
        set result4 [catch {torch::cudaMemoryInfo -device_id 0} output4]
        
        # All should succeed and return the same output
        expr {$result1 == 0 && $result2 == 0 && $result3 == 0 && $result4 == 0 && 
              $output1 eq $output2 && $output2 eq $output3 && $output3 eq $output4}
    } else {
        # Skip test if CUDA not available
        return 1
    }
} {1}

test cuda_memory_info-4.3 {Memory values are reasonable} {
    if {[is_cuda_available]} {
        set result [catch {torch::cuda_memory_info} output]
        if {$result == 0} {
            # Extract memory values and check they are reasonable
            regexp {Used=(\d+)MB.*Free=(\d+)MB.*Total=(\d+)MB} $output match used free total
            # Check that Used + Free = Total (approximately, allowing for small discrepancies)
            set calculated_total [expr {$used + $free}]
            set diff [expr {abs($total - $calculated_total)}]
            expr {$diff <= 1}
        } else {
            return 0
        }
    } else {
        # Skip test if CUDA not available
        return 1
    }
} {1}

# Edge cases
test cuda_memory_info-5.1 {Large device_id when CUDA available} {
    if {[is_cuda_available]} {
        set result [catch {torch::cuda_memory_info 999} output]
        # Should return invalid device error
        expr {$result == 1 && [string match "*Invalid device ID*" $output]}
    } else {
        # Skip test if CUDA not available
        return 1
    }
} {1}

test cuda_memory_info-5.2 {Zero device_id explicit test} {
    set result [catch {torch::cuda_memory_info -device_id 0} output]
    if {[is_cuda_available]} {
        # Should work and return device 0 memory info
        expr {$result == 0 && [string match "*Device 0 Memory:*" $output]}
    } else {
        # Should return CUDA not available error
        expr {$result == 1 && [string match "*CUDA not available*" $output]}
    }
} {1}

# Stress tests
test cuda_memory_info-6.1 {Multiple consecutive calls} {
    set all_passed 1
    for {set i 0} {$i < 5} {incr i} {
        set result [catch {torch::cuda_memory_info} output]
        if {[is_cuda_available]} {
            if {$result != 0 || ![string match "*Device 0 Memory:*" $output]} {
                set all_passed 0
                break
            }
        } else {
            if {$result != 1 || ![string match "*CUDA not available*" $output]} {
                set all_passed 0
                break
            }
        }
    }
    expr {$all_passed}
} {1}

test cuda_memory_info-6.2 {Mixed syntax calls} {
    if {[is_cuda_available]} {
        set result1 [catch {torch::cuda_memory_info} output1]
        set result2 [catch {torch::cuda_memory_info -device_id 0} output2]
        set result3 [catch {torch::cudaMemoryInfo 0} output3]
        
        # All should succeed and return consistent results
        expr {$result1 == 0 && $result2 == 0 && $result3 == 0 && 
              [string match "*Device 0 Memory:*" $output1] && 
              [string match "*Device 0 Memory:*" $output2] &&
              [string match "*Device 0 Memory:*" $output3]}
    } else {
        # All should fail with CUDA not available
        set result1 [catch {torch::cuda_memory_info} output1]
        set result2 [catch {torch::cuda_memory_info -device_id 0} output2]
        set result3 [catch {torch::cudaMemoryInfo 0} output3]
        
        expr {$result1 == 1 && $result2 == 1 && $result3 == 1 &&
              [string match "*CUDA not available*" $output1] &&
              [string match "*CUDA not available*" $output2] &&
              [string match "*CUDA not available*" $output3]}
    }
} {1}

# Memory tracking consistency
test cuda_memory_info-7.1 {Memory info stays consistent during calls} {
    if {[is_cuda_available]} {
        set result1 [catch {torch::cuda_memory_info} output1]
        set result2 [catch {torch::cuda_memory_info} output2]
        
        if {$result1 == 0 && $result2 == 0} {
            # Memory usage should be similar (within reason) between consecutive calls
            regexp {Used=(\d+)MB} $output1 match1 used1
            regexp {Used=(\d+)MB} $output2 match2 used2
            
            # Allow some variation in memory usage but not huge differences
            set diff [expr {abs($used1 - $used2)}]
            expr {$diff <= 100}
        } else {
            return 0
        }
    } else {
        # Skip test if CUDA not available
        return 1
    }
} {1}

cleanupTests 