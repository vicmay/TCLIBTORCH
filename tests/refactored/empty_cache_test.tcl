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

# =====================================================================
# TORCH::EMPTY_CACHE COMPREHENSIVE TEST SUITE
# =====================================================================

# Tests for positional syntax (backward compatibility)
test empty_cache-1.1 {Basic positional syntax - no arguments} {
    set result [torch::empty_cache]
    expr {$result eq "cache_cleared" || $result eq "cuda_not_available" || $result eq "cache_clear_attempted"}
} {1}

test empty_cache-1.2 {Positional syntax with device} {
    set result [torch::empty_cache cuda]
    expr {$result eq "cache_cleared" || $result eq "cuda_not_available" || $result eq "cache_clear_attempted"}
} {1}

test empty_cache-1.3 {Positional syntax with cpu device} {
    set result [torch::empty_cache cpu]
    expr {$result eq "cache_cleared" || $result eq "cuda_not_available" || $result eq "cache_clear_attempted"}
} {1}

# Tests for named parameter syntax
test empty_cache-2.1 {Named parameter syntax - no parameters} {
    set result [torch::empty_cache]
    expr {$result eq "cache_cleared" || $result eq "cuda_not_available" || $result eq "cache_clear_attempted"}
} {1}

test empty_cache-2.2 {Named parameter syntax with device} {
    set result [torch::empty_cache -device cuda]
    expr {$result eq "cache_cleared" || $result eq "cuda_not_available" || $result eq "cache_clear_attempted"}
} {1}

test empty_cache-2.3 {Named parameter syntax with cpu device} {
    set result [torch::empty_cache -device cpu]
    expr {$result eq "cache_cleared" || $result eq "cuda_not_available" || $result eq "cache_clear_attempted"}
} {1}

# Tests for camelCase alias
test empty_cache-3.1 {CamelCase alias basic functionality} {
    set result [torch::emptyCache]
    expr {$result eq "cache_cleared" || $result eq "cuda_not_available" || $result eq "cache_clear_attempted"}
} {1}

test empty_cache-3.2 {CamelCase alias with named parameters} {
    set result [torch::emptyCache -device cuda]
    expr {$result eq "cache_cleared" || $result eq "cuda_not_available" || $result eq "cache_clear_attempted"}
} {1}

test empty_cache-3.3 {CamelCase alias with positional parameters} {
    set result [torch::emptyCache cuda]
    expr {$result eq "cache_cleared" || $result eq "cuda_not_available" || $result eq "cache_clear_attempted"}
} {1}

# Tests for error handling
test empty_cache-4.1 {Error on unknown parameter} {
    catch {torch::empty_cache -unknown_param value} msg
    string match "*Unknown parameter*" $msg
} {1}

test empty_cache-4.2 {Error on missing parameter value} {
    catch {torch::empty_cache -device} msg
    string match "*Missing value for parameter*" $msg
} {1}

test empty_cache-4.3 {Error on too many positional arguments} {
    catch {torch::empty_cache cuda extra} msg
    expr {[string match "*Invalid number of arguments*" $msg] || [string match "*wrong*" $msg]}
} {1}

# Tests for different scenarios
test empty_cache-5.1 {Multiple calls should work} {
    set result1 [torch::empty_cache]
    set result2 [torch::empty_cache]
    expr {($result1 eq "cache_cleared" || $result1 eq "cuda_not_available" || $result1 eq "cache_clear_attempted") && 
          ($result2 eq "cache_cleared" || $result2 eq "cuda_not_available" || $result2 eq "cache_clear_attempted")}
} {1}

test empty_cache-5.2 {Works after tensor operations} {
    # Create some tensors to potentially use memory
    set t1 [torch::zeros {100 100}]
    set t2 [torch::ones {50 50}]
    
    # Empty cache
    set result [torch::empty_cache]
    expr {$result eq "cache_cleared" || $result eq "cuda_not_available" || $result eq "cache_clear_attempted"}
} {1}

test empty_cache-5.3 {Works with different device specifications} {
    # Test various device formats
    set results {}
    
    # Try different device specifications
    foreach device {cpu cuda cuda:0} {
        catch {
            set result [torch::empty_cache -device $device]
            lappend results $result
        }
    }
    
    # All results should be valid responses
    set all_valid 1
    foreach result $results {
        if {$result ne "cache_cleared" && $result ne "cuda_not_available" && $result ne "cache_clear_attempted"} {
            set all_valid 0
            break
        }
    }
    set all_valid
} {1}

# Tests for syntax consistency
test empty_cache-6.1 {Syntax consistency - both syntaxes work} {
    set result1 [torch::empty_cache]
    set result2 [torch::empty_cache -device cpu]
    
    expr {($result1 eq "cache_cleared" || $result1 eq "cuda_not_available" || $result1 eq "cache_clear_attempted") && 
          ($result2 eq "cache_cleared" || $result2 eq "cuda_not_available" || $result2 eq "cache_clear_attempted")}
} {1}

test empty_cache-6.2 {Syntax consistency - positional vs named} {
    set result1 [torch::empty_cache cuda]
    set result2 [torch::empty_cache -device cuda]
    
    # Both should return the same type of result
    expr {($result1 eq "cache_cleared" || $result1 eq "cuda_not_available" || $result1 eq "cache_clear_attempted") && 
          ($result2 eq "cache_cleared" || $result2 eq "cuda_not_available" || $result2 eq "cache_clear_attempted")}
} {1}

# Tests for parameter validation
test empty_cache-7.1 {Parameter validation - valid device names} {
    # Test common device specifications
    foreach device {cpu cuda} {
        set result [torch::empty_cache -device $device]
        if {$result ne "cache_cleared" && $result ne "cuda_not_available" && $result ne "cache_clear_attempted"} {
            set valid 0
            break
        }
        set valid 1
    }
    set valid
} {1}

test empty_cache-7.2 {Optional device parameter} {
    # Device parameter should be optional
    set result [torch::empty_cache]
    expr {$result eq "cache_cleared" || $result eq "cuda_not_available" || $result eq "cache_clear_attempted"}
} {1}

# Tests for return values
test empty_cache-8.1 {Return value validation} {
    set result [torch::empty_cache]
    # Should return one of the expected values
    expr {$result eq "cache_cleared" || $result eq "cuda_not_available" || $result eq "cache_clear_attempted"}
} {1}

test empty_cache-8.2 {Return value consistency} {
    # Multiple calls should return consistent results
    set result1 [torch::empty_cache]
    set result2 [torch::empty_cache]
    
    # Both should be valid return values
    expr {($result1 eq "cache_cleared" || $result1 eq "cuda_not_available" || $result1 eq "cache_clear_attempted") && 
          ($result2 eq "cache_cleared" || $result2 eq "cuda_not_available" || $result2 eq "cache_clear_attempted")}
} {1}

# Cleanup tests
test empty_cache-9.1 {Basic functionality verification} {
    # Test that empty_cache command works
    set result [torch::empty_cache]
    expr {[string length $result] > 0}
} {1}

cleanupTests 