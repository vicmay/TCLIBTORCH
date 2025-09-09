#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# Test cases for positional syntax (backward compatibility)
test emptyCache-1.1 {Basic positional syntax - no arguments} {
    set result [torch::empty_cache]
    return $result
} {cache_cleared}

test emptyCache-1.2 {Positional syntax with device} {
    set result [torch::empty_cache cpu]
    return $result
} {cache_cleared}

test emptyCache-1.3 {Positional syntax with CUDA device} {
    set result [torch::empty_cache cuda:0]
    return $result
} {cache_cleared}

;# Test cases for named parameter syntax
test emptyCache-2.1 {Named parameter syntax with -device} {
    set result [torch::empty_cache -device cpu]
    return $result
} {cache_cleared}

test emptyCache-2.2 {Named parameter syntax with CUDA device} {
    set result [torch::empty_cache -device cuda:0]
    return $result
} {cache_cleared}

;# Test cases for camelCase alias
test emptyCache-3.1 {CamelCase alias torch::emptyCache} {
    set result [torch::emptyCache]
    return $result
} {cache_cleared}

test emptyCache-3.2 {CamelCase alias with device} {
    set result [torch::emptyCache -device cpu]
    return $result
} {cache_cleared}

;# Test error handling
test emptyCache-4.1 {Error: Too many positional arguments} {
    catch {torch::empty_cache cpu cuda} result
    expr {[string first "Invalid number of arguments" $result] >= 0}
} {1}

test emptyCache-4.2 {Error: Named parameter without value} {
    catch {torch::empty_cache -device} result
    expr {[string first "Missing value" $result] >= 0}
} {1}

test emptyCache-4.3 {Error: Unknown named parameter} {
    catch {torch::empty_cache -unknown cpu} result
    expr {[string first "Unknown parameter" $result] >= 0}
} {1}

;# Test different devices
test emptyCache-5.1 {CPU device} {
    set result [torch::empty_cache -device cpu]
    return $result
} {cache_cleared}

test emptyCache-5.2 {CUDA device 0} {
    set result [torch::empty_cache -device cuda:0]
    return $result
} {cache_cleared}

test emptyCache-5.3 {CUDA device 1} {
    set result [torch::empty_cache -device cuda:1]
    return $result
} {cache_cleared}

;# Test edge cases
test emptyCache-6.1 {Edge case: empty device string} {
    set result [torch::empty_cache -device ""]
    return $result
} {cache_cleared}

;# Test syntax consistency
test emptyCache-7.1 {Syntax consistency: positional vs named} {
    set result1 [torch::empty_cache cpu]
    set result2 [torch::empty_cache -device cpu]
    return [expr {$result1 == $result2}]
} {1}

test emptyCache-7.2 {Syntax consistency: snake_case vs camelCase} {
    set result1 [torch::empty_cache]
    set result2 [torch::emptyCache]
    return [expr {$result1 == $result2}]
} {1}

test emptyCache-7.3 {Syntax consistency: with device} {
    set result1 [torch::empty_cache cuda:0]
    set result2 [torch::emptyCache -device cuda:0]
    return [expr {$result1 == $result2}]
} {1}

;# Test CUDA availability handling
test emptyCache-8.1 {CUDA availability check} {
    set result [torch::empty_cache]
    ;# Should return either "cache_cleared" or "cuda_not_available"
    expr {[string first "cache_cleared" $result] >= 0 || [string first "cuda_not_available" $result] >= 0}
} {1}

cleanupTests 