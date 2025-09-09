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
# Test torch::grad_scaler_get_scale - Get scale value from gradient scaler with dual syntax support
# ============================================================================

test grad_scaler_get_scale-1.1 {Basic positional syntax} -setup {
    set scaler [torch::grad_scaler_new]
} -body {
    torch::grad_scaler_get_scale $scaler
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {65536.0}

test grad_scaler_get_scale-1.2 {Named parameter syntax} -setup {
    set scaler [torch::grad_scaler_new]
} -body {
    torch::grad_scaler_get_scale -scaler $scaler
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {65536.0}

test grad_scaler_get_scale-1.3 {Alternative named parameter syntax} -setup {
    set scaler [torch::grad_scaler_new]
} -body {
    torch::grad_scaler_get_scale -gradscaler $scaler
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {65536.0}

test grad_scaler_get_scale-1.4 {CamelCase alias} -setup {
    set scaler [torch::grad_scaler_new]
} -body {
    torch::gradScalerGetScale $scaler
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {65536.0}

test grad_scaler_get_scale-1.5 {CamelCase alias with named parameters} -setup {
    set scaler [torch::grad_scaler_new]
} -body {
    torch::gradScalerGetScale -scaler $scaler
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {65536.0}

# ============================================================================
# Test with different initial scale values
# ============================================================================

test grad_scaler_get_scale-2.1 {Custom initial scale value} -setup {
    set scaler [torch::grad_scaler_new 1024.0]
} -body {
    torch::grad_scaler_get_scale $scaler
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {1024.0}

test grad_scaler_get_scale-2.2 {Small initial scale value} -setup {
    set scaler [torch::grad_scaler_new 1.0]
} -body {
    torch::grad_scaler_get_scale -scaler $scaler
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {1.0}

test grad_scaler_get_scale-2.3 {Large initial scale value} -setup {
    set scaler [torch::grad_scaler_new 131072.0]
} -body {
    torch::grad_scaler_get_scale -scaler $scaler
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {131072.0}

test grad_scaler_get_scale-2.4 {Fractional scale value} -setup {
    set scaler [torch::grad_scaler_new 128.5]
} -body {
    torch::grad_scaler_get_scale $scaler
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {128.5}

# ============================================================================
# Error Handling Tests
# ============================================================================

test grad_scaler_get_scale-3.1 {Missing arguments - positional} -body {
    torch::grad_scaler_get_scale
} -returnCodes error -match glob -result "*Required parameters missing*"

test grad_scaler_get_scale-3.2 {Missing arguments - named} -body {
    torch::grad_scaler_get_scale -scaler
} -returnCodes error -match glob -result "*Named parameters must come in pairs*"

test grad_scaler_get_scale-3.3 {Invalid scaler handle} -body {
    torch::grad_scaler_get_scale "invalid_scaler"
} -returnCodes error -match glob -result "*Gradient scaler not found*"

test grad_scaler_get_scale-3.4 {Unknown parameter} -setup {
    set scaler [torch::grad_scaler_new]
} -body {
    torch::grad_scaler_get_scale -scaler $scaler -unknown "value"
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -returnCodes error -match glob -result "*Unknown parameter*"

test grad_scaler_get_scale-3.5 {Missing scaler parameter} -body {
    torch::grad_scaler_get_scale -gradscaler
} -returnCodes error -match glob -result "*Named parameters must come in pairs*"

# ============================================================================
# Backward Compatibility Tests
# ============================================================================

test grad_scaler_get_scale-4.1 {Backward compatibility - positional matches named} -setup {
    set scaler [torch::grad_scaler_new 2048.0]
} -body {
    set result1 [torch::grad_scaler_get_scale $scaler]
    set result2 [torch::grad_scaler_get_scale -scaler $scaler]
    expr {$result1 == $result2}
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {1}

test grad_scaler_get_scale-4.2 {CamelCase alias produces same result} -setup {
    set scaler [torch::grad_scaler_new 512.0]
} -body {
    set result1 [torch::grad_scaler_get_scale $scaler]
    set result2 [torch::gradScalerGetScale $scaler]
    expr {$result1 == $result2}
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {1}

test grad_scaler_get_scale-4.3 {Parameter aliases produce same result} -setup {
    set scaler [torch::grad_scaler_new 4096.0]
} -body {
    set result1 [torch::grad_scaler_get_scale -scaler $scaler]
    set result2 [torch::grad_scaler_get_scale -gradscaler $scaler]
    expr {$result1 == $result2}
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {1}

# ============================================================================
# Multiple Scalers Tests
# ============================================================================

test grad_scaler_get_scale-5.1 {Multiple scalers with different scales} -setup {
    set scaler1 [torch::grad_scaler_new 1000.0]
    set scaler2 [torch::grad_scaler_new 2000.0]
    set scaler3 [torch::grad_scaler_new 3000.0]
} -body {
    set scale1 [torch::grad_scaler_get_scale $scaler1]
    set scale2 [torch::grad_scaler_get_scale $scaler2]
    set scale3 [torch::grad_scaler_get_scale $scaler3]
    list $scale1 $scale2 $scale3
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {1000.0 2000.0 3000.0}

test grad_scaler_get_scale-5.2 {Multiple scalers with mixed syntax} -setup {
    set scaler1 [torch::grad_scaler_new 100.0]
    set scaler2 [torch::grad_scaler_new 200.0]
} -body {
    set scale1 [torch::grad_scaler_get_scale $scaler1]
    set scale2 [torch::grad_scaler_get_scale -scaler $scaler2]
    list $scale1 $scale2
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {100.0 200.0}

# ============================================================================
# Return Type Tests
# ============================================================================

test grad_scaler_get_scale-6.1 {Return type is numeric} -setup {
    set scaler [torch::grad_scaler_new 1.5]
} -body {
    set result [torch::grad_scaler_get_scale $scaler]
    expr {[string is double -strict $result] ? "numeric" : "not_numeric"}
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {numeric}

test grad_scaler_get_scale-6.2 {Return precision for small values} -setup {
    set scaler [torch::grad_scaler_new 0.125]
} -body {
    torch::grad_scaler_get_scale $scaler
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {0.125}

test grad_scaler_get_scale-6.3 {Return precision for large values} -setup {
    set scaler [torch::grad_scaler_new 1048576.0]
} -body {
    torch::grad_scaler_get_scale $scaler
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {1048576.0}

# ============================================================================
# Edge Cases Tests
# ============================================================================

test grad_scaler_get_scale-7.1 {Very small scale value} -setup {
    set scaler [torch::grad_scaler_new 1e-10]
} -body {
    set result [torch::grad_scaler_get_scale $scaler]
    expr {$result > 9e-11 && $result < 1.1e-10}
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {1}

test grad_scaler_get_scale-7.2 {Very large scale value} -setup {
    set scaler [torch::grad_scaler_new 1e10]
} -body {
    set result [torch::grad_scaler_get_scale $scaler]
    expr {$result > 9e9 && $result < 1.1e10}
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {1}

test grad_scaler_get_scale-7.3 {Scale value of 1.0} -setup {
    set scaler [torch::grad_scaler_new 1.0]
} -body {
    torch::grad_scaler_get_scale -scaler $scaler
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {1.0}

# ============================================================================
# Integration Tests
# ============================================================================

test grad_scaler_get_scale-8.1 {Integration with scaler update} -setup {
    set scaler [torch::grad_scaler_new 1024.0]
} -body {
    # Get initial scale
    set initial_scale [torch::grad_scaler_get_scale $scaler]
    
    # Update scaler (may change the scale)
    torch::grad_scaler_update $scaler
    
    # Get scale after update
    set final_scale [torch::grad_scaler_get_scale $scaler]
    
    # Both should be numeric values
    expr {[string is double -strict $initial_scale] && [string is double -strict $final_scale]}
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {1}

test grad_scaler_get_scale-8.2 {Scale value consistency} -setup {
    set scaler [torch::grad_scaler_new 8192.0]
} -body {
    # Get scale multiple times - should be consistent
    set scale1 [torch::grad_scaler_get_scale $scaler]
    set scale2 [torch::grad_scaler_get_scale $scaler]
    set scale3 [torch::grad_scaler_get_scale $scaler]
    expr {$scale1 == $scale2 && $scale2 == $scale3}
} -cleanup {
    # No cleanup needed - scalers are managed automatically
} -result {1}

cleanupTests 