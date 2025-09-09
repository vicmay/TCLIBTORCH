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
# Test torch::grad_scaler_new - Positional Syntax (Backward Compatibility)
# ============================================================================

test grad_scaler_new-1.1 {Basic positional syntax - default parameters} {
    set scaler [torch::grad_scaler_new]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-1.2 {Positional syntax - custom init_scale} {
    set scaler [torch::grad_scaler_new 1024.0]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-1.3 {Positional syntax - init_scale and growth_factor} {
    set scaler [torch::grad_scaler_new 2048.0 4.0]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-1.4 {Positional syntax - three parameters} {
    set scaler [torch::grad_scaler_new 4096.0 2.5 0.25]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-1.5 {Positional syntax - all parameters} {
    set scaler [torch::grad_scaler_new 8192.0 3.0 0.125 1000]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

# ============================================================================
# Test torch::grad_scaler_new - Named Parameter Syntax
# ============================================================================

test grad_scaler_new-2.1 {Named parameter syntax - initScale only} {
    set scaler [torch::grad_scaler_new -initScale 1024.0]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-2.2 {Named parameter syntax - growthFactor only} {
    set scaler [torch::grad_scaler_new -growthFactor 4.0]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-2.3 {Named parameter syntax - backoffFactor only} {
    set scaler [torch::grad_scaler_new -backoffFactor 0.25]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-2.4 {Named parameter syntax - growthInterval only} {
    set scaler [torch::grad_scaler_new -growthInterval 1000]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-2.5 {Named parameter syntax - multiple parameters} {
    set scaler [torch::grad_scaler_new -initScale 2048.0 -growthFactor 3.0 -backoffFactor 0.125]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-2.6 {Named parameter syntax - all parameters} {
    set scaler [torch::grad_scaler_new -initScale 4096.0 -growthFactor 2.5 -backoffFactor 0.25 -growthInterval 1500]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-2.7 {Named parameter syntax - parameters in different order} {
    set scaler [torch::grad_scaler_new -growthInterval 800 -initScale 512.0 -backoffFactor 0.5 -growthFactor 2.0]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

# ============================================================================
# Test torch::grad_scaler_new - Snake Case Aliases
# ============================================================================

test grad_scaler_new-3.1 {Snake case aliases - init_scale} {
    set scaler [torch::grad_scaler_new -init_scale 1024.0]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-3.2 {Snake case aliases - growth_factor} {
    set scaler [torch::grad_scaler_new -growth_factor 4.0]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-3.3 {Snake case aliases - backoff_factor} {
    set scaler [torch::grad_scaler_new -backoff_factor 0.25]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-3.4 {Snake case aliases - growth_interval} {
    set scaler [torch::grad_scaler_new -growth_interval 1000]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-3.5 {Snake case aliases - all parameters} {
    set scaler [torch::grad_scaler_new -init_scale 2048.0 -growth_factor 3.0 -backoff_factor 0.125 -growth_interval 1200]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

# ============================================================================
# Test torch::gradScalerNew - CamelCase Alias
# ============================================================================

test grad_scaler_new-4.1 {CamelCase alias - positional syntax} {
    set scaler [torch::gradScalerNew]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-4.2 {CamelCase alias - named parameter syntax} {
    set scaler [torch::gradScalerNew -initScale 1024.0 -growthFactor 2.0]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-4.3 {CamelCase alias - all parameters} {
    set scaler [torch::gradScalerNew -initScale 4096.0 -growthFactor 2.5 -backoffFactor 0.25 -growthInterval 1500]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

# ============================================================================
# Test Error Handling
# ============================================================================

test grad_scaler_new-5.1 {Error handling - negative init_scale} {
    catch {torch::grad_scaler_new -initScale -1.0} result
    expr {[string match "*positive*" $result] || [string match "*Invalid*" $result]}
} {1}

test grad_scaler_new-5.2 {Error handling - negative growth_factor} {
    catch {torch::grad_scaler_new -growthFactor -2.0} result
    expr {[string match "*positive*" $result] || [string match "*Invalid*" $result]}
} {1}

test grad_scaler_new-5.3 {Error handling - negative backoff_factor} {
    catch {torch::grad_scaler_new -backoffFactor -0.5} result
    expr {[string match "*positive*" $result] || [string match "*Invalid*" $result]}
} {1}

test grad_scaler_new-5.4 {Error handling - negative growth_interval} {
    catch {torch::grad_scaler_new -growthInterval -1000} result
    expr {[string match "*positive*" $result] || [string match "*Invalid*" $result]}
} {1}

test grad_scaler_new-5.5 {Error handling - unknown parameter} {
    catch {torch::grad_scaler_new -invalidParam 1.0} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test grad_scaler_new-5.6 {Error handling - odd number of parameters} {
    catch {torch::grad_scaler_new -initScale} result
    expr {[string match "*pairs*" $result]}
} {1}

test grad_scaler_new-5.7 {Error handling - too many positional parameters} {
    catch {torch::grad_scaler_new 1.0 2.0 3.0 4 5} result
    expr {[string match "*Usage*" $result]}
} {1}

# ============================================================================
# Test Functional Verification
# ============================================================================

test grad_scaler_new-6.1 {Functional verification - can use scaler with grad_scaler_get_scale} {
    set scaler [torch::grad_scaler_new -initScale 1024.0]
    set scale [torch::grad_scaler_get_scale $scaler]
    expr {$scale == 1024.0}
} {1}

test grad_scaler_new-6.2 {Functional verification - different scalers have different handles} {
    set scaler1 [torch::grad_scaler_new -initScale 1024.0]
    set scaler2 [torch::grad_scaler_new -initScale 2048.0]
    expr {$scaler1 ne $scaler2}
} {1}

test grad_scaler_new-6.3 {Functional verification - verify custom parameters work} {
    set scaler [torch::grad_scaler_new -initScale 512.0 -growthFactor 4.0 -backoffFactor 0.125 -growthInterval 500]
    set scale [torch::grad_scaler_get_scale $scaler]
    expr {$scale == 512.0}
} {1}

# ============================================================================
# Test Mixed Parameter Types
# ============================================================================

test grad_scaler_new-7.1 {Mixed parameter types - integer as double} {
    set scaler [torch::grad_scaler_new -initScale 1024 -growthFactor 2]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-7.2 {Mixed parameter types - double as double} {
    set scaler [torch::grad_scaler_new -initScale 1024.5 -growthFactor 2.5]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-7.3 {Mixed parameter types - scientific notation} {
    set scaler [torch::grad_scaler_new -initScale 1e3 -growthFactor 2e0]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

# ============================================================================
# Test Edge Cases
# ============================================================================

test grad_scaler_new-8.1 {Edge case - very small init_scale} {
    set scaler [torch::grad_scaler_new -initScale 1e-10]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-8.2 {Edge case - very large init_scale} {
    set scaler [torch::grad_scaler_new -initScale 1e10]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-8.3 {Edge case - growth_interval of 1} {
    set scaler [torch::grad_scaler_new -growthInterval 1]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

test grad_scaler_new-8.4 {Edge case - very small backoff_factor} {
    set scaler [torch::grad_scaler_new -backoffFactor 0.001]
    set result [string match "scaler*" $scaler]
    unset scaler
    set result
} {1}

cleanupTests 