#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Configure test output
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# -----------------------------------------------------------------------------
# Test 1: Basic functionality
# -----------------------------------------------------------------------------

# Positional syntax
test gamma-1.1 {Basic gamma positional syntax} {
    set result [torch::gamma {2 3} 2.0 3.0]
    expr {[string match "tensor*" $result]}
} {1}

# Named parameter syntax
test gamma-1.2 {Basic gamma named parameter syntax} {
    set result [torch::gamma -size {2 3} -alpha 2.0 -beta 3.0]
    expr {[string match "tensor*" $result]}
} {1}

# -----------------------------------------------------------------------------
# Test 2: Parameter variations
# -----------------------------------------------------------------------------

# Single dimension
test gamma-2.1 {Single dimension tensor} {
    set result [torch::gamma {5} 1.5 1.0]
    expr {[string match "tensor*" $result]}
} {1}

# Multi-dimensional named
test gamma-2.2 {Multi-dimensional named syntax} {
    set result [torch::gamma -size {3 4} -alpha 0.5 -beta 2.0]
    expr {[string match "tensor*" $result]}
} {1}

# Data type support
test gamma-2.3 {Float64 dtype support} {
    set result [torch::gamma -size {3} -alpha 2.0 -beta 1.0 -dtype float64]
    expr {[string match "tensor*" $result]}
} {1}

# Device support (CPU)
test gamma-2.4 {CPU device support} {
    set result [torch::gamma {2} 2.0 1.0 float32 cpu]
    expr {[string match "tensor*" $result]}
} {1}

# -----------------------------------------------------------------------------
# Test 3: Error handling
# -----------------------------------------------------------------------------

# Missing arguments
test gamma-3.1 {Error: Missing args} {
    catch {torch::gamma} result
    expr {[string match "*Usage*" $result] || [string match "*Required parameters*" $result]}
} {1}

# Invalid alpha (negative)
test gamma-3.2 {Error: Negative alpha} {
    catch {torch::gamma {2} -1.0 1.0} result
    expr {[string match "*alpha*" $result] || [string match "*invalid*" $result]}
} {1}

# Invalid beta (zero)
test gamma-3.3 {Error: Zero beta} {
    catch {torch::gamma -size {2} -alpha 1.0 -beta 0.0} result
    expr {[string match "*beta*" $result] || [string match "*invalid*" $result]}
} {1}

# Unknown parameter
test gamma-3.4 {Error: Unknown parameter} {
    catch {torch::gamma -size {2} -alpha 1.0 -beta 1.0 -foo bar} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

# Named parameter without value
test gamma-3.5 {Error: Named parameter without value} {
    catch {torch::gamma -size} result
    expr {[string match "*Usage*" $result] || [string match "*Named parameter requires a value*" $result]}
} {1}

# -----------------------------------------------------------------------------
# Test 4: Syntax consistency
# -----------------------------------------------------------------------------

# Ensure both syntaxes return tensors
test gamma-4.1 {Both syntaxes produce valid tensors} {
    set res_pos [torch::gamma {2 2} 2.0 1.0]
    set res_named [torch::gamma -size {2 2} -alpha 2.0 -beta 1.0]
    expr {[string match "tensor*" $res_pos] && [string match "tensor*" $res_named]}
} {1}

cleanupTests 