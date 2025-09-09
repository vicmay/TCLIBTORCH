#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Configure test output
configure -verbose {pass fail skip error}

# Load the extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# -----------------------------------------------------------------------------
# 1. Positional syntax tests
# -----------------------------------------------------------------------------

# Basic positional (default parameters)
test dropout-1.1 {Basic positional syntax - default parameters} {
    set module [torch::dropout]
    string match "dropout*" $module
} {1}

# Positional with p parameter
test dropout-1.2 {Positional syntax with p parameter} {
    set module [torch::dropout 0.3]
    string match "dropout*" $module
} {1}

# Positional with p and training
test dropout-1.3 {Positional syntax with p and training} {
    set module [torch::dropout 0.2 true]
    string match "dropout*" $module
} {1}

# Positional with all parameters
test dropout-1.4 {Positional syntax with all parameters} {
    set module [torch::dropout 0.1 false true]
    string match "dropout*" $module
} {1}

# -----------------------------------------------------------------------------
# 2. Named parameter syntax tests
# -----------------------------------------------------------------------------

# Basic named (default parameters)
test dropout-2.1 {Named syntax default parameters} {
    set module [torch::dropout -p 0.5]
    string match "dropout*" $module
} {1}

# Named with p parameter
test dropout-2.2 {Named syntax with p parameter} {
    set module [torch::dropout -p 0.3]
    string match "dropout*" $module
} {1}

# Named with p and training
test dropout-2.3 {Named syntax with p and training} {
    set module [torch::dropout -p 0.2 -training true]
    string match "dropout*" $module
} {1}

# Named with all parameters
test dropout-2.4 {Named syntax with all parameters} {
    set module [torch::dropout -p 0.1 -training false -inplace true]
    string match "dropout*" $module
} {1}

# Named with mixed parameter order
test dropout-2.5 {Named syntax mixed parameter order} {
    set module [torch::dropout -training true -p 0.4 -inplace false]
    string match "dropout*" $module
} {1}

# -----------------------------------------------------------------------------
# 3. Parameter validation tests
# -----------------------------------------------------------------------------

# Valid p values
test dropout-3.1 {Valid p value 0.0} {
    set module [torch::dropout -p 0.0]
    string match "dropout*" $module
} {1}

test dropout-3.2 {Valid p value 1.0} {
    set module [torch::dropout -p 1.0]
    string match "dropout*" $module
} {1}

test dropout-3.3 {Valid p value 0.5} {
    set module [torch::dropout -p 0.5]
    string match "dropout*" $module
} {1}

# -----------------------------------------------------------------------------
# 4. Error handling tests
# -----------------------------------------------------------------------------

# Invalid p value (negative)
test dropout-4.1 {Error: negative p value} {
    set code [catch {torch::dropout -p -0.1} msg]
    list $code [string match "*0.0 and 1.0*" $msg]
} {1 1}

# Invalid p value (greater than 1)
test dropout-4.2 {Error: p value greater than 1} {
    set code [catch {torch::dropout -p 1.5} msg]
    list $code [string match "*0.0 and 1.0*" $msg]
} {1 1}

# Invalid parameter name
test dropout-4.3 {Error: invalid parameter name} {
    set code [catch {torch::dropout -invalid_param 0.5} msg]
    list $code [string match "*Unknown parameter*" $msg]
} {1 1}

# Missing parameter value
test dropout-4.4 {Error: missing parameter value} {
    set code [catch {torch::dropout -p} msg]
    list $code [string match "*pairs*" $msg]
} {1 1}

# Invalid p value type
test dropout-4.5 {Error: invalid p value type} {
    set code [catch {torch::dropout -p not_a_number} msg]
    list $code [string match "*Invalid p parameter*" $msg]
} {1 1}

# -----------------------------------------------------------------------------
# 5. Boolean parameter validation
# -----------------------------------------------------------------------------

# Training parameter variations
test dropout-5.1 {Training parameter true} {
    set module [torch::dropout -p 0.5 -training true]
    string match "dropout*" $module
} {1}

test dropout-5.2 {Training parameter 1} {
    set module [torch::dropout -p 0.5 -training 1]
    string match "dropout*" $module
} {1}

test dropout-5.3 {Training parameter false} {
    set module [torch::dropout -p 0.5 -training false]
    string match "dropout*" $module
} {1}

test dropout-5.4 {Training parameter 0} {
    set module [torch::dropout -p 0.5 -training 0]
    string match "dropout*" $module
} {1}

# Inplace parameter variations
test dropout-5.5 {Inplace parameter true} {
    set module [torch::dropout -p 0.5 -inplace true]
    string match "dropout*" $module
} {1}

test dropout-5.6 {Inplace parameter 1} {
    set module [torch::dropout -p 0.5 -inplace 1]
    string match "dropout*" $module
} {1}

test dropout-5.7 {Inplace parameter false} {
    set module [torch::dropout -p 0.5 -inplace false]
    string match "dropout*" $module
} {1}

test dropout-5.8 {Inplace parameter 0} {
    set module [torch::dropout -p 0.5 -inplace 0]
    string match "dropout*" $module
} {1}

# -----------------------------------------------------------------------------
# 6. Syntax consistency tests
# -----------------------------------------------------------------------------

# Both syntaxes should work (we can't easily compare modules but can verify creation)
test dropout-6.1 {Syntax consistency - both create modules} {
    set module1 [torch::dropout 0.3 true false]
    set module2 [torch::dropout -p 0.3 -training true -inplace false]
    
    set match1 [string match "dropout*" $module1]
    set match2 [string match "dropout*" $module2]
    
    list $match1 $match2
} {1 1}

# Default behavior consistency
test dropout-6.2 {Default parameter consistency} {
    set module1 [torch::dropout]
    set module2 [torch::dropout -p 0.5]
    
    set match1 [string match "dropout*" $module1]
    set match2 [string match "dropout*" $module2]
    
    list $match1 $match2
} {1 1}

# -----------------------------------------------------------------------------
# 7. Edge cases
# -----------------------------------------------------------------------------

# Multiple modules creation
test dropout-7.1 {Multiple module creation} {
    set m1 [torch::dropout -p 0.1]
    set m2 [torch::dropout -p 0.2]
    set m3 [torch::dropout -p 0.3]
    
    set all_match [expr {[string match "dropout*" $m1] && [string match "dropout*" $m2] && [string match "dropout*" $m3]}]
    set all_different [expr {$m1 != $m2 && $m2 != $m3 && $m1 != $m3}]
    
    list $all_match $all_different
} {1 1}

# Boundary values
test dropout-7.2 {Boundary p values} {
    set m1 [torch::dropout -p 0.0]
    set m2 [torch::dropout -p 1.0]
    
    set match1 [string match "dropout*" $m1]
    set match2 [string match "dropout*" $m2]
    
    list $match1 $match2
} {1 1}

# -----------------------------------------------------------------------------
cleanupTests 