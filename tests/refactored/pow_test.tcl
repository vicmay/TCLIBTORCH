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

# Helper to create test tensors
proc create_test_tensor {args} {
    # Flexible helper: if called with two arguments, treat first as variable
    # name and second as the shape list. If called with a single argument,
    # simply return a zeros tensor with the given shape.
    if {[llength $args] == 1} {
        # Usage: create_test_tensor {2 2}
        set tensorShape [lindex $args 0]
        return [torch::zeros $tensorShape]
    } elseif {[llength $args] == 2} {
        # Usage: create_test_tensor myVar {2 2}
        set varName [lindex $args 0]
        set tensorShape [lindex $args 1]
        upvar 1 $varName var
        set var [torch::zeros $tensorShape]
        return $var
    } else {
        error {wrong # args: should be "create_test_tensor ?varName? shape"}
    }
}

# ============================================================================
# Basic Pow Tests - Positional Syntax
# ============================================================================

test pow-1.1 {Basic pow with positional syntax} {
    set base [torch::ones {3 4}]
    set exponent [torch::ones {3 4}]
    set result [torch::pow $base $exponent]
    string match "tensor*" $result
} {1}

test pow-1.2 {Pow with different tensor sizes - positional syntax} {
    set base [torch::ones {2 3}]
    set exponent [torch::ones {2 3}]
    set result [torch::pow $base $exponent]
    string match "tensor*" $result
} {1}

test pow-1.3 {Pow with larger tensor - positional syntax} {
    set base [torch::ones {4 5}]
    set exponent [torch::ones {4 5}]
    set result [torch::pow $base $exponent]
    string match "tensor*" $result
} {1}

test pow-1.4 {Pow with 1D tensor - positional syntax} {
    set base [torch::ones {10}]
    set exponent [torch::ones {10}]
    set result [torch::pow $base $exponent]
    string match "tensor*" $result
} {1}

test pow-1.5 {Pow with 3D tensor - positional syntax} {
    set base [torch::ones {2 3 4}]
    set exponent [torch::ones {2 3 4}]
    set result [torch::pow $base $exponent]
    string match "tensor*" $result
} {1}

# ============================================================================
# Pow Tests - Named Parameter Syntax
# ============================================================================

test pow-2.1 {Basic pow with named parameter syntax - base/exponent parameters} {
    set base [torch::ones {3 4}]
    set exponent [torch::ones {3 4}]
    set result [torch::pow -base $base -exponent $exponent]
    string match "tensor*" $result
} {1}

test pow-2.2 {Pow with named parameter syntax - input1/input2 parameters} {
    set base [torch::ones {2 3}]
    set exponent [torch::ones {2 3}]
    set result [torch::pow -input1 $base -input2 $exponent]
    string match "tensor*" $result
} {1}

test pow-2.3 {Pow with different tensor sizes - named syntax} {
    set base [torch::ones {4 5}]
    set exponent [torch::ones {4 5}]
    set result [torch::pow -base $base -exponent $exponent]
    string match "tensor*" $result
} {1}

test pow-2.4 {Pow with 1D tensor - named syntax} {
    set base [torch::ones {10}]
    set exponent [torch::ones {10}]
    set result [torch::pow -base $base -exponent $exponent]
    string match "tensor*" $result
} {1}

test pow-2.5 {Pow with 3D tensor - named syntax} {
    set base [torch::ones {2 3 4}]
    set exponent [torch::ones {2 3 4}]
    set result [torch::pow -exponent $exponent -base $base]
    string match "tensor*" $result
} {1}

test pow-2.6 {Pow with mixed parameter names} {
    set base [torch::ones {2 2}]
    set exponent [torch::ones {2 2}]
    set result [torch::pow -input1 $base -exponent $exponent]
    string match "tensor*" $result
} {1}

# ============================================================================
# CamelCase Alias Tests (pow is already camelCase compatible)
# ============================================================================

test pow-3.1 {Pow camelCase compatibility with positional syntax} {
    set base [torch::ones {3 4}]
    set exponent [torch::ones {3 4}]
    set result [torch::pow $base $exponent]
    string match "tensor*" $result
} {1}

test pow-3.2 {Pow camelCase compatibility with named parameters} {
    set base [torch::ones {2 3}]
    set exponent [torch::ones {2 3}]
    set result [torch::pow -base $base -exponent $exponent]
    string match "tensor*" $result
} {1}

# ============================================================================
# Parameter Validation Tests
# ============================================================================

test pow-4.1 {Error: Missing required parameters - positional} {
    catch {torch::pow} error
    string match "*Usage*" $error
} {1}

test pow-4.2 {Error: Missing second parameter - positional} {
    set base [torch::ones {2 2}]
    catch {torch::pow $base} error
    string match "*Usage*" $error
} {1}

test pow-4.3 {Error: Too many positional arguments} {
    set base [torch::ones {2 2}]
    set exponent [torch::ones {2 2}]
    catch {torch::pow $base $exponent extra_arg} error
    string match "*Usage*" $error
} {1}

test pow-4.4 {Error: Missing required parameter - named} {
    catch {torch::pow -base} error
    string match "*Missing value*" $error
} {1}

test pow-4.5 {Error: Invalid base tensor name - positional} {
    set exponent [torch::ones {2 2}]
    catch {torch::pow nonexistent_tensor $exponent} error
    string match "*Invalid base tensor name*" $error
} {1}

test pow-4.6 {Error: Invalid exponent tensor name - positional} {
    set base [torch::ones {2 2}]
    catch {torch::pow $base nonexistent_tensor} error
    string match "*Invalid exponent tensor name*" $error
} {1}

test pow-4.7 {Error: Invalid base tensor name - named} {
    set exponent [torch::ones {2 2}]
    catch {torch::pow -base nonexistent_tensor -exponent $exponent} error
    string match "*Invalid base tensor name*" $error
} {1}

test pow-4.8 {Error: Invalid exponent tensor name - named} {
    set base [torch::ones {2 2}]
    catch {torch::pow -base $base -exponent nonexistent_tensor} error
    string match "*Invalid exponent tensor name*" $error
} {1}

test pow-4.9 {Error: Unknown parameter - named} {
    set base [torch::ones {2 2}]
    set exponent [torch::ones {2 2}]
    catch {torch::pow -base $base -exponent $exponent -invalidParam value} error
    string match "*Unknown parameter*" $error
} {1}

test pow-4.10 {Error: Missing base parameter - named} {
    set exponent [torch::ones {2 2}]
    catch {torch::pow -exponent $exponent} error
    string match "*Required parameters missing*" $error
} {1}

test pow-4.11 {Error: Missing exponent parameter - named} {
    set base [torch::ones {2 2}]
    catch {torch::pow -base $base} error
    string match "*Required parameters missing*" $error
} {1}

# ============================================================================
# Mathematical Correctness Tests
# ============================================================================

test pow-5.1 {Pow mathematical correctness: 2^2 = 4} {
    set base [torch::tensor_create {2.0}]
    set exponent [torch::tensor_create {2.0}]
    set result [torch::pow $base $exponent]
    set value [torch::tensor_item $result]
    expr {abs($value - 4.0) < 1e-6}
} {1}

test pow-5.2 {Pow mathematical correctness: 3^3 = 27} {
    set base [torch::tensor_create {3.0}]
    set exponent [torch::tensor_create {3.0}]
    set result [torch::pow $base $exponent]
    set value [torch::tensor_item $result]
    expr {abs($value - 27.0) < 1e-6}
} {1}

test pow-5.3 {Pow mathematical correctness: 5^0 = 1} {
    set base [torch::tensor_create {5.0}]
    set exponent [torch::tensor_create {0.0}]
    set result [torch::pow $base $exponent]
    set value [torch::tensor_item $result]
    expr {abs($value - 1.0) < 1e-6}
} {1}

test pow-5.4 {Pow with named parameters: 2^3 = 8} {
    set base [torch::tensor_create {2.0}]
    set exponent [torch::tensor_create {3.0}]
    set result [torch::pow -base $base -exponent $exponent]
    set value [torch::tensor_item $result]
    expr {abs($value - 8.0) < 1e-6}
} {1}

test pow-5.5 {Pow with mixed parameter names: 4^0.5 = 2} {
    set base [torch::tensor_create {4.0}]
    set exponent [torch::tensor_create {0.5}]
    set result [torch::pow -input1 $base -input2 $exponent]
    set value [torch::tensor_item $result]
    expr {abs($value - 2.0) < 1e-6}
} {1}

# ============================================================================
# Edge Cases and Special Values
# ============================================================================

test pow-6.1 {Large tensor dimensions} {
    set base [torch::ones {100 200}]
    set exponent [torch::ones {100 200}]
    set result [torch::pow $base $exponent]
    string match "tensor*" $result
} {1}

test pow-6.2 {Single element tensors} {
    set base [torch::ones {1}]
    set exponent [torch::ones {1}]
    set result [torch::pow $base $exponent]
    string match "tensor*" $result
} {1}

test pow-6.3 {4D tensors} {
    set base [torch::ones {2 3 4 5}]
    set exponent [torch::ones {2 3 4 5}]
    set result [torch::pow $base $exponent]
    string match "tensor*" $result
} {1}

test pow-6.4 {5D tensors} {
    set base [torch::ones {2 3 4 5 6}]
    set exponent [torch::ones {2 3 4 5 6}]
    set result [torch::pow $base $exponent]
    string match "tensor*" $result
} {1}

# ============================================================================
# Syntax Consistency Tests
# ============================================================================

test pow-7.1 {Positional and named syntax equivalence} {
    set base1 [torch::tensor_create {2.0}]
    set base2 [torch::tensor_create {2.0}]
    set exp1 [torch::tensor_create {3.0}]
    set exp2 [torch::tensor_create {3.0}]
    
    set result1 [torch::pow $base1 $exp1]
    set result2 [torch::pow -base $base2 -exponent $exp2]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    expr {abs($value1 - $value2) < 1e-6}
} {1}

test pow-7.2 {Different parameter name equivalence} {
    set base1 [torch::tensor_create {3.0}]
    set base2 [torch::tensor_create {3.0}]
    set exp1 [torch::tensor_create {2.0}]
    set exp2 [torch::tensor_create {2.0}]
    
    set result1 [torch::pow -base $base1 -exponent $exp1]
    set result2 [torch::pow -input1 $base2 -input2 $exp2]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    expr {abs($value1 - $value2) < 1e-6}
} {1}

cleanupTests 