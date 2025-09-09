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

# Helper to create test tensors (using standard pattern)
proc create_test_tensor {args} {
    # Flexible helper: if called with two arguments, treat first as variable
    # name and second as the shape list. If called with a single argument,
    # simply return a ones tensor with the given shape (suitable for acosh).
    if {[llength $args] == 1} {
        # Usage: create_test_tensor {2 2}
        set tensorShape [lindex $args 0]
        return [torch::ones $tensorShape]
    } elseif {[llength $args] == 2} {
        # Usage: create_test_tensor myVar {2 2}
        set varName [lindex $args 0]
        set tensorShape [lindex $args 1]
        upvar 1 $varName var
        set var [torch::ones $tensorShape]
        return $var
    } else {
        error {wrong # args: should be "create_test_tensor ?varName? shape"}
    }
}

# ============================================================================
# Basic Acosh Tests - Positional Syntax
# ============================================================================

test acosh-1.1 {Basic acosh with positional syntax} {
    set input [torch::ones {3 4}]
    set result [torch::acosh $input]
    string match "tensor*" $result
} {1}

test acosh-1.2 {Acosh with different tensor sizes - positional syntax} {
    set input [torch::ones {2 3}]
    set result [torch::acosh $input]
    string match "tensor*" $result
} {1}

test acosh-1.3 {Acosh with larger tensor - positional syntax} {
    set input [torch::ones {4 5}]
    set result [torch::acosh $input]
    string match "tensor*" $result
} {1}

test acosh-1.4 {Acosh with 1D tensor - positional syntax} {
    set input [torch::ones {10}]
    set result [torch::acosh $input]
    string match "tensor*" $result
} {1}

test acosh-1.5 {Acosh with 3D tensor - positional syntax} {
    set input [torch::ones {2 3 4}]
    set result [torch::acosh $input]
    string match "tensor*" $result
} {1}

# ============================================================================
# Acosh Tests - Named Parameter Syntax
# ============================================================================

test acosh-2.1 {Basic acosh with named parameter syntax - input parameter} {
    set input [torch::ones {3 4}]
    set result [torch::acosh -input $input]
    string match "tensor*" $result
} {1}

test acosh-2.2 {Acosh with named parameter syntax - tensor parameter} {
    set input [torch::ones {2 3}]
    set result [torch::acosh -tensor $input]
    string match "tensor*" $result
} {1}

test acosh-2.3 {Acosh with different tensor sizes - named syntax} {
    set input [torch::ones {4 5}]
    set result [torch::acosh -input $input]
    string match "tensor*" $result
} {1}

test acosh-2.4 {Acosh with 1D tensor - named syntax} {
    set input [torch::ones {10}]
    set result [torch::acosh -input $input]
    string match "tensor*" $result
} {1}

test acosh-2.5 {Acosh with 3D tensor - named syntax} {
    set input [torch::ones {2 3 4}]
    set result [torch::acosh -tensor $input]
    string match "tensor*" $result
} {1}

# ============================================================================
# Parameter Validation Tests
# ============================================================================

test acosh-4.1 {Error: Missing required parameter - positional} {
    catch {torch::acosh} error
    string match "*Usage*" $error
} {1}

test acosh-4.2 {Error: Missing required parameter - named} {
    catch {torch::acosh -input} error
    string match "*Missing value*" $error
} {1}

test acosh-4.3 {Error: Invalid tensor name - positional} {
    catch {torch::acosh nonexistent_tensor} error
    string match "*Invalid tensor name*" $error
} {1}

test acosh-4.4 {Error: Invalid tensor name - named} {
    catch {torch::acosh -input nonexistent_tensor} error
    string match "*Invalid tensor name*" $error
} {1}

test acosh-4.5 {Error: Unknown parameter - named} {
    set input [torch::ones {2 2}]
    catch {torch::acosh -input $input -invalidParam value} error
    string match "*Unknown parameter*" $error
} {1}

test acosh-4.6 {Error: Too many positional arguments} {
    set input [torch::ones {2 2}]
    catch {torch::acosh $input extra_arg} error
    string match "*Usage*" $error
} {1}

test acosh-4.7 {Error: Missing parameter value - named} {
    catch {torch::acosh -tensor} error
    string match "*Missing value*" $error
} {1}

# ============================================================================
# Edge Cases and Special Values
# ============================================================================

test acosh-5.1 {Large tensor dimensions} {
    set input [torch::ones {100 200}]
    set result [torch::acosh $input]
    string match "tensor*" $result
} {1}

test acosh-5.2 {Single element tensor} {
    set input [torch::ones {1}]
    set result [torch::acosh $input]
    string match "tensor*" $result
} {1}

test acosh-5.3 {4D tensor} {
    set input [torch::ones {2 3 4 5}]
    set result [torch::acosh $input]
    string match "tensor*" $result
} {1}

cleanupTests 