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

# Helper function to check if two tensors are approximately equal
proc tensor_approx_equal {t1 t2 {tolerance 1e-5}} {
    set diff [torch::tensor_sub $t1 $t2]
    set abs_diff [torch::tensor_abs $diff]
    set max_diff [torch::tensor_max $abs_diff]
    set max_val [torch::tensor_item $max_diff]
    expr {$max_val < $tolerance}
}

# Test cases for positional syntax
test tan-1.1 {Basic positional syntax} {
    set t1 [torch::tensor_create {0.0} float32]
    set result [torch::tan $t1]
    set expected [torch::tensor_create {0.0} float32]
    tensor_approx_equal $result $expected
} {1}

test tan-1.2 {Positional syntax with pi/4} {
    set t1 [torch::tensor_create {0.7853981633974483} float32]  ;# pi/4
    set result [torch::tan $t1]
    set expected [torch::tensor_create {1.0} float32]
    tensor_approx_equal $result $expected
} {1}

test tan-1.3 {Positional syntax with pi/2} {
    set t1 [torch::tensor_create {1.5707963267948966} float32]  ;# pi/2
    set result [torch::tan $t1]
    # tan(pi/2) is undefined (infinite), but torch returns a large value. We'll check its absolute value is > 1e6
    expr {abs([torch::tensor_item $result]) > 1e6}
} {1}

test tan-1.4 {Positional syntax with vector} {
    set t1 [torch::tensor_create {0.0 0.7853981633974483 1.5707963267948966} float32]
    set result [torch::tan $t1]
    # Try to use torch::tensor_to_list if available, otherwise skip
    if {[catch {set vals [torch::tensor_to_list $result]}]} {
        return -code skip "tensor_to_list not available"
    }
    set r0 [expr {abs([lindex $vals 0]) < 1e-5}]
    set r1 [expr {abs([lindex $vals 1] - 1.0) < 1e-5}]
    set r2 [expr {abs([lindex $vals 2]) > 1e6}]
    expr {$r0 && $r1 && $r2}
} {1}

# Test cases for named parameter syntax
test tan-2.1 {Named parameter syntax with -input} {
    set t1 [torch::tensor_create {0.0} float32]
    set result [torch::tan -input $t1]
    set expected [torch::tensor_create {0.0} float32]
    tensor_approx_equal $result $expected
} {1}

test tan-2.2 {Named parameter syntax with -tensor} {
    set t1 [torch::tensor_create {0.7853981633974483} float32]
    set result [torch::tan -tensor $t1]
    set expected [torch::tensor_create {1.0} float32]
    tensor_approx_equal $result $expected
} {1}

# Error handling tests
test tan-3.1 {Error on missing tensor} {
    catch {torch::tan} msg
    set msg
} {Usage: torch::tan tensor | torch::tan -input tensor}

test tan-3.2 {Error on invalid tensor name} {
    catch {torch::tan invalid_tensor} msg
    set msg
} {Invalid tensor name}

test tan-3.3 {Error on invalid parameter name} {
    set t1 [torch::tensor_create {0.0} float32]
    catch {torch::tan -invalid $t1} msg
    set msg
} {Unknown parameter: -invalid. Valid parameters are: -input, -tensor}

test tan-3.4 {Error on missing parameter value} {
    catch {torch::tan -input} msg
    set msg
} {Missing value for parameter}

cleanupTests 