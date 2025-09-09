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
test sin-1.1 {Basic positional syntax} {
    set t1 [torch::tensor_create {0.0} float32]
    set result [torch::sin $t1]
    set expected [torch::tensor_create {0.0} float32]
    tensor_approx_equal $result $expected
} {1}

test sin-1.2 {Positional syntax with pi/2} {
    set t1 [torch::tensor_create {1.5707963267948966} float32]  ;# pi/2
    set result [torch::sin $t1]
    set expected [torch::tensor_create {1.0} float32]
    tensor_approx_equal $result $expected
} {1}

test sin-1.3 {Positional syntax with pi} {
    set t1 [torch::tensor_create {3.141592653589793} float32]  ;# pi
    set result [torch::sin $t1]
    set expected [torch::tensor_create {0.0} float32]
    tensor_approx_equal $result $expected 1e-5
} {1}

test sin-1.4 {Positional syntax with vector} {
    set t1 [torch::tensor_create {0.0 1.5707963267948966 3.141592653589793} float32]
    set result [torch::sin $t1]
    set expected [torch::tensor_create {0.0 1.0 0.0} float32]
    tensor_approx_equal $result $expected
} {1}

# Test cases for named parameter syntax
test sin-2.1 {Named parameter syntax with -input} {
    set t1 [torch::tensor_create {0.0} float32]
    set result [torch::sin -input $t1]
    set expected [torch::tensor_create {0.0} float32]
    tensor_approx_equal $result $expected
} {1}

test sin-2.2 {Named parameter syntax with -tensor} {
    set t1 [torch::tensor_create {1.5707963267948966} float32]
    set result [torch::sin -tensor $t1]
    set expected [torch::tensor_create {1.0} float32]
    tensor_approx_equal $result $expected
} {1}

# Error handling tests
test sin-3.1 {Error on missing tensor} {
    catch {torch::sin} msg
    set msg
} {Usage: torch::sin tensor | torch::sin -input tensor}

test sin-3.2 {Error on invalid tensor name} {
    catch {torch::sin invalid_tensor} msg
    set msg
} {Invalid tensor name}

test sin-3.3 {Error on invalid parameter name} {
    set t1 [torch::tensor_create {0.0} float32]
    catch {torch::sin -invalid $t1} msg
    set msg
} {Unknown parameter: -invalid. Valid parameters are: -input, -tensor}

test sin-3.4 {Error on missing parameter value} {
    catch {torch::sin -input} msg
    set msg
} {Missing value for parameter}

cleanupTests 