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

# Positional syntax

test tensor-var-1.1 {Basic positional syntax, 1D tensor} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float32]
    set result [torch::tensor_var $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 2.5) < 1e-5}
} {1}

test tensor-var-1.2 {Positional syntax, 2D tensor, dim=0} {
    set t2 [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
    set result [torch::tensor_var $t2 0]
    set result_list [torch::tensor_to_list -input $result]
    expr {[llength $result_list] == 3 && abs([lindex $result_list 0] - 4.5) < 1e-5 && abs([lindex $result_list 1] - 4.5) < 1e-5 && abs([lindex $result_list 2] - 4.5) < 1e-5 ? 1 : 0}
} {1}

test tensor-var-1.3 {Positional syntax, 2D tensor, dim=1, unbiased=false} {
    set t3 [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
    set result [torch::tensor_var $t3 1 0]
    set result_list [torch::tensor_to_list -input $result]
    expr {[llength $result_list] == 2 && abs([lindex $result_list 0] - 0.6666666865348816) < 1e-5 && abs([lindex $result_list 1] - 0.6666666865348816) < 1e-5 ? 1 : 0}
} {1}

# Named parameter syntax
test tensor-var-2.1 {Named parameter syntax, 1D tensor} {
    set t4 [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float32]
    set result [torch::tensor_var -input $t4]
    set value [torch::tensor_item $result]
    expr {abs($value - 2.5) < 1e-5}
} {1}

test tensor-var-2.2 {Named parameter syntax, 2D tensor, dim=0} {
    set t5 [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
    set result [torch::tensor_var -input $t5 -dim 0]
    set result_list [torch::tensor_to_list -input $result]
    expr {[llength $result_list] == 3 && abs([lindex $result_list 0] - 4.5) < 1e-5 && abs([lindex $result_list 1] - 4.5) < 1e-5 && abs([lindex $result_list 2] - 4.5) < 1e-5 ? 1 : 0}
} {1}

test tensor-var-2.3 {Named parameter syntax, 2D tensor, dim=1, unbiased=false} {
    set t6 [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
    set result [torch::tensor_var -input $t6 -dim 1 -unbiased 0]
    set result_list [torch::tensor_to_list -input $result]
    expr {[llength $result_list] == 2 && abs([lindex $result_list 0] - 0.6666666865348816) < 1e-5 && abs([lindex $result_list 1] - 0.6666666865348816) < 1e-5 ? 1 : 0}
} {1}

# CamelCase alias
test tensor-var-3.1 {CamelCase alias, 1D tensor} {
    set t7 [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float32]
    set result [torch::tensorVar -input $t7]
    set value [torch::tensor_item $result]
    expr {abs($value - 2.5) < 1e-5}
} {1}

test tensor-var-3.2 {CamelCase alias, 2D tensor, dim=1, unbiased=false} {
    set t8 [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
    set result [torch::tensorVar -input $t8 -dim 1 -unbiased 0]
    set result_list [torch::tensor_to_list -input $result]
    expr {[llength $result_list] == 2 && abs([lindex $result_list 0] - 0.6666666865348816) < 1e-5 && abs([lindex $result_list 1] - 0.6666666865348816) < 1e-5 ? 1 : 0}
} {1}

# Error handling
test tensor-var-4.1 {Error: missing tensor} {
    catch {torch::tensor_var} msg
    expr {[string match "*Usage*" $msg] || [string match "*Required*" $msg]}
} {1}

test tensor-var-4.2 {Error: invalid tensor name} {
    catch {torch::tensor_var -input not_a_tensor} msg
    expr {[string match "*Invalid tensor name*" $msg] ? 1 : 0}
} {1}

test tensor-var-4.3 {Error: unknown parameter} {
    set t9 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    catch {torch::tensor_var -input $t9 -foo 1} msg
    string match "*Unknown parameter*" $msg
} {1}

test tensor-var-4.4 {Error: missing parameter value} {
    set t10 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    catch {torch::tensor_var -input $t10 -dim} msg
    string match "*Missing value for parameter*" $msg
} {1}

# Edge cases
test tensor-var-5.1 {Edge: all same values} {
    set t11 [torch::tensor_create -data {7.0 7.0 7.0 7.0} -dtype float32]
    set result [torch::tensor_var $t11]
    set value [torch::tensor_item $result]
    expr {abs($value) < 1e-5}
} {1}

test tensor-var-5.2 {Edge: single value tensor} {
    set t12 [torch::tensor_create -data {42.0} -dtype float32]
    set result [torch::tensor_var $t12]
    set value [torch::tensor_item $result]
    expr {[string match "*-nan*" $value] ? 1 : 0}
} {1}

cleanupTests 