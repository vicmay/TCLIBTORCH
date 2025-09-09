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
test tensor_advanced_index-1.1 {Basic positional syntax} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    set idx [torch::tensor_create {0 1} int64]
    set result [torch::tensor_advanced_index $t [list $idx]]
    expr {$result ne ""}
} {1}

test tensor_advanced_index-1.2 {Positional syntax with multiple indices} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    set idx1 [torch::tensor_create {0 1} int64]
    set idx2 [torch::tensor_create {1 2} int64]
    set result [torch::tensor_advanced_index $t [list $idx1 $idx2]]
    expr {$result ne ""}
} {1}

# Test cases for named parameter syntax
test tensor_advanced_index-2.1 {Named parameter syntax with -tensor and -indices} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    set idx [torch::tensor_create {0 1} int64]
    set result [torch::tensor_advanced_index -tensor $t -indices [list $idx]]
    expr {$result ne ""}
} {1}

test tensor_advanced_index-2.2 {Named parameter syntax with multiple indices} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    set idx1 [torch::tensor_create {0 1} int64]
    set idx2 [torch::tensor_create {1 2} int64]
    set result [torch::tensor_advanced_index -tensor $t -indices [list $idx1 $idx2]]
    expr {$result ne ""}
} {1}

# Test cases for camelCase alias
test tensor_advanced_index-3.1 {CamelCase alias} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    set idx [torch::tensor_create {0 1} int64]
    set result [torch::tensorAdvancedIndex -tensor $t -indices [list $idx]]
    expr {$result ne ""}
} {1}

# Error handling tests
test tensor_advanced_index-4.1 {Error on missing tensor} {
    catch {torch::tensor_advanced_index} msg
    set msg
} {Usage: torch::tensor_advanced_index tensor indices_list | torch::tensor_advanced_index -tensor tensor -indices indices_list}

test tensor_advanced_index-4.2 {Error on invalid tensor name} {
    set idx [torch::tensor_create {0 1} int64]
    catch {torch::tensor_advanced_index invalid_tensor [list $idx]} msg
    set msg
} {Tensor not found}

test tensor_advanced_index-4.3 {Error on invalid index tensor name} {
    set t [torch::tensor_create {1.0 2.0 3.0}]
    catch {torch::tensor_advanced_index $t [list invalid_index]} msg
    set msg
} {Index tensor not found}

test tensor_advanced_index-4.4 {Error on empty indices list} {
    set t [torch::tensor_create {1.0 2.0 3.0}]
    catch {torch::tensor_advanced_index $t [list]} msg
    set msg
} {Required parameters missing: tensor and indices list required}

test tensor_advanced_index-4.5 {Error on invalid parameter name} {
    set t [torch::tensor_create {1.0 2.0 3.0}]
    set idx [torch::tensor_create {0 1} int64]
    catch {torch::tensor_advanced_index -invalid $t -indices [list $idx]} msg
    set msg
} {Unknown parameter: -invalid. Valid parameters are: -tensor, -indices}

test tensor_advanced_index-4.6 {Error on missing parameter value} {
    catch {torch::tensor_advanced_index -tensor} msg
    set msg
} {Missing value for parameter}

test tensor_advanced_index-4.7 {Error on invalid indices list format} {
    set t [torch::tensor_create {1.0 2.0 3.0}]
    catch {torch::tensor_advanced_index $t "not_a_list"} msg
    set msg
} {Index tensor not found}

# Syntax consistency tests
test tensor_advanced_index-5.1 {Syntax consistency - positional vs named} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    set idx [torch::tensor_create {0 1} int64]
    set result1 [torch::tensor_advanced_index $t [list $idx]]
    set result2 [torch::tensor_advanced_index -tensor $t -indices [list $idx]]
    expr {$result1 ne "" && $result2 ne ""}
} {1}

cleanupTests 