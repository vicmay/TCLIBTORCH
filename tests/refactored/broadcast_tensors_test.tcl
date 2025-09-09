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

# Test cases for positional syntax (backward compatibility)
test broadcast_tensors-1.1 {Basic positional syntax with two tensors} {
    set t1 [torch::tensor_create {1 2 3} float32]
    set t2 [torch::tensor_create {4} float32]
    set result [torch::broadcast_tensors $t1 $t2]
    # Should return a list of two tensor handles
    expr {[llength $result] == 2}
} {1}

test broadcast_tensors-1.2 {Positional syntax with three tensors} {
    set t1 [torch::tensor_create {1} float32]
    set t2 [torch::tensor_create {2} float32]
    set t3 [torch::tensor_create {3} float32]
    set result [torch::broadcast_tensors $t1 $t2 $t3]
    # Should return a list of three tensor handles
    expr {[llength $result] == 3}
} {1}

test broadcast_tensors-1.3 {Positional syntax with scalars and vectors} {
    set t1 [torch::tensor_create {5} float32]
    set t2 [torch::tensor_create {1 2} float32]
    set result [torch::broadcast_tensors $t1 $t2]
    # Should return a list of two tensor handles
    expr {[llength $result] == 2}
} {1}

# Test cases for named parameter syntax
test broadcast_tensors-2.1 {Named parameters with list format} {
    set t1 [torch::tensor_create {1 2} float32]
    set t2 [torch::tensor_create {3} float32]
    set result [torch::broadcast_tensors -tensors [list $t1 $t2]]
    # Should return a list of two tensor handles
    expr {[llength $result] == 2}
} {1}

test broadcast_tensors-2.2 {Named parameters with three tensors in list} {
    set t1 [torch::tensor_create {1} float32]
    set t2 [torch::tensor_create {2} float32]
    set t3 [torch::tensor_create {3} float32]
    set tensor_list [list $t1 $t2 $t3]
    set result [torch::broadcast_tensors -tensors $tensor_list]
    # Should return a list of three tensor handles
    expr {[llength $result] == 3}
} {1}

test broadcast_tensors-2.3 {Named parameters with single tensor as string} {
    set t1 [torch::tensor_create {1 2 3} float32]
    set t2 [torch::tensor_create {4} float32]
    # Note: This tests the fallback to single tensor name when list parsing fails
    set result1 [torch::broadcast_tensors -tensors $t1]
    set result2 [torch::broadcast_tensors -tensors $t2]
    # Each should return a list with one tensor handle
    expr {[llength $result1] == 1 && [llength $result2] == 1}
} {1}

# Test cases for camelCase alias
test broadcast_tensors-3.1 {CamelCase alias with positional syntax} {
    set t1 [torch::tensor_create {1 2} float32]
    set t2 [torch::tensor_create {3} float32]
    set result [torch::broadcastTensors $t1 $t2]
    # Should return a list of two tensor handles
    expr {[llength $result] == 2}
} {1}

test broadcast_tensors-3.2 {CamelCase alias with named parameters} {
    set t1 [torch::tensor_create {7} float32]
    set t2 [torch::tensor_create {8 9} float32]
    set result [torch::broadcastTensors -tensors [list $t1 $t2]]
    # Should return a list of two tensor handles
    expr {[llength $result] == 2}
} {1}

# Test that both syntaxes produce equivalent results
test broadcast_tensors-4.1 {Both syntaxes produce same number of outputs} {
    set t1 [torch::tensor_create {1} float32]
    set t2 [torch::tensor_create {2 3} float32]
    
    set result1 [torch::broadcast_tensors $t1 $t2]
    set result2 [torch::broadcast_tensors -tensors [list $t1 $t2]]
    set result3 [torch::broadcastTensors $t1 $t2]
    set result4 [torch::broadcastTensors -tensors [list $t1 $t2]]
    
    # All results should have the same number of tensors
    expr {[llength $result1] == [llength $result2] && 
          [llength $result2] == [llength $result3] && 
          [llength $result3] == [llength $result4] && 
          [llength $result1] == 2}
} {1}

# Error handling tests
test broadcast_tensors-5.1 {Error on no arguments} {
    catch {torch::broadcast_tensors} result
    expr {[string match "*Usage*" $result] || [string match "*tensor*" $result]}
} {1}

test broadcast_tensors-5.2 {Error on invalid tensor name} {
    catch {torch::broadcast_tensors invalid_tensor} result
    expr {[string match "*Invalid tensor*" $result]}
} {1}

test broadcast_tensors-5.3 {Error on unknown parameter} {
    set t1 [torch::tensor_create {1 2} float32]
    catch {torch::broadcast_tensors -unknown_param $t1} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test broadcast_tensors-5.4 {Error on missing value for parameter} {
    catch {torch::broadcast_tensors -tensors} result
    expr {[string match "*Missing value for parameter*" $result]}
} {1}

test broadcast_tensors-5.5 {Error with mixed valid and invalid tensors} {
    set t1 [torch::tensor_create {1 2} float32]
    catch {torch::broadcast_tensors $t1 invalid_tensor} result
    expr {[string match "*Invalid tensor*" $result]}
} {1}

# Test mathematical correctness (basic verification)
test broadcast_tensors-6.1 {Results are valid tensor handles} {
    set t1 [torch::tensor_create {1} float32]
    set t2 [torch::tensor_create {2 3} float32]
    set result [torch::broadcast_tensors $t1 $t2]
    
    # Extract tensor handles and verify they're not empty
    set tensor1 [lindex $result 0]
    set tensor2 [lindex $result 1]
    expr {$tensor1 ne "" && $tensor2 ne ""}
} {1}

cleanupTests 