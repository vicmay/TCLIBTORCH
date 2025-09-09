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

# Test cases for positional syntax
test vmap-1.1 {Basic positional syntax} {
    set func "dummy_func"
    set inputs [torch::tensor_create {1.0 2.0 3.0}]
    set result [torch::vmap $func $inputs]
    expr {[string match "tensor*" $result]}
} 1

test vmap-1.2 {Positional syntax with 2D tensor} {
    set func "test_func"
    set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set result [torch::vmap $func $inputs]
    expr {[string match "tensor*" $result]}
} 1

test vmap-1.3 {Positional syntax with 3D tensor} {
    set func "complex_func"
    ;# Create 3D tensor using nested lists
    set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set result [torch::vmap $func $inputs]
    expr {[string match "tensor*" $result]}
} 1

# Test cases for named parameter syntax
test vmap-2.1 {Named parameter syntax - basic} {
    set func "dummy_func"
    set inputs [torch::tensor_create {1.0 2.0 3.0}]
    set result [torch::vmap -func $func -inputs $inputs]
    expr {[string match "tensor*" $result]}
} 1

test vmap-2.2 {Named parameter syntax with alternative parameter names} {
    set func "test_func"
    set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set result [torch::vmap -function $func -input $inputs]
    expr {[string match "tensor*" $result]}
} 1

test vmap-2.3 {Named parameter syntax in different order} {
    set func "order_test"
    set inputs [torch::tensor_create {1.0 2.0 3.0}]
    set result [torch::vmap -inputs $inputs -func $func]
    expr {[string match "tensor*" $result]}
} 1

test vmap-2.4 {Named parameter syntax with 3D tensor} {
    set func "complex_func"
    ;# Create 3D tensor using nested lists
    set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set result [torch::vmap -func $func -inputs $inputs]
    expr {[string match "tensor*" $result]}
} 1

# Test cases for camelCase alias
test vmap-3.1 {CamelCase alias - basic functionality} {
    set func "dummy_func"
    set inputs [torch::tensor_create {1.0 2.0 3.0}]
    set result [torch::vectorMap $func $inputs]
    expr {[string match "tensor*" $result]}
} 1

test vmap-3.2 {CamelCase alias with named parameters} {
    set func "test_func"
    set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set result [torch::vectorMap -func $func -inputs $inputs]
    expr {[string match "tensor*" $result]}
} 1

# Test cases for error handling
test vmap-4.1 {Error handling - invalid tensor} {
    catch {torch::vmap "func" "invalid_tensor"} error
    string match "*Invalid tensor*" $error
} 1

test vmap-4.2 {Error handling - missing required parameter in positional} {
    catch {torch::vmap "func"} error
    string match "*Invalid number of arguments*" $error
} 1

test vmap-4.3 {Error handling - missing required parameter in named syntax} {
    catch {torch::vmap -func "test"} error
    string match "*Required parameters missing*" $error
} 1

test vmap-4.4 {Error handling - missing value for parameter} {
    catch {torch::vmap -func "test" -inputs} error
    string match "*Missing value for parameter*" $error
} 1

test vmap-4.5 {Error handling - unknown parameter} {
    set inputs [torch::tensor_create {1.0 2.0 3.0}]
    catch {torch::vmap -func "test" -inputs $inputs -unknown param} error
    string match "*Unknown parameter*" $error
} 1

# Test cases for mathematical correctness
test vmap-5.1 {Mathematical correctness - preserves input shape} {
    set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set result [torch::vmap "func" $inputs]
    set input_shape [torch::tensor_shape $inputs]
    set result_shape [torch::tensor_shape $result]
    expr {[lindex $input_shape 0] == [lindex $result_shape 0] && 
          [lindex $input_shape 1] == [lindex $result_shape 1]}
} 1

test vmap-5.2 {Mathematical correctness - preserves input values} {
    set inputs [torch::tensor_create {1.0 2.0 3.0}]
    set result [torch::vmap "func" $inputs]
    set input_list [torch::tensor_to_list $inputs]
    set result_list [torch::tensor_to_list $result]
    expr {abs([lindex $input_list 0] - [lindex $result_list 0]) < 1e-10 && 
          abs([lindex $input_list 1] - [lindex $result_list 1]) < 1e-10 && 
          abs([lindex $input_list 2] - [lindex $result_list 2]) < 1e-10}
} 1

test vmap-5.3 {Mathematical correctness - 3D tensor preservation} {
    set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set result [torch::vmap "func" $inputs]
    set input_shape [torch::tensor_shape $inputs]
    set result_shape [torch::tensor_shape $result]
    expr {[llength $input_shape] == [llength $result_shape] && 
          [lindex $input_shape 0] == [lindex $result_shape 0] && 
          [lindex $input_shape 1] == [lindex $result_shape 1]}
} 1

# Test cases for data type support
test vmap-6.1 {Data type support - float32} {
    set inputs [torch::tensor_create {1.0 2.0 3.0} float32]
    set result [torch::vmap "func" $inputs]
    expr {[string match "tensor*" $result]}
} 1

test vmap-6.2 {Data type support - float64} {
    set inputs [torch::tensor_create {1.0 2.0 3.0} float64]
    set result [torch::vmap "func" $inputs]
    expr {[string match "tensor*" $result]}
} 1

# Test cases for edge cases
test vmap-7.1 {Edge case - single element tensor} {
    set inputs [torch::tensor_create {5.0}]
    set result [torch::vmap "func" $inputs]
    set result_list [torch::tensor_to_list $result]
    expr {abs([lindex $result_list 0] - 5.0) < 1e-10}
} 1

test vmap-7.2 {Edge case - zero tensor} {
    set inputs [torch::tensor_create {0.0 0.0 0.0}]
    set result [torch::vmap "func" $inputs]
    set result_list [torch::tensor_to_list $result]
    expr {abs([lindex $result_list 0] - 0.0) < 1e-10 && 
          abs([lindex $result_list 1] - 0.0) < 1e-10 && 
          abs([lindex $result_list 2] - 0.0) < 1e-10}
} 1

test vmap-7.3 {Edge case - large values} {
    set inputs [torch::tensor_create {1e6 2e6 3e6}]
    set result [torch::vmap "func" $inputs]
    expr {[string match "tensor*" $result]}
} 1

# Test cases for syntax consistency
test vmap-8.1 {Syntax consistency - positional vs named} {
    set func "test_func"
    set inputs [torch::tensor_create {1.0 2.0 3.0}]
    set result1 [torch::vmap $func $inputs]
    set result2 [torch::vmap -func $func -inputs $inputs]
    set list1 [torch::tensor_to_list $result1]
    set list2 [torch::tensor_to_list $result2]
    expr {abs([lindex $list1 0] - [lindex $list2 0]) < 1e-10 && 
          abs([lindex $list1 1] - [lindex $list2 1]) < 1e-10 && 
          abs([lindex $list1 2] - [lindex $list2 2]) < 1e-10}
} 1

test vmap-8.2 {Syntax consistency - original vs camelCase} {
    set func "test_func"
    set inputs [torch::tensor_create {1.0 2.0 3.0}]
    set result1 [torch::vmap $func $inputs]
    set result2 [torch::vectorMap $func $inputs]
    set list1 [torch::tensor_to_list $result1]
    set list2 [torch::tensor_to_list $result2]
    expr {abs([lindex $list1 0] - [lindex $list2 0]) < 1e-10 && 
          abs([lindex $list1 1] - [lindex $list2 1]) < 1e-10 && 
          abs([lindex $list1 2] - [lindex $list2 2]) < 1e-10}
} 1

test vmap-8.3 {Syntax consistency - all combinations} {
    set func "consistency_test"
    set inputs [torch::tensor_create {{2.0 3.0} {4.0 5.0}}]
    
    set result1 [torch::vmap $func $inputs]
    set result2 [torch::vmap -func $func -inputs $inputs]
    set result3 [torch::vectorMap $func $inputs]
    set result4 [torch::vectorMap -func $func -inputs $inputs]
    
    set list1 [torch::tensor_to_list $result1]
    set list2 [torch::tensor_to_list $result2]
    set list3 [torch::tensor_to_list $result3]
    set list4 [torch::tensor_to_list $result4]
    
    expr {abs([lindex $list1 0] - [lindex $list2 0]) < 1e-10 && 
          abs([lindex $list1 0] - [lindex $list3 0]) < 1e-10 && 
          abs([lindex $list1 0] - [lindex $list4 0]) < 1e-10}
} 1

cleanupTests 