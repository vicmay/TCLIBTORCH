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
test vjp-1.1 {Basic positional syntax} {
    set func "dummy_func"
    set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set v [torch::tensor_create {1.0 1.0}]
    set result [torch::vjp $func $inputs $v]
    expr {[string match "tensor*" $result]}
} 1

test vjp-1.2 {Positional syntax with different tensors} {
    set func "test_func" 
    set inputs [torch::tensor_create {{2.0 3.0} {4.0 5.0}}]
    set v [torch::tensor_create {0.5 1.5}]
    set result [torch::vjp $func $inputs $v]
    expr {[string match "tensor*" $result]}
} 1

test vjp-1.3 {Positional syntax with 1D tensors} {
    set func "simple_func"
    set inputs [torch::tensor_create {1.0 2.0 3.0}]
    set v [torch::tensor_create {1.0 1.0 1.0}]
    set result [torch::vjp $func $inputs $v]
    expr {[string match "tensor*" $result]}
} 1

# Test cases for named parameter syntax
test vjp-2.1 {Named parameter syntax - basic} {
    set func "dummy_func"
    set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set v [torch::tensor_create {1.0 1.0}]
    set result [torch::vjp -func $func -inputs $inputs -v $v]
    expr {[string match "tensor*" $result]}
} 1

test vjp-2.2 {Named parameter syntax with alternative parameter names} {
    set func "test_func"
    set inputs [torch::tensor_create {{2.0 3.0} {4.0 5.0}}]
    set v [torch::tensor_create {0.5 1.5}]
    set result [torch::vjp -function $func -input $inputs -vector $v]
    expr {[string match "tensor*" $result]}
} 1

test vjp-2.3 {Named parameter syntax in different order} {
    set func "order_test"
    set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set v [torch::tensor_create {1.0 1.0}]
    set result [torch::vjp -v $v -func $func -inputs $inputs]
    expr {[string match "tensor*" $result]}
} 1

test vjp-2.4 {Named parameter syntax with 1D tensors} {
    set func "simple_func"
    set inputs [torch::tensor_create {1.0 2.0 3.0}]
    set v [torch::tensor_create {1.0 1.0 1.0}]
    set result [torch::vjp -func $func -inputs $inputs -v $v]
    expr {[string match "tensor*" $result]}
} 1

# Test cases for camelCase alias
test vjp-3.1 {CamelCase alias - basic functionality} {
    set func "dummy_func"
    set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set v [torch::tensor_create {1.0 1.0}]
    set result [torch::vectorJacobianProduct $func $inputs $v]
    expr {[string match "tensor*" $result]}
} 1

test vjp-3.2 {CamelCase alias with named parameters} {
    set func "test_func"
    set inputs [torch::tensor_create {{2.0 3.0} {4.0 5.0}}]
    set v [torch::tensor_create {0.5 1.5}]
    set result [torch::vectorJacobianProduct -func $func -inputs $inputs -v $v]
    expr {[string match "tensor*" $result]}
} 1

# Test cases for error handling
test vjp-4.1 {Error handling - invalid tensor} {
    catch {torch::vjp "func" "invalid_tensor" "invalid_v"} error
    string match "*Invalid tensor*" $error
} 1

test vjp-4.2 {Error handling - missing required parameter in positional} {
    catch {torch::vjp "func" "inputs"} error
    string match "*Invalid number of arguments*" $error
} 1

test vjp-4.3 {Error handling - missing required parameter in named syntax} {
    catch {torch::vjp -func "test" -inputs "tensor1"} error
    string match "*Required parameters missing*" $error
} 1

test vjp-4.4 {Error handling - missing value for parameter} {
    catch {torch::vjp -func "test" -inputs} error
    string match "*Missing value for parameter*" $error
} 1

test vjp-4.5 {Error handling - unknown parameter} {
    set inputs [torch::tensor_create {{1.0 2.0}}]
    set v [torch::tensor_create {1.0}]
    catch {torch::vjp -func "test" -inputs $inputs -v $v -unknown param} error
    string match "*Unknown parameter*" $error
} 1

# Test cases for mathematical correctness
test vjp-5.1 {Mathematical correctness - simple case} {
    set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set v [torch::tensor_create {1.0 1.0}]
    set result [torch::vjp "func" $inputs $v]
    set result_list [torch::tensor_to_list $result]
    expr {[llength $result_list] == 2}
} 1

test vjp-5.2 {Mathematical correctness - check output shape} {
    set inputs [torch::tensor_create {{1.0 2.0 3.0} {4.0 5.0 6.0}}]
    set v [torch::tensor_create {1.0 1.0}]
    set result [torch::vjp "func" $inputs $v]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 3}
} 1

test vjp-5.3 {Mathematical correctness - different dimensions} {
    set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0} {5.0 6.0}}]
    set v [torch::tensor_create {0.5 1.0 1.5}]
    set result [torch::vjp "func" $inputs $v]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 2}
} 1

# Test cases for data type support
test vjp-6.1 {Data type support - float32} {
    set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32]
    set v [torch::tensor_create {1.0 1.0} float32]
    set result [torch::vjp "func" $inputs $v]
    expr {[string match "tensor*" $result]}
} 1

test vjp-6.2 {Data type support - float64} {
    set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float64]
    set v [torch::tensor_create {1.0 1.0} float64]
    set result [torch::vjp "func" $inputs $v]
    expr {[string match "tensor*" $result]}
} 1

# Test cases for edge cases
test vjp-7.1 {Edge case - single element tensors} {
    set inputs [torch::tensor_create {{5.0}}]
    set v [torch::tensor_create {2.0}]
    set result [torch::vjp "func" $inputs $v]
    set result_list [torch::tensor_to_list $result]
    expr {abs([lindex $result_list 0] - 10.0) < 1e-5}
} 1

test vjp-7.2 {Edge case - zero tensors} {
    set inputs [torch::tensor_create {{0.0 0.0} {0.0 0.0}}]
    set v [torch::tensor_create {1.0 1.0}]
    set result [torch::vjp "func" $inputs $v]
    set result_list [torch::tensor_to_list $result]
    expr {abs([lindex $result_list 0] - 0.0) < 1e-10 && abs([lindex $result_list 1] - 0.0) < 1e-10}
} 1

test vjp-7.3 {Edge case - large values} {
    set inputs [torch::tensor_create {{1e6 2e6} {3e6 4e6}}]
    set v [torch::tensor_create {1.0 1.0}]
    set result [torch::vjp "func" $inputs $v]
    expr {[string match "tensor*" $result]}
} 1

# Test cases for syntax consistency
test vjp-8.1 {Syntax consistency - positional vs named} {
    set func "test_func"
    set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set v [torch::tensor_create {1.0 1.0}]
    set result1 [torch::vjp $func $inputs $v]
    set result2 [torch::vjp -func $func -inputs $inputs -v $v]
    set list1 [torch::tensor_to_list $result1]
    set list2 [torch::tensor_to_list $result2]
    expr {abs([lindex $list1 0] - [lindex $list2 0]) < 1e-10 && 
          abs([lindex $list1 1] - [lindex $list2 1]) < 1e-10}
} 1

test vjp-8.2 {Syntax consistency - original vs camelCase} {
    set func "test_func"
    set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set v [torch::tensor_create {1.0 1.0}]
    set result1 [torch::vjp $func $inputs $v]
    set result2 [torch::vectorJacobianProduct $func $inputs $v]
    set list1 [torch::tensor_to_list $result1]
    set list2 [torch::tensor_to_list $result2]
    expr {abs([lindex $list1 0] - [lindex $list2 0]) < 1e-10 && 
          abs([lindex $list1 1] - [lindex $list2 1]) < 1e-10}
} 1

test vjp-8.3 {Syntax consistency - all combinations} {
    set func "consistency_test"
    set inputs [torch::tensor_create {{2.0 3.0} {4.0 5.0}}]
    set v [torch::tensor_create {1.0 1.0}]
    
    set result1 [torch::vjp $func $inputs $v]
    set result2 [torch::vjp -func $func -inputs $inputs -v $v]
    set result3 [torch::vectorJacobianProduct $func $inputs $v]
    set result4 [torch::vectorJacobianProduct -func $func -inputs $inputs -v $v]
    
    set list1 [torch::tensor_to_list $result1]
    set list2 [torch::tensor_to_list $result2]
    set list3 [torch::tensor_to_list $result3]
    set list4 [torch::tensor_to_list $result4]
    
    expr {abs([lindex $list1 0] - [lindex $list2 0]) < 1e-10 && 
          abs([lindex $list1 0] - [lindex $list3 0]) < 1e-10 && 
          abs([lindex $list1 0] - [lindex $list4 0]) < 1e-10}
} 1

cleanupTests 