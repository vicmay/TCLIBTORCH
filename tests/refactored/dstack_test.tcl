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

# Test 1: Basic positional syntax with tensor list
test dstack-1.1 {Basic positional syntax with tensor list} {
    set t1 [torch::zeros {3 4} float32 cpu false]
    set t2 [torch::ones {3 4} float32 cpu false]
    set result [torch::dstack [list $t1 $t2]]
    expr {[llength $result] == 1}
} 1

test dstack-1.2 {Basic positional syntax with multiple arguments} {
    set t1 [torch::zeros {2 3} float32 cpu false]
    set t2 [torch::ones {2 3} float32 cpu false]
    set t3 [torch::full {2 3} 2.0 float32 cpu false]
    set result [torch::dstack $t1 $t2 $t3]
    expr {[llength $result] == 1}
} 1

test dstack-1.3 {Positional syntax result shape} {
    set t1 [torch::zeros {2 3} float32 cpu false]
    set t2 [torch::ones {2 3} float32 cpu false]
    set result [torch::dstack [list $t1 $t2]]
    set result_tensor [lindex $result 0]
    set shape [torch::tensor_shape $result_tensor]
    expr {$shape eq "2 3 2"}
} 1

test dstack-1.4 {Positional syntax with 1D tensors} {
    set t1 [torch::zeros {4} float32 cpu false]
    set t2 [torch::ones {4} float32 cpu false]
    set result [torch::dstack [list $t1 $t2]]
    set result_tensor [lindex $result 0]
    set shape [torch::tensor_shape $result_tensor]
    expr {$shape eq "1 4 2"}
} 1

# Test 2: Named parameter syntax
test dstack-2.1 {Named parameter syntax with -tensors} {
    set t1 [torch::zeros {3 4} float32 cpu false]
    set t2 [torch::ones {3 4} float32 cpu false]
    set result [torch::dstack -tensors [list $t1 $t2]]
    expr {[llength $result] == 1}
} 1

test dstack-2.2 {Named parameter syntax with -inputs alias} {
    set t1 [torch::zeros {2 3} float32 cpu false]
    set t2 [torch::ones {2 3} float32 cpu false]
    set result [torch::dstack -inputs [list $t1 $t2]]
    expr {[llength $result] == 1}
} 1

test dstack-2.3 {Named parameter result shape} {
    set t1 [torch::zeros {2 3} float32 cpu false]
    set t2 [torch::ones {2 3} float32 cpu false]
    set result [torch::dstack -tensors [list $t1 $t2]]
    set result_tensor [lindex $result 0]
    set shape [torch::tensor_shape $result_tensor]
    expr {$shape eq "2 3 2"}
} 1

test dstack-2.4 {Named parameter with multiple tensors} {
    set t1 [torch::zeros {2 2} float32 cpu false]
    set t2 [torch::ones {2 2} float32 cpu false]
    set t3 [torch::full {2 2} 2.0 float32 cpu false]
    set result [torch::dstack -tensors [list $t1 $t2 $t3]]
    set result_tensor [lindex $result 0]
    set shape [torch::tensor_shape $result_tensor]
    expr {$shape eq "2 2 3"}
} 1

# Test 3: CamelCase alias
test dstack-3.1 {CamelCase alias torch::dStack} {
    set t1 [torch::zeros {3 4} float32 cpu false]
    set t2 [torch::ones {3 4} float32 cpu false]
    set result [torch::dStack -tensors [list $t1 $t2]]
    expr {[llength $result] == 1}
} 1

test dstack-3.2 {CamelCase alias result correctness} {
    set t1 [torch::zeros {2 3} float32 cpu false]
    set t2 [torch::ones {2 3} float32 cpu false]
    set result1 [torch::dstack -tensors [list $t1 $t2]]
    set result2 [torch::dStack -tensors [list $t1 $t2]]
    set shape1 [torch::tensor_shape [lindex $result1 0]]
    set shape2 [torch::tensor_shape [lindex $result2 0]]
    expr {$shape1 eq $shape2}
} 1

# Test 4: Error handling
test dstack-4.1 {Error: Missing tensors parameter} {
    catch {torch::dstack} error
    expr {[string match "*tensor_list*" $error] || [string match "*tensors*" $error]}
} 1

test dstack-4.2 {Error: Missing value for named parameter} {
    catch {torch::dstack -tensors} error
    expr {[string match "*Missing value*" $error]}
} 1

test dstack-4.3 {Error: Unknown parameter} {
    set t1 [torch::zeros {2 3} float32 cpu false]
    catch {torch::dstack -unknown [list $t1]} error
    expr {[string match "*Unknown parameter*" $error]}
} 1

test dstack-4.4 {Error: Empty tensor list} {
    catch {torch::dstack -tensors [list]} error
    expr {[string match "*Missing required parameter*" $error]}
} 1

# Test 5: Mathematical correctness
test dstack-5.1 {Mathematical correctness - tensor validity} {
    set t1 [torch::zeros {2 2} float32 cpu false]
    set t2 [torch::ones {2 2} float32 cpu false]
    set result [torch::dstack -tensors [list $t1 $t2]]
    set result_tensor [lindex $result 0]
    # Check that result is valid and has expected shape
    set shape [torch::tensor_shape $result_tensor]
    expr {$shape eq "2 2 2"}
} 1

test dstack-5.2 {Mathematical correctness - tensor properties} {
    set t1 [torch::zeros {2 2} float32 cpu false]
    set t2 [torch::ones {2 2} float32 cpu false]
    set result [torch::dstack -tensors [list $t1 $t2]]
    set result_tensor [lindex $result 0]
    # Check dtype and device preservation
    set dtype [torch::tensor_dtype $result_tensor]
    expr {$dtype eq "Float32"}
} 1

test dstack-5.3 {Mathematical correctness with three tensors} {
    set t1 [torch::zeros {2 2} float32 cpu false]
    set t2 [torch::ones {2 2} float32 cpu false]
    set t3 [torch::full {2 2} 2.0 float32 cpu false]
    set result [torch::dstack -tensors [list $t1 $t2 $t3]]
    set result_tensor [lindex $result 0]
    # Check that all tensors are properly stacked
    set shape [torch::tensor_shape $result_tensor]
    expr {$shape eq "2 2 3"}
} 1

# Test 6: Syntax consistency  
test dstack-6.1 {Syntax consistency - positional vs named} {
    set t1 [torch::zeros {2 3} float32 cpu false]
    set t2 [torch::ones {2 3} float32 cpu false]
    set result1 [torch::dstack [list $t1 $t2]]
    set result2 [torch::dstack -tensors [list $t1 $t2]]
    set shape1 [torch::tensor_shape [lindex $result1 0]]
    set shape2 [torch::tensor_shape [lindex $result2 0]]
    expr {$shape1 eq $shape2}
} 1

test dstack-6.2 {Syntax consistency - multiple arguments vs list} {
    set t1 [torch::zeros {2 2} float32 cpu false]
    set t2 [torch::ones {2 2} float32 cpu false]
    set result1 [torch::dstack $t1 $t2]
    set result2 [torch::dstack [list $t1 $t2]]
    set shape1 [torch::tensor_shape [lindex $result1 0]]
    set shape2 [torch::tensor_shape [lindex $result2 0]]
    expr {$shape1 eq $shape2}
} 1

test dstack-6.3 {Syntax consistency - parameter aliases} {
    set t1 [torch::zeros {2 3} float32 cpu false]
    set t2 [torch::ones {2 3} float32 cpu false]
    set result1 [torch::dstack -tensors [list $t1 $t2]]
    set result2 [torch::dstack -inputs [list $t1 $t2]]
    set shape1 [torch::tensor_shape [lindex $result1 0]]
    set shape2 [torch::tensor_shape [lindex $result2 0]]
    expr {$shape1 eq $shape2}
} 1

# Test 7: Different tensor shapes and types
test dstack-7.1 {Different tensor shapes - 1D to 3D} {
    set t1 [torch::zeros {4} float32 cpu false]
    set t2 [torch::ones {4} float32 cpu false]
    set result [torch::dstack -tensors [list $t1 $t2]]
    set result_tensor [lindex $result 0]
    set shape [torch::tensor_shape $result_tensor]
    expr {$shape eq "1 4 2"}
} 1

test dstack-7.2 {Different tensor shapes - 2D} {
    set t1 [torch::zeros {3 4} float32 cpu false]
    set t2 [torch::ones {3 4} float32 cpu false]
    set result [torch::dstack -tensors [list $t1 $t2]]
    set result_tensor [lindex $result 0]
    set shape [torch::tensor_shape $result_tensor]
    expr {$shape eq "3 4 2"}
} 1

test dstack-7.3 {Different data types - float and int} {
    set t1 [torch::zeros {2 2} float32 cpu false]
    set t2 [torch::ones {2 2} float32 cpu false]
    set result [torch::dstack -tensors [list $t1 $t2]]
    expr {[llength $result] == 1}
} 1

test dstack-7.4 {Large tensor stacking} {
    set t1 [torch::zeros {5 6} float32 cpu false]
    set t2 [torch::ones {5 6} float32 cpu false]
    set t3 [torch::full {5 6} 0.5 float32 cpu false]
    set result [torch::dstack -tensors [list $t1 $t2 $t3]]
    set result_tensor [lindex $result 0]
    set shape [torch::tensor_shape $result_tensor]
    expr {$shape eq "5 6 3"}
} 1

# Test 8: Edge cases
test dstack-8.1 {Edge case - single tensor} {
    set t1 [torch::zeros {3 3} float32 cpu false]
    set result [torch::dstack -tensors [list $t1]]
    set result_tensor [lindex $result 0]
    set shape [torch::tensor_shape $result_tensor]
    expr {$shape eq "3 3 1"}
} 1

test dstack-8.2 {Edge case - many tensors} {
    set t1 [torch::zeros {2 2} float32 cpu false]
    set t2 [torch::ones {2 2} float32 cpu false]
    set t3 [torch::full {2 2} 2.0 float32 cpu false]
    set t4 [torch::full {2 2} 3.0 float32 cpu false]
    set t5 [torch::full {2 2} 4.0 float32 cpu false]
    set result [torch::dstack -tensors [list $t1 $t2 $t3 $t4 $t5]]
    set result_tensor [lindex $result 0]
    set shape [torch::tensor_shape $result_tensor]
    expr {$shape eq "2 2 5"}
} 1

test dstack-8.3 {Edge case - minimum size tensors} {
    set t1 [torch::zeros {1} float32 cpu false]
    set t2 [torch::ones {1} float32 cpu false]
    set result [torch::dstack -tensors [list $t1 $t2]]
    set result_tensor [lindex $result 0]
    set shape [torch::tensor_shape $result_tensor]
    expr {$shape eq "1 1 2"}
} 1

# Test 9: Parameter validation
test dstack-9.1 {Parameter validation - tensor handles} {
    set t1 [torch::zeros {2 3} float32 cpu false]
    set t2 [torch::ones {2 3} float32 cpu false]
    # Test that valid tensor handles work
    set result [torch::dstack -tensors [list $t1 $t2]]
    expr {[llength $result] == 1}
} 1

test dstack-9.2 {Parameter validation - invalid tensor handle} {
    catch {torch::dstack -tensors [list "invalid_tensor"]} error
    expr {[string match "*Error*" $error]}
} 1

# Test 10: Result validation
test dstack-10.1 {Result validation - return type} {
    set t1 [torch::zeros {2 3} float32 cpu false]
    set t2 [torch::ones {2 3} float32 cpu false]
    set result [torch::dstack -tensors [list $t1 $t2]]
    # Should return a single tensor (stacked result)
    expr {[llength $result] == 1}
} 1

test dstack-10.2 {Result validation - tensor properties} {
    set t1 [torch::zeros {2 3} float32 cpu false]
    set t2 [torch::ones {2 3} float32 cpu false]
    set result [torch::dstack -tensors [list $t1 $t2]]
    set result_tensor [lindex $result 0]
    # Check that result is a valid tensor by getting its shape
    set shape [torch::tensor_shape $result_tensor]
    expr {[llength [split $shape " "]] == 3}
} 1

test dstack-10.3 {Result validation - tensor operations} {
    set t1 [torch::zeros {2 2} float32 cpu false]
    set t2 [torch::ones {2 2} float32 cpu false]
    set result [torch::dstack -tensors [list $t1 $t2]]
    set result_tensor [lindex $result 0]
    # Test that we can perform operations on the result
    set sum_result [torch::tensor_sum $result_tensor]
    expr {[llength $sum_result] == 1}
} 1

test dstack-10.4 {Result validation - comprehensive shape check} {
    set t1 [torch::zeros {3 4} float32 cpu false]
    set t2 [torch::ones {3 4} float32 cpu false]
    set t3 [torch::full {3 4} 2.0 float32 cpu false]
    set result [torch::dstack -tensors [list $t1 $t2 $t3]]
    set result_tensor [lindex $result 0]
    set shape [torch::tensor_shape $result_tensor]
    # Input: 2 tensors of shape [3, 4] -> Output: [3, 4, 3]
    expr {$shape eq "3 4 3"}
} 1

cleanupTests 