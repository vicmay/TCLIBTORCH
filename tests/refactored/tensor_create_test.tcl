# tests/refactored/tensor_create_test.tcl
# Test file for refactored torch::tensor_create / torch::tensorCreate command

#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test 1: Basic positional syntax
test tensor_create-1.1 {Basic positional syntax} {
    set t [torch::tensor_create {1.0 2.0 3.0}]
    expr {$t ne ""}
} {1}

# Test 2: Basic named parameter syntax
test tensor_create-2.1 {Basic named parameter syntax} {
    set t [torch::tensor_create -data {1.0 2.0 3.0}]
    expr {$t ne ""}
} {1}

# Test 3: CamelCase alias
test tensor_create-3.1 {CamelCase alias} {
    set t [torch::tensorCreate -data {1.0 2.0 3.0}]
    expr {$t ne ""}
} {1}

# Test 4: Positional syntax with dtype
test tensor_create-4.1 {Positional syntax with dtype} {
    set t [torch::tensor_create {1 2 3} int32]
    expr {$t ne ""}
} {1}

# Test 5: Named syntax with dtype
test tensor_create-5.1 {Named syntax with dtype} {
    set t [torch::tensor_create -data {1 2 3} -dtype int32]
    expr {$t ne ""}
} {1}

# Test 6: Positional syntax with device
test tensor_create-6.1 {Positional syntax with device} {
    set t [torch::tensor_create {1.0 2.0 3.0} float32 cpu]
    expr {$t ne ""}
} {1}

# Test 7: Named syntax with device
test tensor_create-7.1 {Named syntax with device} {
    set t [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    expr {$t ne ""}
} {1}

# Test 8: Positional syntax with requires_grad
test tensor_create-8.1 {Positional syntax with requires_grad} {
    set t [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    expr {$t ne ""}
} {1}

# Test 9: Named syntax with requires_grad
test tensor_create-9.1 {Named syntax with requires_grad} {
    set t [torch::tensor_create -data {1.0 2.0 3.0} -requiresGrad true]
    expr {$t ne ""}
} {1}

# Test 10: Positional syntax with shape
test tensor_create-10.1 {Positional syntax with shape} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    set shape [torch::tensor_shape $t]
    set shape
} {2 3}

# Test 11: Named syntax with shape
test tensor_create-11.1 {Named syntax with shape} {
    set t [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3}]
    set shape [torch::tensor_shape $t]
    set shape
} {2 3}

# Test 12: Positional syntax with shape and dtype
test tensor_create-12.1 {Positional syntax with shape and dtype} {
    set t [torch::tensor_create {1 2 3 4 5 6} {2 3} int64]
    set shape [torch::tensor_shape $t]
    set shape
} {2 3}

# Test 13: Named syntax with all parameters
test tensor_create-13.1 {Named syntax with all parameters} {
    set t [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float64 -device cpu -requiresGrad false]
    set shape [torch::tensor_shape $t]
    set shape
} {2 2}

# Test 14: Different data types - float64
test tensor_create-14.1 {Different data types - float64} {
    set t [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64]
    expr {$t ne ""}
} {1}

# Test 15: Different data types - int64
test tensor_create-15.1 {Different data types - int64} {
    set t [torch::tensor_create -data {1 2 3} -dtype int64]
    expr {$t ne ""}
} {1}

# Test 16: Different data types - bool
test tensor_create-16.1 {Different data types - bool} {
    set t [torch::tensor_create -data {1 0 1} -dtype bool]
    expr {$t ne ""}
} {1}

# Test 17: Alternative dtype names - float
test tensor_create-17.1 {Alternative dtype names - float} {
    set t [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    expr {$t ne ""}
} {1}

# Test 18: Alternative dtype names - int
test tensor_create-18.1 {Alternative dtype names - int} {
    set t [torch::tensor_create -data {1 2 3} -dtype int32]
    expr {$t ne ""}
} {1}

# Test 19: Syntax consistency - positional vs named
test tensor_create-19.1 {Syntax consistency - positional vs named} {
    set t1 [torch::tensor_create {1.0 2.0 3.0} float32 cpu false]
    set t2 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad false]
    # Both should create valid tensors
    expr {$t1 ne "" && $t2 ne ""}
} {1}

# Test 20: Error handling - missing data parameter
test tensor_create-20.1 {Error handling - missing data parameter} {
    catch {torch::tensor_create -dtype float32} result
    string match "*Missing required parameter: -data*" $result
} {1}

# Test 21: Error handling - invalid dtype
test tensor_create-21.1 {Error handling - invalid dtype} {
    catch {torch::tensor_create -data {1 2 3} -dtype badtype} result
    string match "*Invalid dtype: badtype*" $result
} {1}

# Test 22: Error handling - unknown parameter
test tensor_create-22.1 {Error handling - unknown parameter} {
    catch {torch::tensor_create -data {1 2 3} -unknown param} result
    string match "*Unknown parameter: -unknown*" $result
} {1}

# Test 23: Error handling - missing value for parameter
test tensor_create-23.1 {Error handling - missing value for parameter} {
    catch {torch::tensor_create -data {1 2 3} -dtype} result
    string match "*Missing value for parameter*" $result
} {1}

# Test 24: Error handling - invalid boolean for requiresGrad
test tensor_create-24.1 {Error handling - invalid boolean for requiresGrad} {
    catch {torch::tensor_create -data {1 2 3} -requiresGrad invalid} result
    string match "*Invalid boolean for -requiresGrad*" $result
} {1}

# Test 25: Empty data list
test tensor_create-25.1 {Empty data list} {
    set t [torch::tensor_create -data {}]
    expr {$t ne ""}
} {1}

# Test 26: Single value tensor
test tensor_create-26.1 {Single value tensor} {
    set t [torch::tensor_create -data {5.0}]
    expr {$t ne ""}
} {1}

# Test 27: Negative values
test tensor_create-27.1 {Negative values} {
    set t [torch::tensor_create -data {-1.0 -2.0 -3.0}]
    expr {$t ne ""}
} {1}

# Test 28: Mixed positive and negative
test tensor_create-28.1 {Mixed positive and negative} {
    set t [torch::tensor_create -data {-1.0 0.0 1.0}]
    expr {$t ne ""}
} {1}

# Test 29: Large numbers
test tensor_create-29.1 {Large numbers} {
    set t [torch::tensor_create -data {1000000.0 2000000.0 3000000.0}]
    expr {$t ne ""}
} {1}

# Test 30: Fractional numbers
test tensor_create-30.1 {Fractional numbers} {
    set t [torch::tensor_create -data {0.1 0.2 0.3}]
    expr {$t ne ""}
} {1}

# Test 31: Parameter order flexibility (named syntax)
test tensor_create-31.1 {Parameter order flexibility} {
    set t [torch::tensor_create -dtype float32 -data {1.0 2.0 3.0} -device cpu]
    expr {$t ne ""}
} {1}

# Test 32: Backward compatibility verification
test tensor_create-32.1 {Backward compatibility verification} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0}]
    expr {$t ne ""}
} {1}

# Test 33: 3D shape tensor
test tensor_create-33.1 {3D shape tensor} {
    set t [torch::tensor_create -data {1 2 3 4 5 6 7 8} -shape {2 2 2}]
    set shape [torch::tensor_shape $t]
    set shape
} {2 2 2}

cleanupTests 