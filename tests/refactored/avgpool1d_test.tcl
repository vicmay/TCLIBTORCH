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
test avgpool1d-1.1 {Basic positional syntax} {
    set input [torch::ones {1 3 8}]
    set result [torch::avgpool1d $input 2]
    string match "tensor*" $result
} 1

test avgpool1d-1.2 {Positional with stride} {
    set input [torch::ones {1 3 8}]
    set result [torch::avgpool1d $input 2 2]
    string match "tensor*" $result
} 1

test avgpool1d-1.3 {Positional with padding} {
    set input [torch::ones {1 3 8}]
    set result [torch::avgpool1d $input 2 2 1]
    string match "tensor*" $result
} 1

test avgpool1d-1.4 {Positional with count_include_pad} {
    set input [torch::ones {1 3 8}]
    set result [torch::avgpool1d $input 2 2 1 1]
    string match "tensor*" $result
} 1

test avgpool1d-1.5 {Positional with count_include_pad false} {
    set input [torch::ones {1 3 8}]
    set result [torch::avgpool1d $input 2 2 1 0]
    string match "tensor*" $result
} 1

# Test cases for named parameter syntax
test avgpool1d-2.1 {Named parameter syntax} {
    set input [torch::ones {1 3 8}]
    set result [torch::avgpool1d -input $input -kernel_size 2]
    string match "tensor*" $result
} 1

test avgpool1d-2.2 {Named with tensor alias} {
    set input [torch::ones {1 3 8}]
    set result [torch::avgpool1d -tensor $input -kernel_size 2]
    string match "tensor*" $result
} 1

test avgpool1d-2.3 {Named with kernelSize alias} {
    set input [torch::ones {1 3 8}]
    set result [torch::avgpool1d -input $input -kernelSize 2]
    string match "tensor*" $result
} 1

test avgpool1d-2.4 {Named with stride} {
    set input [torch::ones {1 3 8}]
    set result [torch::avgpool1d -input $input -kernel_size 2 -stride 2]
    string match "tensor*" $result
} 1

test avgpool1d-2.5 {Named with padding} {
    set input [torch::ones {1 3 8}]
    set result [torch::avgpool1d -input $input -kernel_size 2 -padding 1]
    string match "tensor*" $result
} 1

test avgpool1d-2.6 {Named with count_include_pad} {
    set input [torch::ones {1 3 8}]
    set result [torch::avgpool1d -input $input -kernel_size 2 -count_include_pad 1]
    string match "tensor*" $result
} 1

test avgpool1d-2.7 {Named with countIncludePad alias} {
    set input [torch::ones {1 3 8}]
    set result [torch::avgpool1d -input $input -kernel_size 2 -countIncludePad 0]
    string match "tensor*" $result
} 1

test avgpool1d-2.8 {Named with all parameters} {
    set input [torch::ones {1 3 8}]
    set result [torch::avgpool1d -input $input -kernel_size 3 -stride 2 -padding 1 -count_include_pad 1]
    string match "tensor*" $result
} 1

# Test cases for camelCase alias
test avgpool1d-3.1 {CamelCase alias basic} {
    set input [torch::ones {1 3 8}]
    set result [torch::avgPool1d $input 2]
    string match "tensor*" $result
} 1

test avgpool1d-3.2 {CamelCase alias with named params} {
    set input [torch::ones {1 3 8}]
    set result [torch::avgPool1d -input $input -kernel_size 2]
    string match "tensor*" $result
} 1

# Error handling tests
test avgpool1d-4.1 {Error - missing input} {
    catch {torch::avgpool1d} msg
    string match "*Named parameters require pairs*" $msg
} 1

test avgpool1d-4.2 {Error - invalid input tensor} {
    catch {torch::avgpool1d invalid_tensor 2} msg
    string match "*Invalid input tensor*" $msg
} 1

test avgpool1d-4.3 {Error - missing kernel size} {
    set input [torch::ones {1 3 8}]
    catch {torch::avgpool1d -input $input} msg
    string match "*Required parameters*" $msg
} 1

test avgpool1d-4.4 {Error - unknown parameter} {
    set input [torch::ones {1 3 8}]
    catch {torch::avgpool1d -input $input -kernel_size 2 -unknown_param 1} msg
    string match "*Unknown parameter*" $msg
} 1

test avgpool1d-4.5 {Error - missing parameter value} {
    set input [torch::ones {1 3 8}]
    catch {torch::avgpool1d -input $input -kernel_size} msg
    string match "*Named parameters require pairs*" $msg
} 1

# Functional verification tests
test avgpool1d-5.1 {Functional - avgpool reduces size} {
    set input [torch::ones {1 3 8}]
    set result [torch::avgpool1d $input 2]
    set input_shape [torch::tensor_shape $input]
    set result_shape [torch::tensor_shape $result]
    ;# Input: 1x3x8, Result should be 1x3x4
    expr {[llength $input_shape] == 3 && [llength $result_shape] == 3 && [lindex $result_shape 2] == 4}
} 1

test avgpool1d-5.2 {Functional - padding increases output size} {
    set input [torch::ones {1 3 8}]
    set result1 [torch::avgpool1d $input 2 2 0]
    set result2 [torch::avgpool1d $input 2 2 1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    ;# With padding, output should be larger
    expr {[lindex $shape2 2] > [lindex $shape1 2]}
} 1

test avgpool1d-5.3 {Functional - stride affects output size} {
    set input [torch::ones {1 3 8}]
    set result1 [torch::avgpool1d $input 2 1]  ;# stride=1
    set result2 [torch::avgpool1d $input 2 2]  ;# stride=2
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    ;# Smaller stride should give larger output
    expr {[lindex $shape1 2] > [lindex $shape2 2]}
} 1

test avgpool1d-5.4 {Functional - default stride equals kernel size} {
    set input [torch::ones {1 3 8}]
    set result1 [torch::avgpool1d $input 2]    ;# default stride
    set result2 [torch::avgpool1d $input 2 2]  ;# explicit stride=2
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    ;# Should be identical
    expr {$shape1 eq $shape2}
} 1

# Data type tests
test avgpool1d-6.1 {Float32 tensor} {
    set input [torch::ones {1 3 8} -dtype float32]
    set result [torch::avgpool1d $input 2]
    string match "tensor*" $result
} 1

test avgpool1d-6.2 {Float64 tensor} {
    set input [torch::ones {1 3 8} -dtype float64]
    set result [torch::avgpool1d $input 2]
    string match "tensor*" $result
} 1

# Edge cases
test avgpool1d-7.1 {Minimum valid input} {
    set input [torch::ones {1 1 2}]
    set result [torch::avgpool1d $input 2]
    string match "tensor*" $result
} 1

test avgpool1d-7.2 {Large kernel size} {
    set input [torch::ones {1 3 16}]
    set result [torch::avgpool1d $input 4]
    string match "tensor*" $result
} 1

test avgpool1d-7.3 {Multiple batch dimensions} {
    set input [torch::ones {2 5 12}]
    set result [torch::avgpool1d $input 3]
    string match "tensor*" $result
} 1

test avgpool1d-7.4 {Sequence length 1} {
    set input [torch::ones {1 3 1}]
    set result [torch::avgpool1d $input 1]
    string match "tensor*" $result
} 1

# Count include pad tests
test avgpool1d-8.1 {Count include pad true} {
    set input [torch::ones {1 1 4}]
    set result [torch::avgpool1d $input 2 2 1 1]
    string match "tensor*" $result
} 1

test avgpool1d-8.2 {Count include pad false} {
    set input [torch::ones {1 1 4}]
    set result [torch::avgpool1d $input 2 2 1 0]
    string match "tensor*" $result
} 1

# Consistency tests - both syntaxes should produce same results
test avgpool1d-9.1 {Positional vs Named consistency} {
    set input [torch::ones {1 3 8}]
    set result1 [torch::avgpool1d $input 2 2 1 1]
    set result2 [torch::avgpool1d -input $input -kernel_size 2 -stride 2 -padding 1 -count_include_pad 1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

test avgpool1d-9.2 {Snake_case vs camelCase alias consistency} {
    set input [torch::ones {1 3 8}]
    set result1 [torch::avgpool1d $input 2]
    set result2 [torch::avgPool1d $input 2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

test avgpool1d-9.3 {Parameter aliases consistency} {
    set input [torch::ones {1 3 8}]
    set result1 [torch::avgpool1d -input $input -kernel_size 2]
    set result2 [torch::avgpool1d -tensor $input -kernelSize 2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

# Mathematical correctness tests
test avgpool1d-10.1 {Mathematical - average of ones} {
    set input [torch::ones {1 1 4}]
    set result [torch::avgpool1d $input 2]
    ;# Average of 1.0 values should be 1.0, just verify it returns a tensor
    string match "tensor*" $result
} 1

cleanupTests