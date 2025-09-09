#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Configure test output
configure -verbose {pass fail skip error}

# Load the extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Helper: compare tensors with tolerance
proc tensors_equal {tensor1 tensor2 {tolerance 1e-5}} {
    set diff [torch::tensor_sub $tensor1 $tensor2]
    set abs_diff [torch::tensor_abs $diff]
    set max_diff [torch::tensor_max $abs_diff]
    set max_val [torch::tensor_item $max_diff]
    return [expr {$max_val < $tolerance}]
}

# -----------------------------------------------------------------------------
# 1. Positional syntax tests
# -----------------------------------------------------------------------------

# Basic positional
test conv_transpose2d-1.1 {Basic positional syntax} {
    set input [torch::randn -shape {1 1 5 5}]
    set weight [torch::randn -shape {1 1 3 3}]
    set result [torch::conv_transpose_2d $input $weight]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Positional with bias
test conv_transpose2d-1.2 {Positional syntax with bias} {
    set input  [torch::randn -shape {1 1 4 4}]
    set weight [torch::randn -shape {1 1 3 3}]
    set bias   [torch::randn -shape {1}]
    set result [torch::conv_transpose_2d $input $weight $bias]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Positional with stride
test conv_transpose2d-1.3 {Positional syntax with stride} {
    set input  [torch::randn -shape {1 1 4 4}]
    set weight [torch::randn -shape {1 1 3 3}]
    set result [torch::conv_transpose_2d $input $weight none {2 2}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Positional with all params
test conv_transpose2d-1.4 {Positional syntax with all params} {
    set input  [torch::randn -shape {1 1 4 4}]
    set weight [torch::randn -shape {1 1 3 3}]
    set bias   [torch::randn -shape {1}]
    set result [torch::conv_transpose_2d $input $weight $bias {2 2} {1 1} {0 0} 1 {1 1}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# -----------------------------------------------------------------------------
# 2. Named parameter syntax tests
# -----------------------------------------------------------------------------
# Basic named
test conv_transpose2d-2.1 {Named syntax required params} {
    set input  [torch::randn -shape {1 1 5 5}]
    set weight [torch::randn -shape {1 1 3 3}]
    set result [torch::conv_transpose_2d -input $input -weight $weight]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Named with bias
test conv_transpose2d-2.2 {Named syntax with bias} {
    set input  [torch::randn -shape {1 1 4 4}]
    set weight [torch::randn -shape {1 1 3 3}]
    set bias   [torch::randn -shape {1}]
    set result [torch::conv_transpose_2d -input $input -weight $weight -bias $bias]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Named with stride
test conv_transpose2d-2.3 {Named syntax with stride} {
    set input  [torch::randn -shape {1 1 4 4}]
    set weight [torch::randn -shape {1 1 3 3}]
    set result [torch::conv_transpose_2d -input $input -weight $weight -stride {2 2}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Named with all params
test conv_transpose2d-2.4 {Named syntax full params} {
    set input  [torch::randn -shape {1 1 4 4}]
    set weight [torch::randn -shape {1 1 3 3}]
    set bias   [torch::randn -shape {1}]
    set result [torch::conv_transpose_2d -input $input -weight $weight -bias $bias -stride {2 2} -padding {1 1} -output_padding {0 0} -groups 1 -dilation {1 1}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Named with mixed order
test conv_transpose2d-2.5 {Named syntax mixed order} {
    set input  [torch::randn -shape {1 1 4 4}]
    set weight [torch::randn -shape {1 1 3 3}]
    set bias   [torch::randn -shape {1}]
    set result [torch::conv_transpose_2d -stride {1 1} -input $input -bias $bias -weight $weight]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# -----------------------------------------------------------------------------
# 3. CamelCase alias tests
# -----------------------------------------------------------------------------

# Basic alias
test conv_transpose2d-3.1 {CamelCase alias basic} {
    set input  [torch::randn -shape {1 1 5 5}]
    set weight [torch::randn -shape {1 1 3 3}]
    set result [torch::convTranspose2d $input $weight]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# CamelCase alias with named params
test conv_transpose2d-3.2 {CamelCase alias named params} {
    set input  [torch::randn -shape {1 1 4 4}]
    set weight [torch::randn -shape {1 1 3 3}]
    set bias   [torch::randn -shape {1}]
    set result [torch::convTranspose2d -input $input -weight $weight -bias $bias -stride {2 2}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Syntax consistency check
test conv_transpose2d-3.3 {Syntax consistency positional vs named vs alias} {
    set input  [torch::ones -shape {1 1 4 4}]
    set weight [torch::ones -shape {1 1 3 3}]
    set bias   [torch::zeros -shape {1}]

    set r1 [torch::conv_transpose_2d $input $weight $bias {2 2} {1 1} {0 0} 1 {1 1}]
    set r2 [torch::conv_transpose_2d -input $input -weight $weight -bias $bias -stride {2 2} -padding {1 1} -output_padding {0 0} -groups 1 -dilation {1 1}]
    set r3 [torch::convTranspose2d -input $input -weight $weight -bias $bias -stride {2 2} -padding {1 1} -output_padding {0 0} -groups 1 -dilation {1 1}]

    list [tensors_equal $r1 $r2] [tensors_equal $r1 $r3]
} {1 1}

# -----------------------------------------------------------------------------
# 4. Error handling tests
# -----------------------------------------------------------------------------

# Missing input
test conv_transpose2d-4.1 {Error: missing input} {
    set weight [torch::randn -shape {1 1 3 3}]
    set code [catch {torch::conv_transpose_2d -weight $weight} msg]
    list $code [string match "*input*" $msg]
} {1 1}

# Missing weight
test conv_transpose2d-4.2 {Error: missing weight} {
    set input [torch::randn -shape {1 1 4 4}]
    set code [catch {torch::conv_transpose_2d -input $input} msg]
    list $code [string match "*weight*" $msg]
} {1 1}

# Invalid input tensor name
test conv_transpose2d-4.3 {Error: invalid input tensor name} {
    set weight [torch::randn -shape {1 1 3 3}]
    set code [catch {torch::conv_transpose_2d invalid_tensor $weight} msg]
    list $code [string match "*Invalid*tensor*" $msg]
} {1 1}

# Invalid bias tensor name
test conv_transpose2d-4.4 {Error: invalid bias tensor} {
    set input  [torch::randn -shape {1 1 4 4}]
    set weight [torch::randn -shape {1 1 3 3}]
    set code [catch {torch::conv_transpose_2d -input $input -weight $weight -bias invalid_bias} msg]
    list $code [string match "*Invalid*bias*" $msg]
} {1 1}

# -----------------------------------------------------------------------------
# 5. Data type & edge cases
# -----------------------------------------------------------------------------

# Different dtype
test conv_transpose2d-5.1 {Float32 dtype support} {
    set input  [torch::randn -shape {1 1 4 4} -dtype float32]
    set weight [torch::randn -shape {1 1 3 3} -dtype float32]
    set result [torch::conv_transpose_2d -input $input -weight $weight]
    set dtype [torch::tensor_dtype $result]
    string match *Float* $dtype
} {1}

# Zero padding edge case
test conv_transpose2d-5.2 {Zero padding edge case} {
    set input  [torch::randn -shape {1 1 2 2}]
    set weight [torch::randn -shape {1 1 1 1}]
    set result [torch::conv_transpose_2d -input $input -weight $weight -padding {0 0}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Large stride edge case
test conv_transpose2d-5.3 {Large stride edge case} {
    set input  [torch::randn -shape {1 1 2 2}]
    set weight [torch::randn -shape {1 1 1 1}]
    set result [torch::conv_transpose_2d -input $input -weight $weight -stride {3 3}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Confirm output dtype matches input dtype
test conv_transpose2d-5.4 {Output dtype matches input} {
    set input  [torch::randn -shape {1 1 2 2} -dtype float32]
    set weight [torch::randn -shape {1 1 1 1} -dtype float32]
    set result [torch::conv_transpose_2d -input $input -weight $weight]
    set dtype [torch::tensor_dtype $result]
    string match *Float* $dtype
} {1}

# -----------------------------------------------------------------------------
cleanupTests 