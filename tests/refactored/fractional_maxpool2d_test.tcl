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
test fractional_maxpool2d-1.1 {Basic positional syntax} {
    set input [torch::randn -shape {1 1 8 8}]
    set result [torch::fractional_maxpool2d $input {2 2}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Positional with output_ratio
test fractional_maxpool2d-1.2 {Positional syntax with output_ratio} {
    set input [torch::randn -shape {1 1 8 8}]
    set result [torch::fractional_maxpool2d $input {2 2} {0.6 0.6}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Positional with different kernel sizes
test fractional_maxpool2d-1.3 {Positional syntax with different kernel sizes} {
    set input [torch::randn -shape {1 1 12 12}]
    set result [torch::fractional_maxpool2d $input {3 3}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Positional with asymmetric kernel
test fractional_maxpool2d-1.4 {Positional syntax with asymmetric kernel} {
    set input [torch::randn -shape {1 1 10 8}]
    set result [torch::fractional_maxpool2d $input {2 3} {0.7 0.5}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# -----------------------------------------------------------------------------
# 2. Named parameter syntax tests
# -----------------------------------------------------------------------------

# Basic named
test fractional_maxpool2d-2.1 {Named syntax required params} {
    set input [torch::randn -shape {1 1 8 8}]
    set result [torch::fractional_maxpool2d -input $input -kernel_size {2 2}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Named with camelCase parameter
test fractional_maxpool2d-2.2 {Named syntax with camelCase kernelSize} {
    set input [torch::randn -shape {1 1 8 8}]
    set result [torch::fractional_maxpool2d -input $input -kernelSize {2 2}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Named with output_ratio
test fractional_maxpool2d-2.3 {Named syntax with output_ratio} {
    set input [torch::randn -shape {1 1 8 8}]
    set result [torch::fractional_maxpool2d -input $input -kernel_size {2 2} -output_ratio {0.6 0.6}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Named with camelCase output_ratio
test fractional_maxpool2d-2.4 {Named syntax with camelCase outputRatio} {
    set input [torch::randn -shape {1 1 8 8}]
    set result [torch::fractional_maxpool2d -input $input -kernel_size {2 2} -outputRatio {0.4 0.4}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Named with mixed parameter order
test fractional_maxpool2d-2.5 {Named syntax mixed parameter order} {
    set input [torch::randn -shape {1 1 8 8}]
    set result [torch::fractional_maxpool2d -output_ratio {0.3 0.3} -input $input -kernel_size {2 2}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# -----------------------------------------------------------------------------
# 3. CamelCase alias tests
# -----------------------------------------------------------------------------

# Basic alias
test fractional_maxpool2d-3.1 {CamelCase alias basic} {
    set input [torch::randn -shape {1 1 8 8}]
    set result [torch::fractionalMaxpool2d $input {2 2}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# CamelCase alias with named params
test fractional_maxpool2d-3.2 {CamelCase alias with named params} {
    set input [torch::randn -shape {1 1 8 8}]
    set result [torch::fractionalMaxpool2d -input $input -kernel_size {2 2} -output_ratio {0.5 0.5}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Syntax consistency check
test fractional_maxpool2d-3.3 {Syntax consistency positional vs named vs alias} {
    set input [torch::ones -shape {1 1 8 8}]
    
    set r1 [torch::fractional_maxpool2d $input {2 2} {0.5 0.5}]
    set r2 [torch::fractional_maxpool2d -input $input -kernel_size {2 2} -output_ratio {0.5 0.5}]
    set r3 [torch::fractionalMaxpool2d -input $input -kernel_size {2 2} -output_ratio {0.5 0.5}]
    
    # Check that all produce results with same shape
    set s1 [torch::tensor_shape $r1]
    set s2 [torch::tensor_shape $r2]
    set s3 [torch::tensor_shape $r3]
    
    expr {$s1 == $s2 && $s2 == $s3}
} {1}

# -----------------------------------------------------------------------------
# 4. Error handling tests
# -----------------------------------------------------------------------------

# Missing input
test fractional_maxpool2d-4.1 {Error: missing input} {
    set code [catch {torch::fractional_maxpool2d -kernel_size {2 2}} msg]
    list $code [string match "*input*" $msg]
} {1 1}

# Missing kernel_size
test fractional_maxpool2d-4.2 {Error: missing kernel_size} {
    set input [torch::randn -shape {1 1 8 8}]
    set code [catch {torch::fractional_maxpool2d -input $input} msg]
    list $code [string match "*kernel_size*" $msg]
} {1 1}

# Invalid input tensor name
test fractional_maxpool2d-4.3 {Error: invalid input tensor name} {
    set code [catch {torch::fractional_maxpool2d invalid_tensor {2 2}} msg]
    list $code [string match "*Invalid*tensor*" $msg]
} {1 1}

# Invalid kernel_size format
test fractional_maxpool2d-4.4 {Error: invalid kernel_size format} {
    set input [torch::randn -shape {1 1 8 8}]
    set code [catch {torch::fractional_maxpool2d -input $input -kernel_size {2 2 2}} msg]
    list $code [string match "*2 integers*" $msg]
} {1 1}

# Invalid output_ratio format
test fractional_maxpool2d-4.5 {Error: invalid output_ratio format} {
    set input [torch::randn -shape {1 1 8 8}]
    set code [catch {torch::fractional_maxpool2d -input $input -kernel_size {2 2} -output_ratio {0.5}} msg]
    list $code [string match "*2 doubles*" $msg]
} {1 1}

# Negative kernel_size
test fractional_maxpool2d-4.6 {Error: negative kernel_size} {
    set input [torch::randn -shape {1 1 8 8}]
    set code [catch {torch::fractional_maxpool2d -input $input -kernel_size {-2 2}} msg]
    list $code [string match "*invalid*" $msg]
} {1 1}

# Zero output_ratio
test fractional_maxpool2d-4.7 {Error: zero output_ratio} {
    set input [torch::randn -shape {1 1 8 8}]
    set code [catch {torch::fractional_maxpool2d -input $input -kernel_size {2 2} -output_ratio {0.0 0.5}} msg]
    list $code [string match "*invalid*" $msg]
} {1 1}

# Unknown parameter
test fractional_maxpool2d-4.8 {Error: unknown parameter} {
    set input [torch::randn -shape {1 1 8 8}]
    set code [catch {torch::fractional_maxpool2d -input $input -kernel_size {2 2} -invalid_param 1} msg]
    list $code [string match "*Unknown parameter*" $msg]
} {1 1}

# -----------------------------------------------------------------------------
# 5. Data type & edge cases
# -----------------------------------------------------------------------------

# Different dtype
test fractional_maxpool2d-5.1 {Different dtype support} {
    set input [torch::randn -shape {1 1 8 8} -dtype float32]
    set result [torch::fractional_maxpool2d -input $input -kernel_size {2 2}]
    set dtype [torch::tensor_dtype $result]
    string match *Float* $dtype
} {1}

# Large kernel size
test fractional_maxpool2d-5.2 {Large kernel size} {
    set input [torch::randn -shape {1 1 16 16}]
    set result [torch::fractional_maxpool2d -input $input -kernel_size {4 4}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Small output ratio
test fractional_maxpool2d-5.3 {Small output ratio} {
    set input [torch::randn -shape {1 1 16 16}]
    set result [torch::fractional_maxpool2d -input $input -kernel_size {2 2} -output_ratio {0.25 0.25}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Asymmetric output ratio
test fractional_maxpool2d-5.4 {Asymmetric output ratio} {
    set input [torch::randn -shape {1 1 12 16}]
    set result [torch::fractional_maxpool2d -input $input -kernel_size {2 2} -output_ratio {0.3 0.7}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# Batch processing
test fractional_maxpool2d-5.5 {Batch processing} {
    set input [torch::randn -shape {3 2 8 8}]
    set result [torch::fractional_maxpool2d -input $input -kernel_size {2 2}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] [lindex $shape 0] [lindex $shape 1]
} {4 3 2}

# Different kernel sizes
test fractional_maxpool2d-5.6 {Asymmetric kernel size} {
    set input [torch::randn -shape {1 1 12 8}]
    set result [torch::fractional_maxpool2d -input $input -kernel_size {3 2}]
    set shape [torch::tensor_shape $result]
    llength $shape
} {4}

# -----------------------------------------------------------------------------
cleanupTests 