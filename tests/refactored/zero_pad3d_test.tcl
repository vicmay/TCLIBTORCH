#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# Helper to create a 2x2x2 tensor with known values
proc make3d {} {
    # Create a 2x2x2 tensor using zeros - simpler approach
    return [torch::zeros {2 2 2} float32 cpu false]
}

;# Test cases for positional syntax
test zero-pad3d-1.1 {Basic positional syntax} {
    set tensor [make3d]
    set result [torch::zero_pad3d $tensor {1 1 1 1 1 1}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{4 4 4}}

test zero-pad3d-1.2 {Positional syntax with different padding} {
    set tensor [make3d]
    set result [torch::zero_pad3d $tensor {2 0 1 3 0 2}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{4 6 4}}

test zero-pad3d-1.3 {Positional syntax with zero padding} {
    set tensor [make3d]
    set result [torch::zero_pad3d $tensor {0 0 0 0 0 0}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{2 2 2}}

;# Test cases for named parameter syntax
test zero-pad3d-2.1 {Named parameter syntax with -input and -padding} {
    set tensor [make3d]
    set result [torch::zero_pad3d -input $tensor -padding {1 1 1 1 1 1}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{4 4 4}}

test zero-pad3d-2.2 {Named parameter syntax with -tensor and -pad} {
    set tensor [make3d]
    set result [torch::zero_pad3d -tensor $tensor -pad {2 0 1 3 0 2}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{4 6 4}}

test zero-pad3d-2.3 {Named parameter syntax with asymmetric padding} {
    set tensor [make3d]
    set result [torch::zero_pad3d -input $tensor -padding {3 1 2 0 1 2}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{5 4 6}}

;# Test cases for camelCase alias
test zero-pad3d-3.1 {CamelCase alias with positional syntax} {
    set tensor [make3d]
    set result [torch::zeroPad3d $tensor {1 1 1 1 1 1}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{4 4 4}}

test zero-pad3d-3.2 {CamelCase alias with named parameters} {
    set tensor [make3d]
    set result [torch::zeroPad3d -input $tensor -padding {2 0 1 3 0 2}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{4 6 4}}

;# Error handling tests
test zero-pad3d-4.1 {Error: insufficient arguments} {
    catch {torch::zero_pad3d} result
    string match *Usage:* $result
} {1}

test zero-pad3d-4.2 {Error: wrong number of positional arguments} {
    set tensor [make3d]
    catch {torch::zero_pad3d $tensor} result
    string match *Usage:* $result
} {1}

test zero-pad3d-4.3 {Error: invalid padding list length} {
    set tensor [make3d]
    catch {torch::zero_pad3d $tensor {1 2 3}} result
    expr {[string match "*Padding must be a list of 6 values for 3D*" $result]}
} {1}

test zero-pad3d-4.4 {Error: unknown named parameter} {
    set tensor [make3d]
    catch {torch::zero_pad3d -unknown $tensor -padding {1 1 1 1 1 1}} result
    expr {[string match "*Unknown parameter:*" $result]}
} {1}

test zero-pad3d-4.5 {Error: missing value for parameter} {
    set tensor [make3d]
    catch {torch::zero_pad3d -input $tensor -padding} result
    expr {[string match "*Missing value for parameter*" $result]}
} {1}

;# Edge cases
test zero-pad3d-5.1 {Large padding values} {
    set tensor [make3d]
    set result [torch::zero_pad3d $tensor {10 10 10 10 10 10}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{22 22 22}}

test zero-pad3d-5.2 {Single element tensor works correctly} {
    set tensor [torch::zeros {1 1 1} float32 cpu false]
    set result [torch::zero_pad3d $tensor {1 1 1 1 1 1}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{3 3 3}}

;# Verify that both syntaxes produce the same result
test zero-pad3d-6.1 {Positional and named syntax produce same result} {
    set tensor [make3d]
    set result1 [torch::zero_pad3d $tensor {1 1 1 1 1 1}]
    set result2 [torch::zero_pad3d -input $tensor -padding {1 1 1 1 1 1}]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    list $shape1 $shape2
} {{4 4 4} {4 4 4}}

test zero-pad3d-6.2 {CamelCase and snake_case produce same result} {
    set tensor [make3d]
    set result1 [torch::zero_pad3d $tensor {1 1 1 1 1 1}]
    set result2 [torch::zeroPad3d $tensor {1 1 1 1 1 1}]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    list $shape1 $shape2
} {{4 4 4} {4 4 4}}

cleanupTests 