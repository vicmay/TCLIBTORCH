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

;# Test cases for positional syntax (backward compatibility)
test zero-pad2d-1.1 {Basic positional syntax} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu false]
    set result [torch::zero_pad2d $tensor {1 1 1 1}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{4 4}}

test zero-pad2d-1.2 {Positional syntax with different padding} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu false]
    set result [torch::zero_pad2d $tensor {2 0 1 3}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{6 4}}

test zero-pad2d-1.3 {Positional syntax with zero padding} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu false]
    set result [torch::zero_pad2d $tensor {0 0 0 0}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{2 2}}

;# Test cases for named parameter syntax
test zero-pad2d-2.1 {Named parameter syntax with -input and -padding} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu false]
    set result [torch::zero_pad2d -input $tensor -padding {1 1 1 1}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{4 4}}

test zero-pad2d-2.2 {Named parameter syntax with -tensor and -pad} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu false]
    set result [torch::zero_pad2d -tensor $tensor -pad {2 0 1 3}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{6 4}}

test zero-pad2d-2.3 {Named parameter syntax with asymmetric padding} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu false]
    set result [torch::zero_pad2d -input $tensor -padding {3 1 2 0}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{4 6}}

;# Test cases for camelCase alias
test zero-pad2d-3.1 {CamelCase alias with positional syntax} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu false]
    set result [torch::zeroPad2d $tensor {1 1 1 1}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{4 4}}

test zero-pad2d-3.2 {CamelCase alias with named parameters} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu false]
    set result [torch::zeroPad2d -input $tensor -padding {2 0 1 3}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{6 4}}

;# Error handling tests
test zero-pad2d-4.1 {Error: insufficient arguments} {
    catch {torch::zero_pad2d} result
    string match *Usage:* $result
} {1}

test zero-pad2d-4.2 {Error: wrong number of positional arguments} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu false]
    catch {torch::zero_pad2d $tensor} result
    string match *Usage:* $result
} {1}

test zero-pad2d-4.3 {Error: invalid padding list length} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu false]
    catch {torch::zero_pad2d $tensor {1 2 3}} result
    expr {[string match "*Padding must be a list of 4 values for 2D*" $result]}
} {1}

test zero-pad2d-4.4 {Error: unknown named parameter} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu false]
    catch {torch::zero_pad2d -unknown $tensor -padding {1 1 1 1}} result
    expr {[string match "*Unknown parameter:*" $result]}
} {1}

test zero-pad2d-4.5 {Error: missing value for parameter} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu false]
    catch {torch::zero_pad2d -input $tensor -padding} result
    expr {[string match "*Missing value for parameter*" $result]}
} {1}

;# Edge cases
test zero-pad2d-5.1 {Large padding values} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu false]
    set result [torch::zero_pad2d $tensor {10 10 10 10}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{22 22}}

# Single element tensor test - works correctly with proper 2D tensor creation
test zero-pad2d-5.2 {Single element 2D tensor works correctly} {
    set tensor [torch::zeros {1 1} float32 cpu false]
    set result [torch::zero_pad2d $tensor {1 1 1 1}]
    set shape [torch::tensor_shape $result]
    list $shape
} {{3 3}}

;# Verify that both syntaxes produce the same result
test zero-pad2d-6.1 {Positional and named syntax produce same result} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu false]
    set result1 [torch::zero_pad2d $tensor {1 1 1 1}]
    set result2 [torch::zero_pad2d -input $tensor -padding {1 1 1 1}]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    list $shape1 $shape2
} {{4 4} {4 4}}

test zero-pad2d-6.2 {CamelCase and snake_case produce same result} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu false]
    set result1 [torch::zero_pad2d $tensor {1 1 1 1}]
    set result2 [torch::zeroPad2d $tensor {1 1 1 1}]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    list $shape1 $shape2
} {{4 4} {4 4}}

cleanupTests 