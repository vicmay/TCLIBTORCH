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

# Create test tensors
set query_data {1.0 2.0 3.0 4.0}
set key_data {1.0 0.0 0.0 1.0}
set value_data {1.0 0.0 0.0 1.0}

set query [torch::tensor_create -data $query_data -dtype float32]
set query [torch::tensor_reshape $query {2 2}]
set key [torch::tensor_create -data $key_data -dtype float32]
set key [torch::tensor_reshape $key {2 2}]
set value [torch::tensor_create -data $value_data -dtype float32]
set value [torch::tensor_reshape $value {2 2}]

# Test cases for positional syntax
test scaled_dot_product_attention-1.1 {Basic positional syntax} {
    set result [torch::scaled_dot_product_attention $query $key $value]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 2 && [lindex $shape 1] == 2}
} {1}

test scaled_dot_product_attention-1.2 {Positional syntax - verify output} {
    set result [torch::scaled_dot_product_attention $query $key $value]
    set shape [torch::tensor_shape $result]
    expr {[llength $shape] == 2}
} {1}

# Test cases for named parameter syntax
test scaled_dot_product_attention-2.1 {Named parameter syntax - with -query} {
    set result [torch::scaled_dot_product_attention -query $query -key $key -value $value]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 2 && [lindex $shape 1] == 2}
} {1}

test scaled_dot_product_attention-2.2 {Named parameter syntax - verify output} {
    set result [torch::scaled_dot_product_attention -query $query -key $key -value $value]
    set shape [torch::tensor_shape $result]
    expr {[llength $shape] == 2}
} {1}

# Test cases for camelCase alias
test scaled_dot_product_attention-3.1 {CamelCase alias - with -query} {
    set result [torch::scaledDotProductAttention -query $query -key $key -value $value]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 2 && [lindex $shape 1] == 2}
} {1}

test scaled_dot_product_attention-3.2 {CamelCase alias - verify output} {
    set result [torch::scaledDotProductAttention -query $query -key $key -value $value]
    set shape [torch::tensor_shape $result]
    expr {[llength $shape] == 2}
} {1}

# Error handling tests
test scaled_dot_product_attention-4.1 {Error - no arguments} {
    catch {torch::scaled_dot_product_attention} err
    set err
} {wrong # args: should be "torch::scaled_dot_product_attention query key value"}

test scaled_dot_product_attention-4.2 {Error - too few arguments} {
    catch {torch::scaled_dot_product_attention $query} err
    set err
} {wrong # args: should be "torch::scaled_dot_product_attention query key value"}

test scaled_dot_product_attention-4.3 {Error - too many arguments} {
    catch {torch::scaled_dot_product_attention $query $key $value extra} err
    set err
} {wrong # args: should be "torch::scaled_dot_product_attention query key value"}

test scaled_dot_product_attention-4.4 {Error - invalid query} {
    catch {torch::scaled_dot_product_attention invalid_tensor $key $value} err
    set err
} {Error in scaled_dot_product_attention: Invalid tensor}

test scaled_dot_product_attention-4.5 {Error - missing value for parameter} {
    catch {torch::scaled_dot_product_attention -query} err
    set err
} {Error in scaled_dot_product_attention: Missing value for parameter}

test scaled_dot_product_attention-4.6 {Error - unknown parameter} {
    catch {torch::scaled_dot_product_attention -invalid $query} err
    set err
} {Error in scaled_dot_product_attention: Unknown parameter: -invalid}

test scaled_dot_product_attention-4.7 {Error - missing required parameter} {
    catch {torch::scaled_dot_product_attention -query $query -key $key} err
    set err
} {Error in scaled_dot_product_attention: Required parameters missing: query, key, and value}

cleanupTests 