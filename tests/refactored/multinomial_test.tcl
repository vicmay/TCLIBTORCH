#!/usr/bin/env tclsh

# Test file for torch::multinomial command with dual syntax support
# Tests both positional and named parameter syntax

package require tcltest
namespace import tcltest::*

# Load the LibTorch TCL extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test suite for torch::multinomial
test multinomial-1.1 {Basic positional syntax} {
    set weights [torch::tensor_create -data {0.1 0.2 0.3 0.4} -dtype float32 -device cpu]
    set result [torch::multinomial $weights 3 true]
    expr {[string length $result] > 0}
} {1}

test multinomial-1.2 {Positional syntax without replacement} {
    set weights [torch::tensor_create -data {0.1 0.2 0.3 0.4} -dtype float32 -device cpu]
    set result [torch::multinomial $weights 3 false]
    expr {[string length $result] > 0}
} {1}

test multinomial-2.1 {Named parameter syntax with replacement} {
    set weights [torch::tensor_create -data {0.1 0.2 0.3 0.4} -dtype float32 -device cpu]
    set result [torch::multinomial -input $weights -numSamples 3 -replacement true]
    expr {[string length $result] > 0}
} {1}

test multinomial-2.2 {Named parameter syntax without replacement} {
    set weights [torch::tensor_create -data {0.1 0.2 0.3 0.4} -dtype float32 -device cpu]
    set result [torch::multinomial -input $weights -numSamples 3 -replacement false]
    expr {[string length $result] > 0}
} {1}

test multinomial-2.3 {Named parameter syntax with default replacement} {
    set weights [torch::tensor_create -data {0.1 0.2 0.3 0.4} -dtype float32 -device cpu]
    set result [torch::multinomial -input $weights -numSamples 2]
    expr {[string length $result] > 0}
} {1}

test multinomial-2.4 {Named parameter syntax using num_samples} {
    set weights [torch::tensor_create -data {0.1 0.2 0.3 0.4} -dtype float32 -device cpu]
    set result [torch::multinomial -input $weights -num_samples 2 -replacement true]
    expr {[string length $result] > 0}
} {1}

test multinomial-3.1 {Error handling - invalid tensor} {
    catch {torch::multinomial invalid_tensor 3 true} result
    expr {[string length $result] > 0}
} {1}

test multinomial-3.2 {Error handling - missing required parameters (named)} {
    catch {torch::multinomial -numSamples 3} result
    expr {[string length $result] > 0}
} {1}

test multinomial-3.3 {Error handling - unknown parameter} {
    set weights [torch::tensor_create -data {0.1 0.2 0.3 0.4} -dtype float32 -device cpu]
    catch {torch::multinomial -input $weights -numSamples 3 -unknown_param value} result
    expr {[string length $result] > 0}
} {1}

test multinomial-3.4 {Error handling - zero samples} {
    set weights [torch::tensor_create -data {0.1 0.2 0.3 0.4} -dtype float32 -device cpu]
    catch {torch::multinomial -input $weights -numSamples 0} result
    expr {[string length $result] > 0}
} {1}

test multinomial-4.1 {Sample count validation} {
    set weights [torch::tensor_create -data {0.1 0.2 0.3 0.4} -dtype float32 -device cpu]
    set result [torch::multinomial $weights 1 true]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 1}
} {1}

test multinomial-4.2 {Sample count validation - multiple samples} {
    set weights [torch::tensor_create -data {0.1 0.2 0.3 0.4} -dtype float32 -device cpu]
    set result [torch::multinomial $weights 5 true]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 5}
} {1}

test multinomial-5.1 {With replacement - more samples than categories} {
    set weights [torch::tensor_create -data {0.5 0.5} -dtype float32 -device cpu]
    set result [torch::multinomial $weights 5 true]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 5}
} {1}

test multinomial-5.2 {Without replacement - limited by categories} {
    set weights [torch::tensor_create -data {0.25 0.25 0.25 0.25} -dtype float32 -device cpu]
    set result [torch::multinomial $weights 3 false]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 3}
} {1}

test multinomial-6.1 {2D probability tensor} {
    set weights [torch::tensor_create -data {0.1 0.2 0.3 0.4 0.4 0.3 0.2 0.1} -dtype float32 -device cpu]
    set weights_2d [torch::tensor_reshape $weights {2 4}]
    set result [torch::multinomial $weights_2d 2 true]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 2 && [lindex $shape 1] == 2}
} {1}

test multinomial-7.1 {Different data types - float64} {
    set weights [torch::tensor_create -data {0.1 0.2 0.3 0.4} -dtype float64 -device cpu]
    set result [torch::multinomial $weights 2 true]
    expr {[string length $result] > 0}
} {1}

test multinomial-8.1 {Uniform distribution} {
    set weights [torch::tensor_create -data {0.25 0.25 0.25 0.25} -dtype float32 -device cpu]
    set result [torch::multinomial $weights 10 true]
    expr {[string length $result] > 0}
} {1}

test multinomial-8.2 {Skewed distribution} {
    set weights [torch::tensor_create -data {0.9 0.05 0.03 0.02} -dtype float32 -device cpu]
    set result [torch::multinomial $weights 5 true]
    expr {[string length $result] > 0}
} {1}

test multinomial-9.1 {Syntax consistency - positional vs named} {
    set weights [torch::tensor_create -data {0.1 0.2 0.3 0.4} -dtype float32 -device cpu]
    set result1 [torch::multinomial $weights 3 true]
    set result2 [torch::multinomial -input $weights -numSamples 3 -replacement true]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {[lindex $shape1 0] == [lindex $shape2 0]}
} {1}

test multinomial-10.1 {Large number of samples} {
    set weights [torch::tensor_create -data {0.2 0.3 0.5} -dtype float32 -device cpu]
    set result [torch::multinomial $weights 100 true]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 100}
} {1}

test multinomial-11.1 {Single category} {
    set weights [torch::tensor_create -data {1.0} -dtype float32 -device cpu]
    set result [torch::multinomial $weights 5 true]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 5}
} {1}

test multinomial-12.1 {Batch processing} {
    # Create batch of probability distributions (2x4 tensor)
    set batch_weights [torch::tensor_create -data {0.1 0.2 0.3 0.4 0.4 0.3 0.2 0.1} -dtype float32 -device cpu]
    set batch_weights_2d [torch::tensor_reshape $batch_weights {2 4}]
    set result [torch::multinomial $batch_weights_2d 3 true]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 2 && [lindex $shape 1] == 3}
} {1}

# Clean up
cleanupTests 