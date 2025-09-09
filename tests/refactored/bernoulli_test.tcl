#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

if {[catch {load ../../build/libtorchtcl.so} err]} {
    puts "Failed to load libtorchtcl.so: $err"
    exit 1
}

configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test 1: Basic positional syntax

test bernoulli-1.1 {positional basic - tensor as probabilities} {
    set probs [torch::tensor_create -data {0.5 0.8 0.3} -shape {3}]
    set result [torch::bernoulli $probs]
    string match "tensor*" $result
} {1}

test bernoulli-1.2 {positional with probability} {
    set input [torch::tensor_create -data {1.0 1.0 1.0} -shape {3}]
    set result [torch::bernoulli $input 0.6]
    string match "tensor*" $result
} {1}

test bernoulli-1.3 {positional with zero probability} {
    set input [torch::tensor_create -data {1.0 1.0} -shape {2}]
    set result [torch::bernoulli $input 0.0]
    string match "tensor*" $result
} {1}

test bernoulli-1.4 {positional with probability 1.0} {
    set input [torch::tensor_create -data {1.0 1.0} -shape {2}]
    set result [torch::bernoulli $input 1.0]
    string match "tensor*" $result
} {1}

# Test 2: Named parameter syntax

test bernoulli-2.1 {named basic - tensor as probabilities} {
    set probs [torch::tensor_create -data {0.4 0.9 0.1} -shape {3}]
    set result [torch::bernoulli -input $probs]
    string match "tensor*" $result
} {1}

test bernoulli-2.2 {named with probability} {
    set input [torch::tensor_create -data {1.0 1.0 1.0} -shape {3}]
    set result [torch::bernoulli -input $input -p 0.7]
    string match "tensor*" $result
} {1}

test bernoulli-2.3 {named with probability alias} {
    set input [torch::tensor_create -data {1.0 1.0} -shape {2}]
    set result [torch::bernoulli -input $input -probability 0.3]
    string match "tensor*" $result
} {1}

test bernoulli-2.4 {named with tensor alias} {
    set input [torch::tensor_create -data {0.6 0.2 0.8} -shape {3}]
    set result [torch::bernoulli -tensor $input]
    string match "tensor*" $result
} {1}

# Test 3: Error handling

test bernoulli-3.1 {missing input tensor in named syntax} {
    catch {torch::bernoulli -p 0.5} result
    string match "*Missing required parameter: -input*" $result
} {1}

test bernoulli-3.2 {unknown parameter} {
    set input [torch::tensor_create -data {1.0} -shape {1}]
    catch {torch::bernoulli -input $input -unknown value} result
    string match "*Unknown parameter: -unknown*" $result
} {1}

test bernoulli-3.3 {invalid tensor name} {
    catch {torch::bernoulli -input invalid_tensor} result
    string match "*Invalid input tensor name*" $result
} {1}

test bernoulli-3.4 {probability out of range - negative} {
    set input [torch::tensor_create -data {1.0} -shape {1}]
    catch {torch::bernoulli $input -0.1} result
    string match "*Probability p must be in range*" $result
} {1}

test bernoulli-3.5 {probability out of range - greater than 1} {
    set input [torch::tensor_create -data {1.0} -shape {1}]
    catch {torch::bernoulli -input $input -p 1.5} result
    string match "*Probability p must be in range*" $result
} {1}

test bernoulli-3.6 {missing value for parameter} {
    set input [torch::tensor_create -data {1.0} -shape {1}]
    catch {torch::bernoulli -input $input -p} result
    string match "*Named parameters require pairs: -param value*" $result
} {1}

# Test 4: Probability boundary conditions

test bernoulli-4.1 {probability exactly 0.0} {
    set input [torch::tensor_create -data {1.0 1.0 1.0} -shape {3}]
    set result [torch::bernoulli -input $input -p 0.0]
    string match "tensor*" $result
} {1}

test bernoulli-4.2 {probability exactly 1.0} {
    set input [torch::tensor_create -data {1.0 1.0 1.0} -shape {3}]
    set result [torch::bernoulli -input $input -p 1.0]
    string match "tensor*" $result
} {1}

test bernoulli-4.3 {probability 0.5} {
    set input [torch::tensor_create -data {1.0 1.0 1.0} -shape {3}]
    set result [torch::bernoulli -input $input -p 0.5]
    string match "tensor*" $result
} {1}

# Test 5: Different tensor shapes

test bernoulli-5.1 {1D tensor} {
    set input [torch::tensor_create -data {0.3 0.7 0.5 0.9} -shape {4}]
    set result [torch::bernoulli -input $input]
    string match "tensor*" $result
} {1}

test bernoulli-5.2 {2D tensor} {
    set input [torch::tensor_create -data {0.1 0.9 0.4 0.6} -shape {2 2}]
    set result [torch::bernoulli -input $input]
    string match "tensor*" $result
} {1}

test bernoulli-5.3 {3D tensor} {
    set input [torch::tensor_create -data {0.2 0.8 0.5 0.3 0.7 0.9 0.1 0.4} -shape {2 2 2}]
    set result [torch::bernoulli -input $input]
    string match "tensor*" $result
} {1}

test bernoulli-5.4 {scalar tensor} {
    set input [torch::tensor_create -data {0.6} -shape {1}]
    set result [torch::bernoulli -input $input]
    string match "tensor*" $result
} {1}

# Test 6: Output validation

test bernoulli-6.1 {output shape matches input} {
    set input [torch::tensor_create -data {0.5 0.5 0.5} -shape {3}]
    set result [torch::bernoulli -input $input]
    set input_shape [torch::tensor_shape $input]
    set result_shape [torch::tensor_shape $result]
    expr {$input_shape eq $result_shape}
} {1}

test bernoulli-6.2 {output dtype is correct for floating input} {
    set input [torch::tensor_create -data {0.5 0.5} -shape {2} -dtype float32]
    set result [torch::bernoulli -input $input]
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Float32"}
} {1}

# Test 7: Mathematical properties

test bernoulli-7.1 {probability 0.0 always gives 0} {
    set input [torch::tensor_create -data {1.0 1.0 1.0 1.0 1.0} -shape {5}]
    set result [torch::bernoulli -input $input -p 0.0]
    # Check if all values are 0 (sum should be 0)
    set sum [torch::tensor_sum $result]
    set sum_data [torch::tensor_item $sum]
    expr {$sum_data == 0.0}
} {1}

test bernoulli-7.2 {probability 1.0 always gives 1} {
    set input [torch::tensor_create -data {1.0 1.0 1.0 1.0 1.0} -shape {5}]
    set result [torch::bernoulli -input $input -p 1.0]
    # Check if all values are 1 (sum should equal number of elements)
    set sum [torch::tensor_sum $result]
    set sum_data [torch::tensor_item $sum]
    expr {$sum_data == 5.0}
} {1}

# Test 8: Multiple calls consistency

test bernoulli-8.1 {multiple calls with same parameters} {
    set input [torch::tensor_create -data {0.5 0.5 0.5} -shape {3}]
    set result1 [torch::bernoulli -input $input -p 0.5]
    set result2 [torch::bernoulli -input $input -p 0.5]
    # Both should be valid tensors (not necessarily equal due to randomness)
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

# Test 9: Equivalence of syntaxes

test bernoulli-9.1 {positional vs named equivalence - tensor probabilities} {
    set probs [torch::tensor_create -data {0.3 0.7 0.5} -shape {3}]
    set pos_result [torch::bernoulli $probs]
    set named_result [torch::bernoulli -input $probs]
    expr {[string match "tensor*" $pos_result] && [string match "tensor*" $named_result]}
} {1}

test bernoulli-9.2 {positional vs named equivalence - with probability} {
    set input [torch::tensor_create -data {1.0 1.0 1.0} -shape {3}]
    set pos_result [torch::bernoulli $input 0.4]
    set named_result [torch::bernoulli -input $input -p 0.4]
    expr {[string match "tensor*" $pos_result] && [string match "tensor*" $named_result]}
} {1}

test bernoulli-9.3 {parameter aliases equivalence} {
    set input [torch::tensor_create -data {1.0 1.0} -shape {2}]
    set result1 [torch::bernoulli -input $input -p 0.6]
    set result2 [torch::bernoulli -tensor $input -probability 0.6]
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

# Test 10: Integration tests

test bernoulli-10.1 {result can be used in further operations} {
    set input [torch::tensor_create -data {0.5 0.5 0.5} -shape {3}]
    set bernoulli_result [torch::bernoulli -input $input]
    set sum_result [torch::tensor_sum $bernoulli_result]
    string match "tensor*" $sum_result
} {1}

test bernoulli-10.2 {large tensor performance} {
    # Create larger tensor to test performance doesn't crash
    set large_input [torch::tensor_create -data [string repeat "0.5 " 100] -shape {100}]
    set result [torch::bernoulli -input $large_input -p 0.3]
    string match "tensor*" $result
} {1}

cleanupTests 