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

# Test cases for positional syntax (backward compatibility)
test rnn_relu-1.1 {Basic positional syntax functionality} {
    set result [torch::rnn_relu 10 20]
    # Check if we got a valid module handle
    expr {[string length $result] > 0}
} {1}

test rnn_relu-1.2 {Positional syntax with all parameters} {
    set result [torch::rnn_relu 64 128 3 true false 0.1 true]
    # Should return a valid module handle
    expr {[string match "rnn*" $result]}
} {1}

test rnn_relu-1.3 {Positional syntax with some optional parameters} {
    set result [torch::rnn_relu 32 64 2]
    # Should return a valid module handle
    expr {[string match "rnn*" $result]}
} {1}

test rnn_relu-1.4 {Positional syntax error handling - too few arguments} {
    catch {torch::rnn_relu 10} msg
    string match "*Usage:*" $msg
} {1}

test rnn_relu-1.5 {Positional syntax error handling - invalid input_size} {
    catch {torch::rnn_relu "invalid" 20} msg
    string match "*Invalid*" $msg
} {1}

test rnn_relu-1.6 {Positional syntax error handling - invalid hidden_size} {
    catch {torch::rnn_relu 10 "invalid"} msg
    string match "*Invalid*" $msg
} {1}

# Test cases for named parameter syntax
test rnn_relu-2.1 {Named parameter syntax basic functionality} {
    set result [torch::rnn_relu -inputSize 10 -hiddenSize 20]
    # Check if we got a valid module handle
    expr {[string length $result] > 0}
} {1}

test rnn_relu-2.2 {Named parameter syntax with all parameters} {
    set result [torch::rnn_relu -inputSize 64 -hiddenSize 128 -numLayers 3 -bias true -batchFirst false -dropout 0.1 -bidirectional true]
    # Should return a valid module handle
    expr {[string match "rnn*" $result]}
} {1}

test rnn_relu-2.3 {Named parameter syntax with some optional parameters} {
    set result [torch::rnn_relu -inputSize 32 -hiddenSize 64 -numLayers 2 -dropout 0.2]
    # Should return a valid module handle
    expr {[string match "rnn*" $result]}
} {1}

test rnn_relu-2.4 {Named parameter syntax error handling - missing required parameter} {
    catch {torch::rnn_relu -inputSize 10} msg
    string match "*Required parameters missing*" $msg
} {1}

test rnn_relu-2.5 {Named parameter syntax error handling - invalid parameter} {
    catch {torch::rnn_relu -inputSize 10 -hiddenSize 20 -invalid_param 5} msg
    string match "*Unknown parameter*" $msg
} {1}

test rnn_relu-2.6 {Named parameter syntax error handling - missing value} {
    catch {torch::rnn_relu -inputSize} msg
    string match "*Missing value*" $msg
} {1}

test rnn_relu-2.7 {Named parameter syntax error handling - invalid inputSize} {
    catch {torch::rnn_relu -inputSize "invalid" -hiddenSize 20} msg
    string match "*Invalid*" $msg
} {1}

# Test cases for camelCase alias
test rnn_relu-3.1 {CamelCase alias basic functionality} {
    set result [torch::rnnRelu 10 20]
    # Check if we got a valid module handle
    expr {[string length $result] > 0}
} {1}

test rnn_relu-3.2 {CamelCase alias with named parameters} {
    set result [torch::rnnRelu -inputSize 32 -hiddenSize 64 -numLayers 2]
    # Should return a valid module handle
    expr {[string match "rnn*" $result]}
} {1}

test rnn_relu-3.3 {CamelCase alias with all parameters positional} {
    set result [torch::rnnRelu 16 32 1 true false 0.0 false]
    # Should return a valid module handle
    expr {[string match "rnn*" $result]}
} {1}

test rnn_relu-3.4 {CamelCase alias error handling} {
    catch {torch::rnnRelu -inputSize 10 -invalidParam 20} msg
    string match "*Unknown parameter*" $msg
} {1}

# Consistency tests - both syntaxes should produce similar results
test rnn_relu-4.1 {Consistency between positional and named syntax} {
    set result1 [torch::rnn_relu 16 32 2 true false 0.1 false]
    set result2 [torch::rnn_relu -inputSize 16 -hiddenSize 32 -numLayers 2 -bias true -batchFirst false -dropout 0.1 -bidirectional false]
    # Both should return valid module handles
    expr {[string match "rnn*" $result1] && [string match "rnn*" $result2]}
} {1}

test rnn_relu-4.2 {Consistency between snake_case and camelCase} {
    set result1 [torch::rnn_relu 8 16]
    set result2 [torch::rnnRelu 8 16]
    # Both should return valid module handles
    expr {[string match "rnn*" $result1] && [string match "rnn*" $result2]}
} {1}

# Parameter validation tests
test rnn_relu-5.1 {Parameter validation - negative input_size} {
    catch {torch::rnn_relu -inputSize -10 -hiddenSize 20} msg
    string match "*Required parameters missing or invalid*" $msg
} {1}

test rnn_relu-5.2 {Parameter validation - zero hidden_size} {
    catch {torch::rnn_relu -inputSize 10 -hiddenSize 0} msg
    string match "*Required parameters missing or invalid*" $msg
} {1}

test rnn_relu-5.3 {Parameter validation - negative dropout} {
    catch {torch::rnn_relu -inputSize 10 -hiddenSize 20 -dropout -0.1} msg
    string match "*Required parameters missing or invalid*" $msg
} {1}

test rnn_relu-5.4 {Parameter validation - zero num_layers} {
    catch {torch::rnn_relu -inputSize 10 -hiddenSize 20 -numLayers 0} msg
    string match "*Required parameters missing or invalid*" $msg
} {1}

# Edge cases and boundary values
test rnn_relu-6.1 {Edge case - minimum valid values} {
    set result [torch::rnn_relu -inputSize 1 -hiddenSize 1 -numLayers 1]
    # Should work with minimum valid values
    expr {[string match "rnn*" $result]}
} {1}

test rnn_relu-6.2 {Edge case - large values} {
    set result [torch::rnn_relu -inputSize 512 -hiddenSize 1024 -numLayers 5]
    # Should work with large values
    expr {[string match "rnn*" $result]}
} {1}

test rnn_relu-6.3 {Edge case - maximum dropout} {
    set result [torch::rnn_relu -inputSize 10 -hiddenSize 20 -dropout 1.0]
    # Should work with dropout = 1.0
    expr {[string match "rnn*" $result]}
} {1}

test rnn_relu-6.4 {Edge case - all boolean options true} {
    set result [torch::rnn_relu -inputSize 10 -hiddenSize 20 -bias true -batchFirst true -bidirectional true]
    # Should work with all boolean options true
    expr {[string match "rnn*" $result]}
} {1}

test rnn_relu-6.5 {Edge case - all boolean options false} {
    set result [torch::rnn_relu -inputSize 10 -hiddenSize 20 -bias false -batchFirst false -bidirectional false]
    # Should work with all boolean options false
    expr {[string match "rnn*" $result]}
} {1}

# Mixed parameter order tests (named syntax)
test rnn_relu-7.1 {Named parameters in different order} {
    set result [torch::rnn_relu -hiddenSize 20 -inputSize 10 -dropout 0.1 -numLayers 2]
    # Should work regardless of parameter order
    expr {[string match "rnn*" $result]}
} {1}

test rnn_relu-7.2 {Named parameters scattered order} {
    set result [torch::rnn_relu -bias true -inputSize 32 -bidirectional false -hiddenSize 64 -batchFirst true]
    # Should work with scattered parameter order
    expr {[string match "rnn*" $result]}
} {1}

cleanupTests 