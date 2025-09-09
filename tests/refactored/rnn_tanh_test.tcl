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

# ============================================================================
# Basic RNN Tanh Tests - Positional Syntax
# ============================================================================

test rnn_tanh-1.1 {Basic RNN tanh creation - positional syntax} {
    set handle [torch::rnn_tanh 10 20]
    string match "rnn*" $handle
} {1}

test rnn_tanh-1.2 {RNN tanh with num_layers - positional syntax} {
    set handle [torch::rnn_tanh 10 20 3]
    string match "rnn*" $handle
} {1}

test rnn_tanh-1.3 {RNN tanh with all parameters - positional syntax} {
    set handle [torch::rnn_tanh 10 20 2 true false 0.1 false]
    string match "rnn*" $handle
} {1}

test rnn_tanh-1.4 {RNN tanh bidirectional - positional syntax} {
    set handle [torch::rnn_tanh 5 10 1 true false 0.0 true]
    string match "rnn*" $handle
} {1}

test rnn_tanh-1.5 {RNN tanh with dropout - positional syntax} {
    set handle [torch::rnn_tanh 8 16 3 true true 0.5]
    string match "rnn*" $handle
} {1}

# ============================================================================
# RNN Tanh Tests - Named Parameter Syntax
# ============================================================================

test rnn_tanh-2.1 {Basic RNN tanh creation - named parameter syntax} {
    set handle [torch::rnn_tanh -inputSize 10 -hiddenSize 20]
    string match "rnn*" $handle
} {1}

test rnn_tanh-2.2 {RNN tanh with num_layers - named parameter syntax} {
    set handle [torch::rnn_tanh -inputSize 10 -hiddenSize 20 -numLayers 3]
    string match "rnn*" $handle
} {1}

test rnn_tanh-2.3 {RNN tanh with all parameters - named parameter syntax} {
    set handle [torch::rnn_tanh -inputSize 10 -hiddenSize 20 -numLayers 2 -bias true -batchFirst false -dropout 0.1 -bidirectional false]
    string match "rnn*" $handle
} {1}

test rnn_tanh-2.4 {RNN tanh bidirectional - named parameter syntax} {
    set handle [torch::rnn_tanh -inputSize 5 -hiddenSize 10 -bidirectional true]
    string match "rnn*" $handle
} {1}

test rnn_tanh-2.5 {RNN tanh with mixed parameter order - named syntax} {
    set handle [torch::rnn_tanh -hiddenSize 15 -inputSize 8 -dropout 0.3 -numLayers 2]
    string match "rnn*" $handle
} {1}

test rnn_tanh-2.6 {RNN tanh batch first enabled - named parameter syntax} {
    set handle [torch::rnn_tanh -inputSize 12 -hiddenSize 24 -batchFirst true -bias false]
    string match "rnn*" $handle
} {1}

# ============================================================================
# CamelCase Alias Tests
# ============================================================================

test rnn_tanh-3.1 {CamelCase alias - positional syntax} {
    set handle [torch::rnnTanh 10 20]
    string match "rnn*" $handle
} {1}

test rnn_tanh-3.2 {CamelCase alias - named parameter syntax} {
    set handle [torch::rnnTanh -inputSize 10 -hiddenSize 20 -numLayers 2]
    string match "rnn*" $handle
} {1}

test rnn_tanh-3.3 {CamelCase alias with all parameters} {
    set handle [torch::rnnTanh -inputSize 6 -hiddenSize 12 -numLayers 3 -bias true -batchFirst true -dropout 0.2 -bidirectional true]
    string match "rnn*" $handle
} {1}

# ============================================================================
# Parameter Validation Tests
# ============================================================================

test rnn_tanh-4.1 {Error: Missing required parameters - positional} {
    catch {torch::rnn_tanh 10} error
    string match "*Usage*" $error
} {1}

test rnn_tanh-4.2 {Error: Missing required parameters - named} {
    catch {torch::rnn_tanh -inputSize 10} error
    string match "*missing*" $error
} {1}

test rnn_tanh-4.3 {Error: Invalid input_size - positional} {
    catch {torch::rnn_tanh 0 20} error
    string match "*missing or invalid*" $error
} {1}

test rnn_tanh-4.4 {Error: Invalid hidden_size - named} {
    catch {torch::rnn_tanh -inputSize 10 -hiddenSize -5} error
    string match "*missing or invalid*" $error
} {1}

test rnn_tanh-4.5 {Error: Invalid num_layers - positional} {
    catch {torch::rnn_tanh 10 20 0} error
    string match "*missing or invalid*" $error
} {1}

test rnn_tanh-4.6 {Error: Invalid dropout value - named} {
    catch {torch::rnn_tanh -inputSize 10 -hiddenSize 20 -dropout -0.1} error
    string match "*missing or invalid*" $error
} {1}

test rnn_tanh-4.7 {Error: Unknown parameter - named} {
    catch {torch::rnn_tanh -inputSize 10 -hiddenSize 20 -invalidParam 5} error
    string match "*Unknown parameter*" $error
} {1}

test rnn_tanh-4.8 {Error: Non-numeric input_size - positional} {
    catch {torch::rnn_tanh abc 20} error
    string match "*Invalid*" $error
} {1}

test rnn_tanh-4.9 {Error: Non-boolean bias - named} {
    catch {torch::rnn_tanh -inputSize 10 -hiddenSize 20 -bias invalid} error
    string match "*Invalid bias*" $error
} {1}

test rnn_tanh-4.10 {Error: Missing parameter value - named} {
    catch {torch::rnn_tanh -inputSize 10 -hiddenSize} error
    string match "*Missing value*" $error
} {1}

# ============================================================================
# Edge Cases and Special Values
# ============================================================================

test rnn_tanh-5.1 {Large input and hidden sizes} {
    set handle [torch::rnn_tanh 1000 2000]
    string match "rnn*" $handle
} {1}

test rnn_tanh-5.2 {Single hidden unit} {
    set handle [torch::rnn_tanh 10 1]
    string match "rnn*" $handle
} {1}

test rnn_tanh-5.3 {Maximum dropout value with multiple layers} {
    set handle [torch::rnn_tanh -inputSize 10 -hiddenSize 20 -numLayers 2 -dropout 1.0]
    string match "rnn*" $handle
} {1}

test rnn_tanh-5.4 {Zero dropout value} {
    set handle [torch::rnn_tanh -inputSize 10 -hiddenSize 20 -dropout 0.0]
    string match "rnn*" $handle
} {1}

test rnn_tanh-5.5 {Many layers} {
    set handle [torch::rnn_tanh -inputSize 10 -hiddenSize 20 -numLayers 10]
    string match "rnn*" $handle
} {1}

# ============================================================================
# Syntax Equivalence Tests
# ============================================================================

test rnn_tanh-6.1 {Positional and named syntax equivalence - basic} {
    set handle1 [torch::rnn_tanh 15 30]
    set handle2 [torch::rnn_tanh -inputSize 15 -hiddenSize 30]
    expr {[string match "rnn*" $handle1] && [string match "rnn*" $handle2]}
} {1}

test rnn_tanh-6.2 {Positional and named syntax equivalence - with layers} {
    set handle1 [torch::rnn_tanh 10 20 3]
    set handle2 [torch::rnn_tanh -inputSize 10 -hiddenSize 20 -numLayers 3]
    expr {[string match "rnn*" $handle1] && [string match "rnn*" $handle2]}
} {1}

test rnn_tanh-6.3 {CamelCase and snake_case equivalence} {
    set handle1 [torch::rnn_tanh 10 20 2]
    set handle2 [torch::rnnTanh 10 20 2]
    expr {[string match "rnn*" $handle1] && [string match "rnn*" $handle2]}
} {1}

# ============================================================================
# Boolean Parameter Tests
# ============================================================================

test rnn_tanh-7.1 {Boolean values - true/false - named} {
    set handle [torch::rnn_tanh -inputSize 10 -hiddenSize 20 -bias true -batchFirst false -bidirectional true]
    string match "rnn*" $handle
} {1}

test rnn_tanh-7.2 {Boolean values - 1/0 - named} {
    set handle [torch::rnn_tanh -inputSize 10 -hiddenSize 20 -bias 1 -batchFirst 0 -bidirectional 1]
    string match "rnn*" $handle
} {1}

test rnn_tanh-7.3 {Boolean values - yes/no - named} {
    set handle [torch::rnn_tanh -inputSize 10 -hiddenSize 20 -bias yes -batchFirst no]
    string match "rnn*" $handle
} {1}

test rnn_tanh-7.4 {Boolean values - positional syntax} {
    set handle [torch::rnn_tanh 10 20 1 0 1]
    string match "rnn*" $handle
} {1}

cleanupTests 