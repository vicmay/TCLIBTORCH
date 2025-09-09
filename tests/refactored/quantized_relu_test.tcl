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
test quantized_relu-1.1 {Positional syntax error handling - too few arguments} {
    catch {torch::quantized_relu} msg
    string match "*Usage:*" $msg
} {1}

test quantized_relu-1.2 {Positional syntax error handling - invalid tensor} {
    catch {torch::quantized_relu "invalid_tensor_handle"} msg
    string match "*Invalid*" $msg
} {1}

# Test cases for named parameter syntax
test quantized_relu-2.1 {Named parameter syntax error handling - missing input} {
    catch {torch::quantized_relu -input} msg
    string match "*Missing value*" $msg
} {1}

test quantized_relu-2.2 {Named parameter syntax error handling - invalid parameter} {
    catch {torch::quantized_relu -invalid_param "tensor_handle"} msg
    string match "*Unknown parameter*" $msg
} {1}

test quantized_relu-2.3 {Named parameter syntax error handling - invalid tensor} {
    catch {torch::quantized_relu -input "invalid_tensor_handle"} msg
    string match "*Invalid*" $msg
} {1}

test quantized_relu-2.4 {Named parameter syntax error handling - no arguments} {
    catch {torch::quantized_relu} msg
    string match "*Usage:*" $msg
} {1}

# Test cases for camelCase alias
test quantized_relu-3.1 {CamelCase alias error handling - invalid tensor} {
    catch {torch::quantizedRelu "invalid_tensor_handle"} msg
    string match "*Invalid*" $msg
} {1}

test quantized_relu-3.2 {CamelCase alias with named parameters - invalid tensor} {
    catch {torch::quantizedRelu -input "invalid_tensor_handle"} msg
    string match "*Invalid*" $msg
} {1}

test quantized_relu-3.3 {CamelCase alias error handling - too few arguments} {
    catch {torch::quantizedRelu} msg
    string match "*Usage:*" $msg
} {1}

test quantized_relu-3.4 {CamelCase alias with invalid parameter} {
    catch {torch::quantizedRelu -invalid_param "tensor_handle"} msg
    string match "*Unknown parameter*" $msg
} {1}

# Syntax verification tests - ensuring command accepts correct syntax patterns
test quantized_relu-4.1 {Verify positional syntax pattern is accepted} {
    # This will fail with "Invalid tensor" but shows the syntax is parsed correctly
    catch {torch::quantized_relu "some_tensor"} msg
    # Should complain about invalid tensor, not syntax error
    string match "*Invalid*" $msg
} {1}

test quantized_relu-4.2 {Verify named parameter syntax pattern is accepted} {
    # This will fail with "Invalid tensor" but shows the syntax is parsed correctly  
    catch {torch::quantized_relu -input "some_tensor"} msg
    # Should complain about invalid tensor, not syntax error
    string match "*Invalid*" $msg
} {1}

test quantized_relu-4.3 {Verify camelCase positional syntax pattern is accepted} {
    # This will fail with "Invalid tensor" but shows the syntax is parsed correctly
    catch {torch::quantizedRelu "some_tensor"} msg
    # Should complain about invalid tensor, not syntax error
    string match "*Invalid*" $msg
} {1}

test quantized_relu-4.4 {Verify camelCase named parameter syntax pattern is accepted} {
    # This will fail with "Invalid tensor" but shows the syntax is parsed correctly
    catch {torch::quantizedRelu -input "some_tensor"} msg
    # Should complain about invalid tensor, not syntax error
    string match "*Invalid*" $msg
} {1}

# Edge cases for parameter validation
test quantized_relu-5.1 {Named parameter with empty value} {
    catch {torch::quantized_relu -input ""} msg
    string match "*Invalid*" $msg
} {1}

test quantized_relu-5.2 {Multiple invalid parameters} {
    catch {torch::quantized_relu -invalid1 "val1" -invalid2 "val2"} msg
    string match "*Unknown parameter*" $msg
} {1}

test quantized_relu-5.3 {Mixed valid and invalid parameters} {
    catch {torch::quantized_relu -input "tensor" -invalid "val"} msg
    string match "*Unknown parameter*" $msg
} {1}

# Functional test with actual tensors (should work since ReLU works on any tensor)
test quantized_relu-6.1 {Test with actual regular tensor using positional syntax} {
    # Create a regular tensor and test if quantized_relu can handle it
    set tensor [torch::tensorCreate -data {-1.0 0.0 1.0} -dtype float32]
    set result [torch::quantized_relu $tensor]
    # Should return a valid tensor handle (ReLU should work on regular tensors too)
    expr {[string match "tensor*" $result]}
} {1}

test quantized_relu-6.2 {Test with actual regular tensor using named syntax} {
    # Create a regular tensor and test if quantized_relu can handle it  
    set tensor [torch::tensorCreate -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32]
    set result [torch::quantized_relu -input $tensor]
    # Should return a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test quantized_relu-6.3 {Test camelCase with actual regular tensor using positional syntax} {
    # Create a regular tensor and test camelCase alias
    set tensor [torch::tensorCreate -data {-1.5 0.5 1.5} -dtype float32]
    set result [torch::quantizedRelu $tensor]
    # Should return a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test quantized_relu-6.4 {Test camelCase with actual regular tensor using named syntax} {
    # Create a regular tensor and test camelCase alias with named parameters
    set tensor [torch::tensorCreate -data {-3.0 0.0 3.0} -dtype float32]
    set result [torch::quantizedRelu -input $tensor]
    # Should return a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

cleanupTests 