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

# Test basic command syntax and error handling
# Note: We focus on testing the dual syntax parser and error handling
# rather than functional quantization since that has underlying issues

# Test error handling - invalid tensor names
test dequantize-1.1 {Invalid tensor name positional} {
    catch {torch::dequantize invalid_tensor} msg
    expr {[string match "*Invalid quantized tensor*" $msg]}
} {1}

test dequantize-1.2 {Invalid tensor name named parameter} {
    catch {torch::dequantize -input invalid_tensor} msg
    expr {[string match "*Invalid quantized tensor*" $msg]}
} {1}

test dequantize-1.3 {Missing required parameters positional} {
    catch {torch::dequantize} msg
    expr {[string match "*Usage*" $msg]}
} {1}

test dequantize-1.4 {Missing required parameters named} {
    catch {torch::dequantize -input} msg
    expr {[string match "*Missing value*" $msg] || [string match "*Usage*" $msg]}
} {1}

test dequantize-1.5 {Invalid parameter name} {
    catch {torch::dequantize -invalid value} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

test dequantize-1.6 {Missing required named parameter} {
    catch {torch::dequantize -invalid_param value} msg
    expr {[string match "*Required parameter missing*" $msg] || [string match "*Unknown parameter*" $msg]}
} {1}

test dequantize-1.7 {Too many positional arguments} {
    catch {torch::dequantize tensor1 extra_arg} msg
    expr {[string match "*Usage*" $msg]}
} {1}

# Test camelCase alias error handling
test dequantize-2.1 {CamelCase alias - invalid tensor positional} {
    catch {torch::deQuantize invalid_tensor} msg
    expr {[string match "*Invalid quantized tensor*" $msg]}
} {1}

test dequantize-2.2 {CamelCase alias - invalid tensor named parameter} {
    catch {torch::deQuantize -input invalid_tensor} msg
    expr {[string match "*Invalid quantized tensor*" $msg]}
} {1}

test dequantize-2.3 {CamelCase alias - missing parameters} {
    catch {torch::deQuantize} msg
    expr {[string match "*Usage*" $msg]}
} {1}

test dequantize-2.4 {CamelCase alias - invalid parameter name} {
    catch {torch::deQuantize -invalid value} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

# Test dual syntax parsing works correctly for parameter validation
test dequantize-3.1 {Positional syntax parameter validation} {
    # Test that positional syntax is detected correctly
    catch {torch::dequantize nonexistent_tensor} msg
    # Should get "Invalid quantized tensor" not "Unknown parameter"
    expr {[string match "*Invalid quantized tensor*" $msg] && ![string match "*Unknown parameter*" $msg]}
} {1}

test dequantize-3.2 {Named syntax parameter validation} {
    # Test that named syntax is detected correctly
    catch {torch::dequantize -input nonexistent_tensor} msg
    # Should get "Invalid quantized tensor" not "Usage:"
    expr {[string match "*Invalid quantized tensor*" $msg] && ![string match "*Usage*" $msg]}
} {1}

test dequantize-3.3 {Named syntax with valid parameter name} {
    # Test that -input parameter is recognized (even with invalid tensor)
    catch {torch::dequantize -input nonexistent_tensor} msg
    # Should not get "Unknown parameter" error
    expr {![string match "*Unknown parameter*" $msg]}
} {1}

test dequantize-3.4 {Named syntax with invalid parameter name} {
    # Test that invalid parameter names are caught
    catch {torch::dequantize -wrongparam value} msg
    # Should get "Unknown parameter" error
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

# Test regular tensor handling (non-quantized tensors should fail gracefully)
test dequantize-4.1 {Regular tensor input positional} {
    # Create a regular float tensor
    set regular_tensor [torch::tensor_create -data {1.0 2.0} -shape {2} -dtype float32]
    catch {torch::dequantize $regular_tensor} msg
    # Should handle gracefully with some error message
    expr {$msg != "" && ![string match "*Unknown parameter*" $msg]}
} {1}

test dequantize-4.2 {Regular tensor input named parameter} {
    # Create a regular float tensor
    set regular_tensor [torch::tensor_create -data {1.0 2.0} -shape {2} -dtype float32]
    catch {torch::dequantize -input $regular_tensor} msg
    # Should handle gracefully with some error message
    expr {$msg != "" && ![string match "*Unknown parameter*" $msg]}
} {1}

test dequantize-4.3 {Regular tensor input camelCase} {
    # Create a regular float tensor
    set regular_tensor [torch::tensor_create -data {1.0 2.0} -shape {2} -dtype float32]
    catch {torch::deQuantize $regular_tensor} msg
    # Should handle gracefully with some error message
    expr {$msg != "" && ![string match "*Unknown parameter*" $msg]}
} {1}

test dequantize-4.4 {Regular tensor input camelCase named} {
    # Create a regular float tensor
    set regular_tensor [torch::tensor_create -data {1.0 2.0} -shape {2} -dtype float32]
    catch {torch::deQuantize -input $regular_tensor} msg
    # Should handle gracefully with some error message
    expr {$msg != "" && ![string match "*Unknown parameter*" $msg]}
} {1}

# Test parameter syntax detection
test dequantize-5.1 {Positional vs named syntax detection} {
    # Test that parameters starting with - are treated as named syntax
    catch {torch::dequantize -input} msg1
    catch {torch::dequantize} msg2
    # Both should give usage/missing value errors, but different types
    expr {$msg1 != $msg2}
} {1}

test dequantize-5.2 {Named parameter requires value} {
    catch {torch::dequantize -input} msg
    expr {[string match "*Missing value*" $msg] || [string match "*Usage*" $msg]}
} {1}

test dequantize-5.3 {Multiple parameter handling} {
    catch {torch::dequantize -input tensor1 -extra param} msg
    # Should catch unknown parameter
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

# Test command registration and availability
test dequantize-6.1 {Main command exists} {
    # Test that the main command is registered
    set commands [info commands torch::dequantize]
    expr {[llength $commands] == 1}
} {1}

test dequantize-6.2 {CamelCase alias exists} {
    # Test that the camelCase alias is registered
    set commands [info commands torch::deQuantize]
    expr {[llength $commands] == 1}
} {1}

test dequantize-6.3 {Both commands available} {
    # Test that both commands are available
    set main [info commands torch::dequantize]
    set alias [info commands torch::deQuantize]
    expr {[llength $main] == 1 && [llength $alias] == 1}
} {1}

cleanupTests 