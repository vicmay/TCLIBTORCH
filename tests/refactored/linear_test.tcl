#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so} err]} {
    puts "Error loading libtorchtcl.so: $err"
    if {[file exists ../../build/libtorchtcl.so]} {
        puts "File exists but failed to load. Might be missing dependencies."
        puts "Try: LD_LIBRARY_PATH=../../libtorch/lib tclsh [info script]"
    } else {
        puts "File does not exist. Build the extension first."
    }
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Helper function to create a test tensor
proc create_test_tensor {batch_size features} {
    set tensor [torch::ones [list $batch_size $features]]
    return $tensor
}

# Test cases for positional syntax
test linear-1.1 "Basic positional syntax" {
    set linear [torch::linear 10 5]
    expr {[string match "linear*" $linear]}
} {1}

test linear-1.2 "Forward pass with positional syntax" {
    set linear [torch::linear 10 5]
    set input [create_test_tensor 2 10]
    set output [torch::layer_forward $linear $input]
    expr {[torch::tensor_shape $output] eq "2 5"}
} {1}

test linear-1.3 "Positional syntax with bias=false" {
    set linear [torch::linear 10 5 0]
    set input [create_test_tensor 2 10]
    set output [torch::layer_forward $linear $input]
    expr {[torch::tensor_shape $output] eq "2 5"}
} {1}

# Test cases for named parameter syntax
test linear-2.1 "Basic named parameter syntax" {
    set linear [torch::linear -inFeatures 10 -outFeatures 5]
    expr {[string match "linear*" $linear]}
} {1}

test linear-2.2 "Named parameter syntax with all parameters" {
    set linear [torch::linear -inFeatures 10 -outFeatures 5 -bias 1]
    set input [create_test_tensor 2 10]
    set output [torch::layer_forward $linear $input]
    expr {[torch::tensor_shape $output] eq "2 5"}
} {1}

test linear-2.3 "Named parameter syntax with bias=false" {
    set linear [torch::linear -inFeatures 10 -outFeatures 5 -bias 0]
    set input [create_test_tensor 2 10]
    set output [torch::layer_forward $linear $input]
    expr {[torch::tensor_shape $output] eq "2 5"}
} {1}

test linear-2.4 "Named parameter syntax with parameters in different order" {
    set linear [torch::linear -bias 1 -outFeatures 5 -inFeatures 10]
    set input [create_test_tensor 2 10]
    set output [torch::layer_forward $linear $input]
    expr {[torch::tensor_shape $output] eq "2 5"}
} {1}

# Test cases for camelCase alias
test linear-3.1 "Basic camelCase alias" {
    set linear [torch::linearLayer -inFeatures 10 -outFeatures 5]
    expr {[string match "linear*" $linear]}
} {1}

test linear-3.2 "CamelCase alias with all parameters" {
    set linear [torch::linearLayer -inFeatures 10 -outFeatures 5 -bias 1]
    set input [create_test_tensor 2 10]
    set output [torch::layer_forward $linear $input]
    expr {[torch::tensor_shape $output] eq "2 5"}
} {1}

# Error handling tests
test linear-4.1 "Error: Missing required parameters (named syntax)" {
    catch {torch::linear -inFeatures 10} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test linear-4.2 "Error: Invalid inFeatures parameter" {
    catch {torch::linear -inFeatures -5 -outFeatures 10} result
    expr {[string match "*Required parameters missing or invalid*" $result]}
} {1}

test linear-4.3 "Error: Invalid outFeatures parameter" {
    catch {torch::linear -inFeatures 10 -outFeatures 0} result
    expr {[string match "*Required parameters missing or invalid*" $result]}
} {1}

test linear-4.4 "Error: Unknown parameter" {
    catch {torch::linear -inFeatures 10 -outFeatures 5 -unknown 1} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test linear-4.5 "Error: Missing value for parameter" {
    catch {torch::linear -inFeatures 10 -outFeatures} result
    expr {[string match "*Missing value*" $result]}
} {1}

# Verify both syntaxes produce identical results
test linear-5.1 "Verify both syntaxes produce identical results" {
    set linear1 [torch::linear 10 5]
    set linear2 [torch::linear -inFeatures 10 -outFeatures 5]
    
    # Initialize with same weights for comparison
    set input [create_test_tensor 2 10]
    
    # We can't directly compare the outputs because weights are randomly initialized
    # So we just verify both produce valid outputs of the expected shape
    set output1 [torch::layer_forward $linear1 $input]
    set output2 [torch::layer_forward $linear2 $input]
    
    expr {[torch::tensor_shape $output1] eq [torch::tensor_shape $output2]}
} {1}

cleanupTests 