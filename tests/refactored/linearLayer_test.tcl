#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Configure test output
configure -verbose {pass fail skip error}

# Load the extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# -----------------------------------------------------------------------------
# 1. Positional syntax tests
# -----------------------------------------------------------------------------

# Basic positional
test linearLayer-1.1 {Basic positional syntax} {
    set layer [torch::linearLayer 10 5]
    string match "linear*" $layer
} {1}

# Positional with bias parameter
test linearLayer-1.2 {Positional syntax with bias parameter} {
    set layer [torch::linearLayer 20 10 false]
    string match "linear*" $layer
} {1}

# Positional with bias true
test linearLayer-1.3 {Positional syntax with bias true} {
    set layer [torch::linearLayer 128 64 true]
    string match "linear*" $layer
} {1}

# Different layer sizes
test linearLayer-1.4 {Positional syntax with different sizes} {
    set layer [torch::linearLayer 784 256]
    string match "linear*" $layer
} {1}

# Large dimensions
test linearLayer-1.5 {Positional syntax with large dimensions} {
    set layer [torch::linearLayer 4096 1024]
    string match "linear*" $layer
} {1}

# -----------------------------------------------------------------------------
# 2. Named parameter syntax tests
# -----------------------------------------------------------------------------

# Basic named
test linearLayer-2.1 {Named syntax required params} {
    set layer [torch::linearLayer -inFeatures 10 -outFeatures 5]
    string match "linear*" $layer
} {1}

# Named with bias
test linearLayer-2.2 {Named syntax with bias parameter} {
    set layer [torch::linearLayer -inFeatures 20 -outFeatures 10 -bias false]
    string match "linear*" $layer
} {1}

# Named with different parameter order
test linearLayer-2.3 {Named syntax mixed parameter order} {
    set layer [torch::linearLayer -bias true -outFeatures 15 -inFeatures 30]
    string match "linear*" $layer
} {1}

# Named with all parameters
test linearLayer-2.4 {Named syntax with all parameters} {
    set layer [torch::linearLayer -inFeatures 512 -outFeatures 256 -bias true]
    string match "linear*" $layer
} {1}

# -----------------------------------------------------------------------------
# 3. CamelCase alias tests (torch::linear)
# -----------------------------------------------------------------------------

# Basic alias
test linearLayer-3.1 {CamelCase alias basic} {
    set layer [torch::linear 10 5]
    string match "linear*" $layer
} {1}

# Alias with named params
test linearLayer-3.2 {CamelCase alias with named params} {
    set layer [torch::linear -inFeatures 20 -outFeatures 10 -bias false]
    string match "linear*" $layer
} {1}

# Syntax consistency check
test linearLayer-3.3 {Syntax consistency positional vs named vs alias} {
    set l1 [torch::linearLayer 100 50 true]
    set l2 [torch::linearLayer -inFeatures 100 -outFeatures 50 -bias true]
    set l3 [torch::linear -inFeatures 100 -outFeatures 50 -bias true]
    
    # Check that all produce valid layer handles
    set r1 [string match "linear*" $l1]
    set r2 [string match "linear*" $l2]
    set r3 [string match "linear*" $l3]
    
    expr {$r1 && $r2 && $r3}
} {1}

# -----------------------------------------------------------------------------
# 4. Error handling tests
# -----------------------------------------------------------------------------

# Missing inFeatures
test linearLayer-4.1 {Error: missing inFeatures} {
    set code [catch {torch::linearLayer -outFeatures 10} msg]
    list $code [string match "*inFeatures*" $msg]
} {1 1}

# Missing outFeatures
test linearLayer-4.2 {Error: missing outFeatures} {
    set code [catch {torch::linearLayer -inFeatures 10} msg]
    list $code [string match "*outFeatures*" $msg]
} {1 1}

# Invalid inFeatures (negative)
test linearLayer-4.3 {Error: negative inFeatures} {
    set code [catch {torch::linearLayer -inFeatures -10 -outFeatures 5} msg]
    list $code [string match "*positive*" $msg]
} {1 1}

# Invalid outFeatures (zero)
test linearLayer-4.4 {Error: zero outFeatures} {
    set code [catch {torch::linearLayer -inFeatures 10 -outFeatures 0} msg]
    list $code [string match "*positive*" $msg]
} {1 1}

# Invalid bias type
test linearLayer-4.5 {Error: invalid bias type} {
    set code [catch {torch::linearLayer -inFeatures 10 -outFeatures 5 -bias invalid} msg]
    list $code [string match "*bias*" $msg]
} {1 1}

# Unknown parameter
test linearLayer-4.6 {Error: unknown parameter} {
    set code [catch {torch::linearLayer -inFeatures 10 -outFeatures 5 -invalid_param 1} msg]
    list $code [string match "*Unknown parameter*" $msg]
} {1 1}

# Missing parameter value
test linearLayer-4.7 {Error: missing parameter value} {
    set code [catch {torch::linearLayer -inFeatures 10 -outFeatures} msg]
    list $code [string match "*Missing value*" $msg]
} {1 1}

# Too few positional arguments
test linearLayer-4.8 {Error: too few positional arguments} {
    set code [catch {torch::linearLayer 10} msg]
    list $code [string match "*Usage*" $msg]
} {1 1}

# Too many positional arguments
test linearLayer-4.9 {Error: too many positional arguments} {
    set code [catch {torch::linearLayer 10 5 true extra} msg]
    list $code [string match "*Usage*" $msg]
} {1 1}

# -----------------------------------------------------------------------------
# 5. Layer functionality tests
# -----------------------------------------------------------------------------

# Forward pass basic
test linearLayer-5.1 {Forward pass basic functionality} {
    set layer [torch::linearLayer 3 2]
    set input [torch::randn -shape {1 3}]
    set output [torch::layer_forward $layer $input]
    set shape [torch::tensor_shape $output]
    set expected {1 2}
    expr {$shape == $expected}
} {1}

# Forward pass with batch
test linearLayer-5.2 {Forward pass with batch} {
    set layer [torch::linearLayer 4 3]
    set input [torch::randn -shape {5 4}]
    set output [torch::layer_forward $layer $input]
    set shape [torch::tensor_shape $output]
    set expected {5 3}
    expr {$shape == $expected}
} {1}

# Forward pass without bias
test linearLayer-5.3 {Forward pass without bias} {
    set layer [torch::linearLayer -inFeatures 5 -outFeatures 3 -bias false]
    set input [torch::randn -shape {2 5}]
    set output [torch::layer_forward $layer $input]
    set shape [torch::tensor_shape $output]
    set expected {2 3}
    expr {$shape == $expected}
} {1}

# Large layer dimensions
test linearLayer-5.4 {Large layer dimensions} {
    set layer [torch::linearLayer 1000 500]
    set input [torch::randn -shape {1 1000}]
    set output [torch::layer_forward $layer $input]
    set shape [torch::tensor_shape $output]
    set expected {1 500}
    expr {$shape == $expected}
} {1}

# Multiple forward passes
test linearLayer-5.5 {Multiple forward passes} {
    set layer [torch::linearLayer 4 2]
    set input1 [torch::randn -shape {1 4}]
    set input2 [torch::randn -shape {1 4}]
    
    set output1 [torch::layer_forward $layer $input1]
    set output2 [torch::layer_forward $layer $input2]
    
    set shape1 [torch::tensor_shape $output1]
    set shape2 [torch::tensor_shape $output2]
    
    expr {$shape1 == {1 2} && $shape2 == {1 2}}
} {1}

# -----------------------------------------------------------------------------
# 6. Edge cases and boundary conditions
# -----------------------------------------------------------------------------

# Single input/output neuron
test linearLayer-6.1 {Single input/output neuron} {
    set layer [torch::linearLayer 1 1]
    set input [torch::randn -shape {1 1}]
    set output [torch::layer_forward $layer $input]
    set shape [torch::tensor_shape $output]
    set expected {1 1}
    expr {$shape == $expected}
} {1}

# Wide layer (many inputs, few outputs)
test linearLayer-6.2 {Wide layer} {
    set layer [torch::linearLayer 1000 10]
    string match "linear*" $layer
} {1}

# Tall layer (few inputs, many outputs)
test linearLayer-6.3 {Tall layer} {
    set layer [torch::linearLayer 10 1000]
    string match "linear*" $layer
} {1}

# Bias variations
test linearLayer-6.4 {Bias parameter variations} {
    set layer1 [torch::linearLayer 5 3 1]
    set layer2 [torch::linearLayer 5 3 0]
    set layer3 [torch::linearLayer 5 3 true]
    set layer4 [torch::linearLayer 5 3 false]
    
    set r1 [string match "linear*" $layer1]
    set r2 [string match "linear*" $layer2]
    set r3 [string match "linear*" $layer3]
    set r4 [string match "linear*" $layer4]
    
    expr {$r1 && $r2 && $r3 && $r4}
} {1}

# Different data types via forward pass
test linearLayer-6.5 {Different input data types} {
    set layer [torch::linearLayer 3 2]
    set input_float [torch::randn -shape {1 3} -dtype float32]
    set output_float [torch::layer_forward $layer $input_float]
    
    set shape [torch::tensor_shape $output_float]
    set expected {1 2}
    expr {$shape == $expected}
} {1}

# -----------------------------------------------------------------------------
cleanupTests 