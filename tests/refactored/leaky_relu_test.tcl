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

# Test 1: Basic functionality with positional syntax
test leaky_relu-1.1 {Basic leaky_relu with positional syntax - default slope} {
    set tensor [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32 -device cpu]
    set result [torch::leaky_relu $tensor]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    expr {$shape eq "5" && [string match "*Float*" $dtype]}
} {1}

test leaky_relu-1.2 {LeakyRelu with positive value (unchanged)} {
    set tensor [torch::full {1} 2.0]
    set result [torch::leaky_relu $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 2.0) < 0.01}
} {1}

test leaky_relu-1.3 {LeakyRelu with negative value (default slope)} {
    set tensor [torch::full {1} -1.0]
    set result [torch::leaky_relu $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - (-0.01)) < 0.001}
} {1}

test leaky_relu-1.4 {LeakyRelu with custom slope} {
    set tensor [torch::full {1} -1.0]
    set result [torch::leaky_relu $tensor 0.2]
    set value [torch::tensor_item $result]
    expr {abs($value - (-0.2)) < 0.01}
} {1}

# Test 2: Named parameter syntax
test leaky_relu-2.1 {LeakyRelu with named parameters - basic} {
    set tensor [torch::full {1} -2.0]
    set result [torch::leaky_relu -input $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - (-0.02)) < 0.001}
} {1}

test leaky_relu-2.2 {LeakyRelu with named parameters - custom slope} {
    set tensor [torch::full {1} -2.0]
    set result [torch::leaky_relu -input $tensor -negativeSlope 0.1]
    set value [torch::tensor_item $result]
    expr {abs($value - (-0.2)) < 0.01}
} {1}

test leaky_relu-2.3 {Alternative parameter names} {
    set tensor [torch::full {1} -2.0]
    set result1 [torch::leaky_relu -input $tensor -negative_slope 0.1]
    set result2 [torch::leaky_relu -input $tensor -slope 0.1]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    expr {abs($value1 - $value2) < 0.000001}
} {1}

# Test 3: CamelCase alias tests
test leaky_relu-3.1 {LeakyRelu camelCase alias basic} {
    set tensor [torch::full {1} -1.0]
    set result [torch::leakyRelu $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - (-0.01)) < 0.001}
} {1}

test leaky_relu-3.2 {LeakyRelu camelCase with named parameters} {
    set tensor [torch::full {1} -3.0]
    set result [torch::leakyRelu -input $tensor -negativeSlope 0.1]
    set value [torch::tensor_item $result]
    expr {abs($value - (-0.3)) < 0.01}
} {1}

# Test 4: Error handling
test leaky_relu-4.1 {LeakyRelu with invalid tensor name} {
    catch {torch::leaky_relu invalid_tensor} error
    string match "*Invalid tensor name*" $error
} 1

test leaky_relu-4.2 {LeakyRelu named syntax with invalid parameter} {
    set tensor [torch::full {1} 1.0]
    catch {torch::leaky_relu -invalid $tensor} error
    string match "*Unknown parameter*" $error
} 1

test leaky_relu-4.3 {LeakyRelu with negative slope validation} {
    set tensor [torch::full {1} 1.0]
    catch {torch::leaky_relu -input $tensor -negativeSlope -0.1} error
    string match "*negative_slope must be >= 0*" $error
} 1

# Test 5: Mathematical properties
test leaky_relu-5.1 {LeakyRelu zero input} {
    set tensor [torch::full {1} 0.0]
    set result [torch::leaky_relu $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value) < 0.001}
} {1}

test leaky_relu-5.2 {LeakyRelu different slopes} {
    set tensor [torch::full {1} -1.0]
    set result1 [torch::leaky_relu $tensor 0.01]
    set result2 [torch::leaky_relu $tensor 0.1]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    expr {abs($value1 - $value2) > 0.01}
} {1}

# Test 6: Syntax consistency
test leaky_relu-6.1 {Positional and named syntax consistency} {
    set tensor [torch::full {1} -2.0]
    set result1 [torch::leaky_relu $tensor 0.1]
    set result2 [torch::leaky_relu -input $tensor -negativeSlope 0.1]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    expr {abs($value1 - $value2) < 0.000001}
} {1}

test leaky_relu-6.2 {Snake_case and camelCase consistency} {
    set tensor [torch::full {1} -1.5]
    set result1 [torch::leaky_relu $tensor 0.1]
    set result2 [torch::leakyRelu $tensor 0.1]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    expr {abs($value1 - $value2) < 0.000001}
} {1}

cleanupTests 