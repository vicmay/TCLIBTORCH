#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# Create test layer and input tensor
set layer_name [torch::linear 10 5]
set input_tensor [torch::randn -shape {3 10}]

;# Test 1: Basic positional syntax (backward compatibility)
test layer_forward-1.1 {Basic positional syntax} {
    set result [torch::layer_forward $layer_name $input_tensor]
    expr {$result ne ""}
} {1}

;# Test 2: Named parameter syntax
test layer_forward-2.1 {Named parameter syntax with -layer and -input} {
    set result [torch::layer_forward -layer $layer_name -input $input_tensor]
    expr {$result ne ""}
} {1}

;# Test 3: Named parameter syntax with different order
test layer_forward-2.2 {Named parameter syntax with different order} {
    set result [torch::layer_forward -input $input_tensor -layer $layer_name]
    expr {$result ne ""}
} {1}

;# Test 4: camelCase alias with positional syntax
test layer_forward-3.1 {camelCase alias with positional syntax} {
    set result [torch::layerForward $layer_name $input_tensor]
    expr {$result ne ""}
} {1}

;# Test 5: camelCase alias with named parameter syntax
test layer_forward-3.2 {camelCase alias with named parameter syntax} {
    set result [torch::layerForward -layer $layer_name -input $input_tensor]
    expr {$result ne ""}
} {1}

;# Test 6: Error handling - invalid layer name (positional)
test layer_forward-4.1 {Error handling - invalid layer name (positional)} {
    catch {torch::layer_forward "invalid_layer" $input_tensor} msg
    expr {[string match "*Invalid layer name*" $msg]}
} {1}

;# Test 7: Error handling - invalid input tensor (positional)
test layer_forward-4.2 {Error handling - invalid input tensor (positional)} {
    catch {torch::layer_forward $layer_name "invalid_tensor"} msg
    expr {[string match "*Invalid input tensor name*" $msg]}
} {1}

;# Test 8: Error handling - invalid layer name (named)
test layer_forward-4.3 {Error handling - invalid layer name (named)} {
    catch {torch::layer_forward -layer "invalid_layer" -input $input_tensor} msg
    expr {[string match "*Invalid layer name*" $msg]}
} {1}

;# Test 9: Error handling - invalid input tensor (named)
test layer_forward-4.4 {Error handling - invalid input tensor (named)} {
    catch {torch::layer_forward -layer $layer_name -input "invalid_tensor"} msg
    expr {[string match "*Invalid input tensor name*" $msg]}
} {1}

;# Test 10: Error handling - missing parameter value
test layer_forward-4.5 {Error handling - missing parameter value} {
    catch {torch::layer_forward -layer} msg
    expr {[string match "*Missing value for parameter*" $msg]}
} {1}

;# Test 11: Error handling - unknown parameter
test layer_forward-4.6 {Error handling - unknown parameter} {
    catch {torch::layer_forward -unknown_param value -layer $layer_name} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

;# Test 12: Error handling - missing required parameters
test layer_forward-4.7 {Error handling - missing required parameters} {
    catch {torch::layer_forward -layer ""} msg
    expr {[string match "*Required parameters missing*" $msg]}
} {1}

;# Test 13: Verify output shape is correct
test layer_forward-5.1 {Verify output shape is correct} {
    set output [torch::layer_forward $layer_name $input_tensor]
    set shape [torch::tensor_shape $output]
    expr {$shape eq "3 5"}
} {1}

;# Test 14: Test different layer types
test layer_forward-5.2 {Test different layer types - conv2d} {
    set conv_layer [torch::conv2d 3 16 3]
    set conv_input [torch::randn -shape {1 3 32 32}]
    set output [torch::layer_forward $conv_layer $conv_input]
    expr {$output ne ""}
} {1}

;# Test 15: Test sequential layer
test layer_forward-5.3 {Test sequential layer} {
    set linear1 [torch::linear 10 8]
    set linear2 [torch::linear 8 5]
    set seq_layer [torch::sequential [list $linear1 $linear2]]
    set output [torch::layer_forward $seq_layer $input_tensor]
    set shape [torch::tensor_shape $output]
    expr {$shape eq "3 5"}
} {1}

;# Test 16: Multiple parameter formats work identically
test layer_forward-6.1 {Multiple parameter formats produce same result} {
    ;# Test that both syntaxes produce the same output shape
    set result1 [torch::layer_forward $layer_name $input_tensor]
    set result2 [torch::layer_forward -layer $layer_name -input $input_tensor]
    set result3 [torch::layerForward $layer_name $input_tensor]
    set result4 [torch::layerForward -layer $layer_name -input $input_tensor]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set shape3 [torch::tensor_shape $result3]
    set shape4 [torch::tensor_shape $result4]
    
    expr {$shape1 eq $shape2 && $shape2 eq $shape3 && $shape3 eq $shape4}
} {1}

;# Test 17: Chaining forward passes
test layer_forward-6.2 {Chaining forward passes} {
    set linear1 [torch::linear 10 8]
    set linear2 [torch::linear 8 5]
    
    ;# Forward through first layer
    set intermediate [torch::layer_forward $linear1 $input_tensor]
    ;# Forward through second layer
    set final_output [torch::layer_forward $linear2 $intermediate]
    
    set shape [torch::tensor_shape $final_output]
    expr {$shape eq "3 5"}
} {1}

;# Test 18: Batch processing
test layer_forward-6.3 {Batch processing with different batch sizes} {
    ;# Test with different batch sizes
    set input_batch1 [torch::randn -shape {1 10}]
    set input_batch5 [torch::randn -shape {5 10}]
    set input_batch10 [torch::randn -shape {10 10}]
    
    set output1 [torch::layer_forward $layer_name $input_batch1]
    set output5 [torch::layer_forward $layer_name $input_batch5]
    set output10 [torch::layer_forward $layer_name $input_batch10]
    
    set shape1 [torch::tensor_shape $output1]
    set shape5 [torch::tensor_shape $output5]
    set shape10 [torch::tensor_shape $output10]
    
    expr {$shape1 eq "1 5" && $shape5 eq "5 5" && $shape10 eq "10 5"}
} {1}

;# Test 19: Different data types
test layer_forward-6.4 {Different data types} {
    ;# Create input with different data type if supported
    set float_input [torch::randn -shape {2 10}]
    set output [torch::layer_forward $layer_name $float_input]
    expr {$output ne ""}
} {1}

;# Test 20: Layer state consistency
test layer_forward-6.5 {Layer state consistency across calls} {
    ;# Multiple forward passes should work consistently
    set output1 [torch::layer_forward $layer_name $input_tensor]
    set output2 [torch::layer_forward $layer_name $input_tensor]
    set output3 [torch::layer_forward $layer_name $input_tensor]
    
    set shape1 [torch::tensor_shape $output1]
    set shape2 [torch::tensor_shape $output2]
    set shape3 [torch::tensor_shape $output3]
    
    expr {$shape1 eq $shape2 && $shape2 eq $shape3}
} {1}

cleanupTests 