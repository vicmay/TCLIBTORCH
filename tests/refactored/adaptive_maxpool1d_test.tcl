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

# Test 1: Basic positional syntax (backward compatibility)
test adaptive_maxpool1d-1.1 {Basic positional syntax} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0}
    set tensor [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad false]
    set reshaped [torch::tensor_reshape $tensor {2 3 4}]
    set result [torch::adaptive_maxpool1d $reshaped 2]
    
    set shape [torch::tensor_shape $result]
    string equal $shape {2 3 2}
} {1}

# Test 2: Named parameter syntax with -input and -output_size
test adaptive_maxpool1d-2.1 {Named parameter syntax with -input and -output_size} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0}
    set tensor [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad false]
    set reshaped [torch::tensor_reshape $tensor {2 3 4}]
    set result [torch::adaptive_maxpool1d -input $reshaped -output_size 2]
    
    set shape [torch::tensor_shape $result]
    string equal $shape {2 3 2}
} {1}

# Test 3: Named parameter syntax with -tensor and -outputSize
test adaptive_maxpool1d-2.2 {Named parameter syntax with -tensor and -outputSize} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0}
    set tensor [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad false]
    set reshaped [torch::tensor_reshape $tensor {1 4 3}]
    set result [torch::adaptive_maxpool1d -tensor $reshaped -outputSize 2]
    
    set shape [torch::tensor_shape $result]
    string equal $shape {1 4 2}
} {1}

# Test 4: Named parameter syntax reversed order
test adaptive_maxpool1d-2.3 {Named parameter syntax reversed order} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0}
    set tensor [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad false]
    set reshaped [torch::tensor_reshape $tensor {3 2 2}]
    set result [torch::adaptive_maxpool1d -output_size 1 -input $reshaped]
    
    set shape [torch::tensor_shape $result]
    string equal $shape {3 2 1}
} {1}

# Test 5: camelCase alias with positional syntax
test adaptive_maxpool1d-3.1 {camelCase alias with positional syntax} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0}
    set tensor [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad false]
    set reshaped [torch::tensor_reshape $tensor {2 3 4}]
    set result [torch::adaptiveMaxpool1d $reshaped 2]
    
    set shape [torch::tensor_shape $result]
    string equal $shape {2 3 2}
} {1}

# Test 6: camelCase alias with named parameters
test adaptive_maxpool1d-3.2 {camelCase alias with named parameters} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0}
    set tensor [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad false]
    set reshaped [torch::tensor_reshape $tensor {1 4 3}]
    set result [torch::adaptiveMaxpool1d -input $reshaped -outputSize 2]
    
    set shape [torch::tensor_shape $result]
    string equal $shape {1 4 2}
} {1}

# Test 7: Both syntaxes produce same shape
test adaptive_maxpool1d-4.1 {Both syntaxes produce same shape} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0}
    set tensor [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad false]
    set reshaped [torch::tensor_reshape $tensor {2 3 4}]
    
    set result_pos [torch::adaptive_maxpool1d $reshaped 2]
    set result_named [torch::adaptive_maxpool1d -input $reshaped -output_size 2]
    
    set shape1 [torch::tensor_shape $result_pos]
    set shape2 [torch::tensor_shape $result_named]
    
    string equal $shape1 $shape2
} {1}

# Test 8: camelCase produces same shape as original
test adaptive_maxpool1d-4.2 {camelCase produces same shape as original} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0}
    set tensor [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad false]
    set reshaped [torch::tensor_reshape $tensor {1 4 3}]
    
    set result_orig [torch::adaptive_maxpool1d $reshaped 2]
    set result_camel [torch::adaptiveMaxpool1d $reshaped 2]
    
    set shape1 [torch::tensor_shape $result_orig]
    set shape2 [torch::tensor_shape $result_camel]
    
    string equal $shape1 $shape2
} {1}

# Test 9: Error on missing parameters - positional
test adaptive_maxpool1d-5.1 {Error on missing parameters - positional} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0}
    set tensor [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad false]
    set reshaped [torch::tensor_reshape $tensor {2 3 1}]
    set result [catch {torch::adaptive_maxpool1d $reshaped} msg]
    expr {$result == 1 && [string match "*Wrong number of arguments*" $msg]}
} {1}

# Test 10: Error on missing parameters - named
test adaptive_maxpool1d-5.2 {Error on missing parameters - named} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0}
    set tensor [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad false]
    set reshaped [torch::tensor_reshape $tensor {2 3 1}]
    set result [catch {torch::adaptive_maxpool1d -input $reshaped} msg]
    expr {$result == 1 && [string match "*Required parameters*" $msg]}
} {1}

# Test 11: Error on invalid tensor name
test adaptive_maxpool1d-5.3 {Error on invalid tensor name} {
    set result [catch {torch::adaptive_maxpool1d invalid_tensor 2} msg]
    expr {$result == 1 && [string match "*Invalid input tensor*" $msg]}
} {1}

# Test 12: Error on invalid output_size - positional
test adaptive_maxpool1d-5.4 {Error on invalid output_size - positional} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0}
    set tensor [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad false]
    set reshaped [torch::tensor_reshape $tensor {2 3 1}]
    set result [catch {torch::adaptive_maxpool1d $reshaped "not_a_number"} msg]
    expr {$result == 1 && [string match "*Invalid output_size*" $msg]}
} {1}

# Test 13: Error on invalid output_size - named
test adaptive_maxpool1d-5.5 {Error on invalid output_size - named} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0}
    set tensor [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad false]
    set reshaped [torch::tensor_reshape $tensor {2 3 1}]
    set result [catch {torch::adaptive_maxpool1d -input $reshaped -output_size "invalid"} msg]
    expr {$result == 1 && [string match "*Invalid -output_size*" $msg]}
} {1}

# Test 14: Error on unknown parameter
test adaptive_maxpool1d-5.6 {Error on unknown parameter} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0}
    set tensor [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad false]
    set reshaped [torch::tensor_reshape $tensor {2 3 1}]
    set result [catch {torch::adaptive_maxpool1d -input $reshaped -unknown_param 2} msg]
    expr {$result == 1 && [string match "*Unknown parameter*" $msg]}
} {1}

# Test 15: Error on zero output_size
test adaptive_maxpool1d-5.7 {Error on zero output_size} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0}
    set tensor [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad false]
    set reshaped [torch::tensor_reshape $tensor {2 3 1}]
    set result [catch {torch::adaptive_maxpool1d -input $reshaped -output_size 0} msg]
    expr {$result == 1 && [string match "*Required parameters*" $msg]}
} {1}

# Test 16: Test different output sizes
test adaptive_maxpool1d-6.1 {Test single output size} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0}
    set tensor [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad false]
    set reshaped [torch::tensor_reshape $tensor {2 2 2}]
    set result [torch::adaptive_maxpool1d $reshaped 1]
    
    set shape [torch::tensor_shape $result]
    string equal $shape {2 2 1}
} {1}

# Test 17: Test larger pooling
test adaptive_maxpool1d-6.2 {Test larger pooling} {
    set data {}
    for {set i 1} {$i <= 60} {incr i} {
        lappend data [expr {$i * 1.0}]
    }
    set tensor [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad false]
    set reshaped [torch::tensor_reshape $tensor {5 2 6}]
    set result [torch::adaptive_maxpool1d -input $reshaped -output_size 3]
    
    set shape [torch::tensor_shape $result]
    string equal $shape {5 2 3}
} {1}

cleanupTests 