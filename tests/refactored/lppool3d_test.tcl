#!/usr/bin/env tclsh

package require tcltest
namespace import tcltest::*

# Load the extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Configure test settings
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# ============================================================================
# LP POOL 3D TESTS - POSITIONAL SYNTAX (BACKWARD COMPATIBILITY)
# ============================================================================

test lppool3d-1.1 {Basic 3D LP pooling - positional syntax} -body {
    set input [torch::zeros {1 1 2 2 2} float32]
    set result [torch::lppool3d $input 2.0 2]
    # Check that we get a tensor back
    string match "tensor*" $result
} -result {1}

test lppool3d-1.2 {L1 norm LP pooling - positional syntax} -body {
    set input [torch::zeros {1 1 2 2 2} float32]
    set result [torch::lppool3d $input 1.0 2]
    string match "tensor*" $result
} -result {1}

test lppool3d-1.3 {Custom norm LP pooling - positional syntax} -body {
    set input [torch::zeros {1 1 2 2 2} float32]
    set result [torch::lppool3d $input 3.0 2]
    string match "tensor*" $result
} -result {1}

# ============================================================================
# LP POOL 3D TESTS - NAMED PARAMETER SYNTAX
# ============================================================================

test lppool3d-2.1 {Named parameter syntax - basic} -body {
    set input [torch::zeros {1 1 2 2 2} float32]
    set result [torch::lppool3d -input $input -normType 2.0 -kernelSize 2]
    string match "tensor*" $result
} -result {1}

test lppool3d-2.2 {Named parameters - all options} -body {
    set input [torch::zeros {1 1 2 4 4} float32]
    set result [torch::lppool3d -input $input -normType 1.0 -kernelSize {2 2 2} -stride {1 1 1} -ceilMode 0]
    string match "tensor*" $result
} -result {1}

test lppool3d-2.3 {Named parameters - flexible order} -body {
    set input [torch::zeros {1 1 2 2 2} float32]
    set result [torch::lppool3d -kernelSize 2 -normType 3.0 -input $input]
    string match "tensor*" $result
} -result {1}

test lppool3d-2.4 {Named parameters - alternative parameter names} -body {
    set input [torch::zeros {1 1 2 2 2} float32]
    set result [torch::lppool3d -tensor $input -norm_type 2.0 -kernel_size 2 -ceil_mode 0]
    string match "tensor*" $result
} -result {1}

# ============================================================================
# LP POOL 3D TESTS - CAMELCASE ALIAS
# ============================================================================

test lppool3d-3.1 {CamelCase alias - lpPool3d} -body {
    set input [torch::zeros {1 1 2 2 2} float32]
    set result [torch::lpPool3d -input $input -normType 2.0 -kernelSize 2]
    string match "tensor*" $result
} -result {1}

test lppool3d-3.2 {CamelCase alias with positional syntax} -body {
    set input [torch::zeros {1 1 2 2 2} float32]
    set result [torch::lpPool3d $input 2.0 2]
    string match "tensor*" $result
} -result {1}

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

test lppool3d-4.1 {Error - missing arguments positional} -body {
    set input [torch::zeros {1 1 2 2 2} float32]
    torch::lppool3d $input
} -returnCodes error -match glob -result {*argument*}

test lppool3d-4.2 {Error - missing required parameter} -body {
    set input [torch::zeros {1 1 2 2 2} float32]
    torch::lppool3d -input $input
} -returnCodes error -match glob -result {*parameter*}

test lppool3d-4.3 {Error - invalid tensor name} -body {
    torch::lppool3d nonexistent_tensor 2.0 2
} -returnCodes error -match glob -result {*Invalid input tensor*}

test lppool3d-4.4 {Error - invalid norm type} -body {
    set input [torch::zeros {1 1 2 2 2} float32]
    torch::lppool3d $input invalid_norm 2
} -returnCodes error -match glob -result {*norm*}

test lppool3d-4.5 {Error - zero norm type} -body {
    set input [torch::zeros {1 1 2 2 2} float32]
    torch::lppool3d -input $input -normType 0.0 -kernelSize 2
} -returnCodes error -match glob -result {*parameter*}

test lppool3d-4.6 {Error - invalid kernel size} -body {
    set input [torch::zeros {1 1 2 2 2} float32]
    torch::lppool3d -input $input -normType 2.0 -kernelSize 0
} -returnCodes error -match glob -result {*parameter*}

test lppool3d-4.7 {Error - unknown parameter} -body {
    set input [torch::zeros {1 1 2 2 2} float32]
    torch::lppool3d -input $input -normType 2.0 -kernelSize 2 -unknown invalid
} -returnCodes error -match glob -result {*Unknown parameter*}

# ============================================================================
# MATHEMATICAL CORRECTNESS TESTS
# ============================================================================

test lppool3d-5.1 {Verify output shape calculation} -body {
    # Input: 1x1x4x4x4, kernel: 2x2x2, stride: 2x2x2
    set input [torch::zeros {1 1 4 4 4} float32]
    set result [torch::lppool3d $input 2.0 2 2]
    
    # Expected output shape: 1x1x2x2x2 (4/2 = 2 for each dimension)
    torch::tensor_shape $result
} -result {1 1 2 2 2}

test lppool3d-5.2 {Verify stride effects} -body {
    # Input: 1x1x3x3x3, kernel: 2x2x2, stride: 1x1x1
    set input [torch::zeros {1 1 3 3 3} float32]
    set result [torch::lppool3d $input 2.0 2 1]
    
    # Expected output shape: 1x1x2x2x2 (3-2+1 = 2 for each dimension)
    torch::tensor_shape $result
} -result {1 1 2 2 2}

test lppool3d-5.3 {Multi-channel input} -body {
    # 2 channels input
    set input [torch::zeros {1 2 2 2 2} float32]
    set result [torch::lppool3d $input 2.0 2]
    
    # Should preserve channel dimension
    torch::tensor_shape $result
} -result {1 2 1 1 1}

test lppool3d-5.4 {Batch processing} -body {
    # Batch of 2 samples
    set input [torch::zeros {2 1 2 2 2} float32]
    set result [torch::lppool3d $input 2.0 2]
    
    # Should preserve batch dimension
    torch::tensor_shape $result
} -result {2 1 1 1 1}

# ============================================================================
# SYNTAX CONSISTENCY TESTS
# ============================================================================

test lppool3d-6.1 {Positional and named syntax equivalence} -body {
    set input [torch::zeros {1 1 2 2 2} float32]
    
    set result_pos [torch::lppool3d $input 2.0 2]
    set result_named [torch::lppool3d -input $input -normType 2.0 -kernelSize 2]
    
    # Should produce same shape
    set shape_pos [torch::tensor_shape $result_pos]
    set shape_named [torch::tensor_shape $result_named]
    
    expr {$shape_pos == $shape_named}
} -result {1}

test lppool3d-6.2 {Snake_case and camelCase equivalence} -body {
    set input [torch::zeros {1 1 2 2 2} float32]
    
    set result_snake [torch::lppool3d -input $input -normType 2.0 -kernelSize 2]
    set result_camel [torch::lpPool3d -input $input -normType 2.0 -kernelSize 2]
    
    # Should produce same shape
    set shape_snake [torch::tensor_shape $result_snake]
    set shape_camel [torch::tensor_shape $result_camel]
    
    expr {$shape_snake == $shape_camel}
} -result {1}

test lppool3d-6.3 {Alternative parameter names equivalence} -body {
    set input [torch::zeros {1 1 2 2 2} float32]
    
    set result1 [torch::lppool3d -input $input -normType 2.0 -kernelSize 2]
    set result2 [torch::lppool3d -tensor $input -norm_type 2.0 -kernel_size 2]
    
    # Should produce same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$shape1 == $shape2}
} -result {1}

# Clean up and run tests
cleanupTests
