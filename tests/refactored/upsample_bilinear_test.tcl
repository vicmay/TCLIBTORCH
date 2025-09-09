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

# Helper function to create test tensors for upsampling
proc create_test_tensor {} {
    # Create input tensor: 4D tensor [batch_size, channels, height, width]
    # Shape: [1, 3, 4, 4] - 1 batch, 3 channels, 4x4 spatial
    set input [torch::ones -shape {1 3 4 4} -dtype float32]
    return $input
}

# Test 1: Positional syntax - basic functionality with output size
test upsample_bilinear-1.1 {Basic positional syntax with output size} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear $input {8 8}]
    expr {[string length $result] > 0}
} {1}

# Test 2: Positional syntax - with align_corners
test upsample_bilinear-1.2 {Positional syntax with align_corners} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear $input {8 8} 1]
    expr {[string length $result] > 0}
} {1}

# Test 3: Positional syntax - with align_corners and antialias
test upsample_bilinear-1.3 {Positional syntax with align_corners and antialias} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear $input {8 8} 1 1]
    expr {[string length $result] > 0}
} {1}

# Test 4: Positional syntax - with scale factor
test upsample_bilinear-1.4 {Positional syntax with scale factor} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear $input {2.0 2.0}]
    expr {[string length $result] > 0}
} {1}

# Test 5: Named parameter syntax - with output_size
test upsample_bilinear-2.1 {Named parameter syntax with output_size} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear -input $input -output_size {8 8}]
    expr {[string length $result] > 0}
} {1}

# Test 6: Named parameter syntax - with scale_factor
test upsample_bilinear-2.2 {Named parameter syntax with scale_factor} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear -input $input -scale_factor {2.0 2.0}]
    expr {[string length $result] > 0}
} {1}

# Test 7: Named parameter syntax - all parameters
test upsample_bilinear-2.3 {Named parameter syntax with all parameters} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear -input $input -output_size {8 8} -align_corners 1 -antialias 0]
    expr {[string length $result] > 0}
} {1}

# Test 8: Named parameter syntax - using -size alias
test upsample_bilinear-2.4 {Named parameter syntax using -size alias} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear -input $input -size {8 8}]
    expr {[string length $result] > 0}
} {1}

# Test 9: camelCase alias - basic functionality
test upsample_bilinear-3.1 {camelCase alias basic functionality} {
    set input [create_test_tensor]
    
    set result [torch::upsampleBilinear $input {8 8}]
    expr {[string length $result] > 0}
} {1}

# Test 10: camelCase alias - with named parameters
test upsample_bilinear-3.2 {camelCase alias with named parameters} {
    set input [create_test_tensor]
    
    set result [torch::upsampleBilinear -input $input -scale_factor {2.0 2.0} -align_corners 1]
    expr {[string length $result] > 0}
} {1}

# Test 11: Mathematical correctness - both syntaxes should produce same results
test upsample_bilinear-4.1 {Both syntaxes produce same results} {
    set input [create_test_tensor]
    
    set result1 [torch::upsample_bilinear $input {8 8} 1]
    set result2 [torch::upsample_bilinear -input $input -output_size {8 8} -align_corners 1]
    
    # Check if results have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$shape1 eq $shape2}
} {1}

# Test 12: Output size scaling
test upsample_bilinear-4.2 {Output size scaling verification} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear -input $input -output_size {8 8}]
    set shape [torch::tensor_shape $result]
    
    # Input was [1, 3, 4, 4], output should be [1, 3, 8, 8]
    expr {[lindex $shape 2] == 8 && [lindex $shape 3] == 8}
} {1}

# Test 13: Scale factor scaling
test upsample_bilinear-4.3 {Scale factor scaling verification} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear -input $input -scale_factor {2.0 2.0}]
    set shape [torch::tensor_shape $result]
    
    # Input was [1, 3, 4, 4], with scale 2.0 should be [1, 3, 8, 8]
    expr {[lindex $shape 2] == 8 && [lindex $shape 3] == 8}
} {1}

# Test 14: Asymmetric scaling
test upsample_bilinear-4.4 {Asymmetric scaling with output_size} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear -input $input -output_size {8 16}]
    set shape [torch::tensor_shape $result]
    
    # Output should be [1, 3, 8, 16]
    expr {[lindex $shape 2] == 8 && [lindex $shape 3] == 16}
} {1}

# Test 15: Asymmetric scale factors
test upsample_bilinear-4.5 {Asymmetric scale factors} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear -input $input -scale_factor {2.0 3.0}]
    set shape [torch::tensor_shape $result]
    
    # Input [1, 3, 4, 4] with scale [2.0, 3.0] should be [1, 3, 8, 12]
    expr {[lindex $shape 2] == 8 && [lindex $shape 3] == 12}
} {1}

# Test 16: Align corners effect
test upsample_bilinear-4.6 {Align corners parameter effect} {
    set input [create_test_tensor]
    
    set result1 [torch::upsample_bilinear -input $input -output_size {8 8} -align_corners 0]
    set result2 [torch::upsample_bilinear -input $input -output_size {8 8} -align_corners 1]
    
    # Both should succeed but may produce different results
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

# Test 17: Single dimension upsampling
test upsample_bilinear-4.7 {Single dimension upsampling} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear -input $input -output_size {4 8}]
    set shape [torch::tensor_shape $result]
    
    # Only width should change: [1, 3, 4, 8]
    expr {[lindex $shape 2] == 4 && [lindex $shape 3] == 8}
} {1}

# Test 18: Downsampling with output_size
test upsample_bilinear-4.8 {Downsampling with output_size} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear -input $input -output_size {2 2}]
    set shape [torch::tensor_shape $result]
    
    # Input [1, 3, 4, 4] to output [1, 3, 2, 2]
    expr {[lindex $shape 2] == 2 && [lindex $shape 3] == 2}
} {1}

# Test 19: Fractional scale factors
test upsample_bilinear-4.9 {Fractional scale factors} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear -input $input -scale_factor {0.5 0.5}]
    set shape [torch::tensor_shape $result]
    
    # Input [1, 3, 4, 4] with scale 0.5 should be [1, 3, 2, 2]
    expr {[lindex $shape 2] == 2 && [lindex $shape 3] == 2}
} {1}

# Test 20: Error handling - missing required parameters
test upsample_bilinear-5.1 {Error handling - missing input parameter} {
    set caught 0
    if {[catch {torch::upsample_bilinear -output_size {8 8}} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 21: Error handling - missing size/scale_factor parameters
test upsample_bilinear-5.2 {Error handling - missing size/scale_factor parameters} {
    set input [create_test_tensor]
    
    set caught 0
    if {[catch {torch::upsample_bilinear -input $input} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 22: Error handling - invalid tensor name
test upsample_bilinear-5.3 {Error handling - invalid tensor name} {
    set caught 0
    if {[catch {torch::upsample_bilinear -input invalid_tensor -output_size {8 8}} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 23: Error handling - unknown parameter
test upsample_bilinear-5.4 {Error handling - unknown parameter} {
    set input [create_test_tensor]
    
    set caught 0
    if {[catch {torch::upsample_bilinear -input $input -output_size {8 8} -unknown_param 1} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 24: Error handling - invalid output_size format
test upsample_bilinear-5.5 {Error handling - invalid output_size format} {
    set input [create_test_tensor]
    
    set caught 0
    if {[catch {torch::upsample_bilinear -input $input -output_size invalid} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 25: Error handling - invalid scale_factor format
test upsample_bilinear-5.6 {Error handling - invalid scale_factor format} {
    set input [create_test_tensor]
    
    set caught 0
    if {[catch {torch::upsample_bilinear -input $input -scale_factor invalid} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 26: Error handling - invalid align_corners
test upsample_bilinear-5.7 {Error handling - invalid align_corners} {
    set input [create_test_tensor]
    
    set caught 0
    if {[catch {torch::upsample_bilinear -input $input -output_size {8 8} -align_corners invalid} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 27: Positional error handling - too few arguments
test upsample_bilinear-5.8 {Error handling - too few positional arguments} {
    set caught 0
    if {[catch {torch::upsample_bilinear input1} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 28: Data type compatibility
test upsample_bilinear-6.1 {Different input data types} {
    # Create float32 tensor
    set input [torch::ones -shape {1 2 4 4} -dtype float32]
    
    set result [torch::upsample_bilinear $input {8 8}]
    expr {[string length $result] > 0}
} {1}

# Test 29: Different tensor shapes
test upsample_bilinear-6.2 {Different input tensor shapes} {
    # Create larger tensor
    set input [torch::ones -shape {2 4 8 8} -dtype float32]
    
    set result [torch::upsample_bilinear -input $input -scale_factor {1.5 1.5}]
    set shape [torch::tensor_shape $result]
    
    # Input [2, 4, 8, 8] with scale 1.5 should be [2, 4, 12, 12]
    expr {[lindex $shape 0] == 2 && [lindex $shape 1] == 4 && [lindex $shape 2] == 12 && [lindex $shape 3] == 12}
} {1}

# Test 30: Edge case - same size upsampling
test upsample_bilinear-6.3 {Edge case - same size upsampling} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear -input $input -output_size {4 4}]
    set shape [torch::tensor_shape $result]
    
    # Output should be same size as input
    expr {[lindex $shape 2] == 4 && [lindex $shape 3] == 4}
} {1}

# Test 31: Edge case - scale factor 1.0
test upsample_bilinear-6.4 {Edge case - scale factor 1.0} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear -input $input -scale_factor {1.0 1.0}]
    set shape [torch::tensor_shape $result]
    
    # Output should be same size as input
    expr {[lindex $shape 2] == 4 && [lindex $shape 3] == 4}
} {1}

# Test 32: Error handling - 3D tensor (invalid for bilinear)
test upsample_bilinear-6.5 {Error handling - 3D tensor (bilinear needs 4D)} {
    # Create 3D tensor [channels, height, width] - invalid for bilinear
    set input [torch::ones -shape {3 4 4} -dtype float32]
    
    set caught 0
    if {[catch {torch::upsample_bilinear -input $input -output_size {8 8}} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 33: Large scale factors
test upsample_bilinear-6.6 {Large scale factors} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear -input $input -scale_factor {4.0 4.0}]
    set shape [torch::tensor_shape $result]
    
    # Input [1, 3, 4, 4] with scale 4.0 should be [1, 3, 16, 16]
    expr {[lindex $shape 2] == 16 && [lindex $shape 3] == 16}
} {1}

# Test 34: Mixed parameter types - test auto-detection
test upsample_bilinear-6.7 {Auto-detection of size vs scale_factor} {
    set input [create_test_tensor]
    
    # This should be detected as output_size (integers)
    set result1 [torch::upsample_bilinear $input {8 8}]
    
    # This should be detected as scale_factor (doubles)
    set result2 [torch::upsample_bilinear $input {2.0 2.0}]
    
    # Both should work and produce the same result
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$shape1 eq $shape2}
} {1}

# Test 35: Backward compatibility with old syntax
test upsample_bilinear-7.1 {Backward compatibility - old syntax} {
    set input [create_test_tensor]
    
    # Test the old 3-argument syntax that was supported before
    set result [torch::upsample_bilinear $input {8 8} 1]
    expr {[string length $result] > 0}
} {1}

# Test 36: Complex upsampling scenario
test upsample_bilinear-7.2 {Complex upsampling scenario} {
    set input [create_test_tensor]
    
    set result [torch::upsample_bilinear -input $input -output_size {6 10} -align_corners 1 -antialias 0]
    set shape [torch::tensor_shape $result]
    
    # Verify complex target size
    expr {[lindex $shape 2] == 6 && [lindex $shape 3] == 10}
} {1}

cleanupTests 