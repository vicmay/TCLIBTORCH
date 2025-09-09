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
# Test torch::interpolate - Tensor Interpolation
# ============================================================================

# Test 1: Basic functionality with positional syntax
test interpolate-1.1 {Basic positional syntax - nearest mode} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result [torch::interpolate $tensor {8 8} nearest]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 8 8}

test interpolate-1.2 {Positional syntax - bilinear mode} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result [torch::interpolate $tensor {8 8} bilinear]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 8 8}

test interpolate-1.3 {Positional syntax with align_corners} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result [torch::interpolate $tensor {8 8} bilinear 1]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 8 8}

# Test 2: Named parameter syntax
test interpolate-2.1 {Named parameter syntax - basic} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result [torch::interpolate -input $tensor -size {8 8}]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 8 8}

test interpolate-2.2 {Named parameter syntax - with mode} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result [torch::interpolate -input $tensor -size {8 8} -mode bilinear]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 8 8}

test interpolate-2.3 {Named parameter syntax - tensor parameter} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result [torch::interpolate -tensor $tensor -size {8 8}]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 8 8}

test interpolate-2.4 {Named parameter syntax - align_corners} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result [torch::interpolate -input $tensor -size {8 8} -mode bilinear -align_corners 1]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 8 8}

test interpolate-2.5 {Named parameter syntax - alignCorners camelCase} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result [torch::interpolate -input $tensor -size {8 8} -mode bilinear -alignCorners 1]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 8 8}

# Test 3: Scale factor instead of size
test interpolate-3.1 {Named parameter syntax - scale_factor} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result [torch::interpolate -input $tensor -scale_factor {2.0 2.0}]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 8 8}

test interpolate-3.2 {Named parameter syntax - scaleFactor camelCase} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result [torch::interpolate -input $tensor -scaleFactor {2.0 2.0}]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 8 8}

test interpolate-3.3 {Positional syntax with scale_factor} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result [torch::interpolate $tensor {8 8} nearest 0 {2.0 2.0}]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 8 8}

# Test 4: Different interpolation modes
test interpolate-4.1 {Mode - nearest} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result [torch::interpolate -input $tensor -size {8 8} -mode nearest]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 8 8}

test interpolate-4.2 {Mode - bilinear} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result [torch::interpolate -input $tensor -size {8 8} -mode bilinear]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 8 8}

test interpolate-4.3 {Mode - area} {
    set tensor [torch::ones -shape {1 1 8 8}]
    set result [torch::interpolate -input $tensor -size {4 4} -mode area]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 4 4}

# Test 5: Different tensor shapes
test interpolate-5.1 {1D interpolation} {
    set tensor [torch::ones -shape {1 1 4}]
    set result [torch::interpolate -input $tensor -size {8}]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 8}

test interpolate-5.2 {3D interpolation} {
    set tensor [torch::ones -shape {1 1 4 4 4}]
    set result [torch::interpolate -input $tensor -size {8 8 8} -mode trilinear]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 8 8 8}

test interpolate-5.3 {Batch dimension} {
    set tensor [torch::ones -shape {2 3 4 4}]
    set result [torch::interpolate -input $tensor -size {8 8}]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 8 8}

# Test 6: Error handling
test interpolate-6.1 {Error handling - missing input} {
    catch {torch::interpolate} msg
    string match "*Required parameters missing*" $msg
} 1

test interpolate-6.2 {Error handling - invalid tensor handle} {
    catch {torch::interpolate invalid_tensor {8 8}} msg
    string match "*Invalid input tensor*" $msg
} 1

test interpolate-6.3 {Error handling - unknown parameter} {
    set tensor [torch::ones -shape {1 1 4 4}]
    catch {torch::interpolate -input $tensor -size {8 8} -unknown_param value} msg
    string match "*Unknown parameter*" $msg
} 1

test interpolate-6.4 {Error handling - missing parameter value} {
    catch {torch::interpolate -input} msg
    string match "*Named parameters must come in pairs*" $msg
} 1

test interpolate-6.5 {Error handling - invalid mode} {
    set tensor [torch::ones -shape {1 1 4 4}]
    catch {torch::interpolate -input $tensor -size {8 8} -mode invalid_mode} msg
    string match "*Invalid mode*" $msg
} 1

test interpolate-6.6 {Error handling - missing size and scale_factor} {
    set tensor [torch::ones -shape {1 1 4 4}]
    catch {torch::interpolate -input $tensor} msg
    string match "*Required parameters missing*" $msg
} 1

# Test 7: Consistency between syntaxes
test interpolate-7.1 {Consistency - positional vs named basic} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result1 [torch::interpolate $tensor {8 8}]
    set result2 [torch::interpolate -input $tensor -size {8 8}]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

test interpolate-7.2 {Consistency - with mode} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result1 [torch::interpolate $tensor {8 8} bilinear]
    set result2 [torch::interpolate -input $tensor -size {8 8} -mode bilinear]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

test interpolate-7.3 {Consistency - with align_corners} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result1 [torch::interpolate $tensor {8 8} bilinear 1]
    set result2 [torch::interpolate -input $tensor -size {8 8} -mode bilinear -align_corners 1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

# Test 8: Different size combinations
test interpolate-8.1 {Single dimension size} {
    set tensor [torch::ones -shape {1 1 4}]
    set result [torch::interpolate -input $tensor -size {8}]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 8}

test interpolate-8.2 {Multiple dimension sizes} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result [torch::interpolate -input $tensor -size {6 8}]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 6 8}

test interpolate-8.3 {3D size} {
    set tensor [torch::ones -shape {1 1 4 4 4}]
    set result [torch::interpolate -input $tensor -size {6 8 10} -mode trilinear]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 6 8 10}

# Test 9: Scale factor variations
test interpolate-9.1 {Single scale factor} {
    set tensor [torch::ones -shape {1 1 4}]
    set result [torch::interpolate -input $tensor -scale_factor {2.0}]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 8}

test interpolate-9.2 {Multiple scale factors} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result [torch::interpolate -input $tensor -scale_factor {1.5 2.0}]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 6 8}

test interpolate-9.3 {Fractional scale factors} {
    set tensor [torch::ones -shape {1 1 8 8}]
    set result [torch::interpolate -input $tensor -scale_factor {0.5 0.5}]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 4 4}

# Test 10: Edge cases and special values
test interpolate-10.1 {Same size (no change)} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result [torch::interpolate -input $tensor -size {4 4}]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 4 4}

test interpolate-10.2 {Scale factor 1.0 (no change)} {
    set tensor [torch::ones -shape {1 1 4 4}]
    set result [torch::interpolate -input $tensor -scale_factor {1.0 1.0}]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 4 4}

test interpolate-10.3 {Large upscaling} {
    set tensor [torch::ones -shape {1 1 2 2}]
    set result [torch::interpolate -input $tensor -size {16 16}]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 16 16}

test interpolate-10.4 {Large downscaling} {
    set tensor [torch::ones -shape {1 1 16 16}]
    set result [torch::interpolate -input $tensor -size {2 2}]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 2 2}

cleanupTests 