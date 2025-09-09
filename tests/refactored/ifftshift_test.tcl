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
# Test torch::ifftshift - Inverse FFT shift
# ============================================================================

# Test 1: Basic functionality with positional syntax
test ifftshift-1.1 {Basic positional syntax - 1D tensor} {
    set tensor [torch::arange -start 1 -end 5 -dtype float32]
    set result [torch::ifftshift $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test ifftshift-1.2 {Basic positional syntax - 2D tensor} {
    set tensor [torch::ones -shape {2 4}]
    set result [torch::ifftshift $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {2 4}

test ifftshift-1.3 {Positional syntax with specific dimension} {
    set tensor [torch::ones -shape {2 4}]
    set result [torch::ifftshift $tensor 1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 4}

# Test 2: Named parameter syntax
test ifftshift-2.1 {Named parameter syntax - basic} {
    set tensor [torch::ones -shape {4}]
    set result [torch::ifftshift -input $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test ifftshift-2.2 {Named parameter syntax - alternative input parameter} {
    set tensor [torch::ones -shape {4}]
    set result [torch::ifftshift -tensor $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test ifftshift-2.3 {Named parameter syntax with dimension} {
    set tensor [torch::ones -shape {2 4}]
    set result [torch::ifftshift -input $tensor -dim 1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 4}

test ifftshift-2.4 {Named parameter syntax with alternative dimension parameter} {
    set tensor [torch::ones -shape {2 4}]
    set result [torch::ifftshift -input $tensor -dimension 0]
    set shape [torch::tensor_shape $result]
    set shape
} {2 4}

# Test 3: CamelCase alias
test ifftshift-3.1 {CamelCase alias - basic functionality} {
    set tensor [torch::ones -shape {4}]
    set result [torch::ifftShift $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test ifftshift-3.2 {CamelCase alias with named parameters} {
    set tensor [torch::ones -shape {2 4}]
    set result [torch::ifftShift -input $tensor -dim 1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 4}

# Test 4: Error handling
test ifftshift-4.1 {Error handling - missing input} {
    catch {torch::ifftshift} msg
    string match "*Required parameters missing*" $msg
} 1

test ifftshift-4.2 {Error handling - invalid dimension} {
    set tensor [torch::ones -shape {4}]
    catch {torch::ifftshift $tensor invalid_dim} msg
    string match "*Invalid dimension*" $msg
} 1

test ifftshift-4.3 {Error handling - unknown parameter} {
    set tensor [torch::ones -shape {4}]
    catch {torch::ifftshift -input $tensor -unknown_param value} msg
    string match "*Unknown parameter*" $msg
} 1

test ifftshift-4.4 {Error handling - missing parameter value} {
    set tensor [torch::ones -shape {4}]
    catch {torch::ifftshift -input $tensor -dim} msg
    string match "*Named parameters must come in pairs*" $msg
} 1

# Test 5: Consistency between syntaxes
test ifftshift-5.1 {Consistency - positional vs named (1D)} {
    set tensor [torch::ones -shape {6}]
    set result1 [torch::ifftshift $tensor]
    set result2 [torch::ifftshift -input $tensor]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

test ifftshift-5.2 {Consistency - positional vs named with dimension} {
    set tensor [torch::ones -shape {2 4}]
    set result1 [torch::ifftshift $tensor 1]
    set result2 [torch::ifftshift -input $tensor -dim 1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

test ifftshift-5.3 {Consistency - snake_case vs camelCase} {
    set tensor [torch::ones -shape {4}]
    set result1 [torch::ifftshift $tensor]
    set result2 [torch::ifftShift $tensor]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

# Test 6: Edge cases
test ifftshift-6.1 {Edge case - single element tensor} {
    set tensor [torch::ones -shape {1}]
    set result [torch::ifftshift $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {1}

test ifftshift-6.2 {Edge case - odd size tensor} {
    set tensor [torch::ones -shape {5}]
    set result [torch::ifftshift $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {5}

test ifftshift-6.3 {Edge case - 3D tensor} {
    set tensor [torch::ones -shape {2 2 2}]
    set result [torch::ifftshift $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2 2}

# Test 7: Different tensor types
test ifftshift-7.1 {Different tensor types - float32} {
    set tensor [torch::ones -shape {4} -dtype float32]
    set result [torch::ifftshift $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test ifftshift-7.2 {Different tensor types - int64} {
    set tensor [torch::ones -shape {4} -dtype int64]
    set result [torch::ifftshift $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

# Test 8: Multiple dimensions
test ifftshift-8.1 {Multiple dimensions - 2D with dimension 0} {
    set tensor [torch::ones -shape {4 6}]
    set result [torch::ifftshift $tensor 0]
    set shape [torch::tensor_shape $result]
    set shape
} {4 6}

test ifftshift-8.2 {Multiple dimensions - 2D with dimension 1} {
    set tensor [torch::ones -shape {4 6}]
    set result [torch::ifftshift $tensor 1]
    set shape [torch::tensor_shape $result]
    set shape
} {4 6}

test ifftshift-8.3 {Multiple dimensions - 3D with dimension 2} {
    set tensor [torch::ones -shape {2 3 4}]
    set result [torch::ifftshift $tensor 2]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 4}

cleanupTests 