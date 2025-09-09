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

# Basic positional (L2 norm)
test lppool2d-1.1 {Basic positional syntax L2 norm} {
    set input [torch::randn -shape {1 4 8 8}]
    set output [torch::lppool2d $input 2.0 3]
    string match "tensor*" $output
} {1}

# Positional with L1 norm
test lppool2d-1.2 {Positional syntax L1 norm} {
    set input [torch::randn -shape {1 4 10 10}]
    set output [torch::lppool2d $input 1.0 2]
    string match "tensor*" $output
} {1}

# Positional with custom stride
test lppool2d-1.3 {Positional syntax with custom stride} {
    set input [torch::randn -shape {1 4 12 12}]
    set output [torch::lppool2d $input 2.0 3 2]
    string match "tensor*" $output
} {1}

# Positional with kernel list
test lppool2d-1.4 {Positional syntax with kernel list} {
    set input [torch::randn -shape {1 4 10 12}]
    set output [torch::lppool2d $input 2.0 {3 2}]
    string match "tensor*" $output
} {1}

# Positional with stride list
test lppool2d-1.5 {Positional syntax with stride list} {
    set input [torch::randn -shape {1 4 12 12}]
    set output [torch::lppool2d $input 2.0 {3 3} {2 2}]
    string match "tensor*" $output
} {1}

# Positional with ceil mode
test lppool2d-1.6 {Positional syntax with ceil mode} {
    set input [torch::randn -shape {1 4 10 10}]
    set output [torch::lppool2d $input 2.0 3 2 true]
    string match "tensor*" $output
} {1}

# -----------------------------------------------------------------------------
# 2. Named parameter syntax tests
# -----------------------------------------------------------------------------

# Basic named parameters
test lppool2d-2.1 {Named syntax basic} {
    set input [torch::randn -shape {1 4 8 8}]
    set output [torch::lppool2d -input $input -normType 2.0 -kernelSize 3]
    string match "tensor*" $output
} {1}

# Named with all parameters
test lppool2d-2.2 {Named syntax with all parameters} {
    set input [torch::randn -shape {1 4 10 10}]
    set output [torch::lppool2d -input $input -normType 1.0 -kernelSize 2 -stride 1 -ceilMode true]
    string match "tensor*" $output
} {1}

# Named with mixed parameter order
test lppool2d-2.3 {Named syntax mixed order} {
    set input [torch::randn -shape {1 4 12 12}]
    set output [torch::lppool2d -kernelSize 3 -input $input -normType 2.0 -ceilMode false]
    string match "tensor*" $output
} {1}

# Named with kernel size list
test lppool2d-2.4 {Named syntax with kernel size list} {
    set input [torch::randn -shape {1 4 10 12}]
    set output [torch::lppool2d -input $input -normType 2.0 -kernelSize {3 2}]
    string match "tensor*" $output
} {1}

# Named with stride list
test lppool2d-2.5 {Named syntax with stride list} {
    set input [torch::randn -shape {1 4 12 12}]
    set output [torch::lppool2d -input $input -normType 2.0 -kernelSize 3 -stride {2 2}]
    string match "tensor*" $output
} {1}

# Named with alternative parameter names
test lppool2d-2.6 {Named syntax with alternative parameter names} {
    set input [torch::randn -shape {1 4 8 8}]
    set output [torch::lppool2d -tensor $input -norm_type 2.0 -kernel_size 3 -ceil_mode false]
    string match "tensor*" $output
} {1}

# -----------------------------------------------------------------------------
# 3. CamelCase alias tests (torch::lpPool2d)
# -----------------------------------------------------------------------------

# Basic alias
test lppool2d-3.1 {CamelCase alias basic} {
    set input [torch::randn -shape {1 4 8 8}]
    set output [torch::lpPool2d $input 2.0 3]
    string match "tensor*" $output
} {1}

# Alias with named parameters
test lppool2d-3.2 {CamelCase alias with named params} {
    set input [torch::randn -shape {1 4 10 10}]
    set output [torch::lpPool2d -input $input -normType 1.0 -kernelSize 2 -stride 1]
    string match "tensor*" $output
} {1}

# Syntax consistency check
test lppool2d-3.3 {Syntax consistency positional vs named vs alias} {
    set input [torch::randn -shape {1 4 8 8}]
    
    set out1 [torch::lppool2d $input 2.0 2 2 false]
    set out2 [torch::lppool2d -input $input -normType 2.0 -kernelSize 2 -stride 2 -ceilMode false]
    set out3 [torch::lpPool2d -input $input -normType 2.0 -kernelSize 2 -stride 2 -ceilMode false]
    
    set r1 [string match "tensor*" $out1]
    set r2 [string match "tensor*" $out2]
    set r3 [string match "tensor*" $out3]
    
    expr {$r1 && $r2 && $r3}
} {1}

# -----------------------------------------------------------------------------
# 4. Error handling tests
# -----------------------------------------------------------------------------

# Missing input
test lppool2d-4.1 {Error: missing input} {
    set code [catch {torch::lppool2d -normType 2.0 -kernelSize 3} msg]
    list $code [string match "*input*" $msg]
} {1 1}

# Missing kernelSize
test lppool2d-4.2 {Error: missing kernelSize} {
    set input [torch::randn -shape {1 4 8 8}]
    set code [catch {torch::lppool2d -input $input -normType 2.0} msg]
    list $code [string match "*kernelSize*" $msg]
} {1 1}

# Invalid normType (negative)
test lppool2d-4.3 {Error: negative normType} {
    set input [torch::randn -shape {1 4 8 8}]
    set code [catch {torch::lppool2d -input $input -normType -1.0 -kernelSize 3} msg]
    list $code [string match "*normType*" $msg]
} {1 1}

# Invalid kernelSize (zero)
test lppool2d-4.4 {Error: zero kernelSize} {
    set input [torch::randn -shape {1 4 8 8}]
    set code [catch {torch::lppool2d -input $input -normType 2.0 -kernelSize 0} msg]
    list $code [string match "*kernelSize*" $msg]
} {1 1}

# Invalid tensor name
test lppool2d-4.5 {Error: invalid tensor name} {
    set code [catch {torch::lppool2d -input invalid_tensor -normType 2.0 -kernelSize 3} msg]
    list $code [string match "*Invalid input tensor*" $msg]
} {1 1}

# Unknown parameter
test lppool2d-4.6 {Error: unknown parameter} {
    set input [torch::randn -shape {1 4 8 8}]
    set code [catch {torch::lppool2d -input $input -normType 2.0 -kernelSize 3 -invalid_param 1} msg]
    list $code [string match "*Unknown parameter*" $msg]
} {1 1}

# Missing parameter value
test lppool2d-4.7 {Error: missing parameter value} {
    set input [torch::randn -shape {1 4 8 8}]
    set code [catch {torch::lppool2d -input $input -normType 2.0 -kernelSize} msg]
    list $code [string match "*Missing value*" $msg]
} {1 1}

# Too few positional arguments
test lppool2d-4.8 {Error: too few positional arguments} {
    set input [torch::randn -shape {1 4 8 8}]
    set code [catch {torch::lppool2d $input 2.0} msg]
    list $code [string match "*Usage*" $msg]
} {1 1}

# Too many positional arguments
test lppool2d-4.9 {Error: too many positional arguments} {
    set input [torch::randn -shape {1 4 8 8}]
    set code [catch {torch::lppool2d $input 2.0 3 2 false extra} msg]
    list $code [string match "*Usage*" $msg]
} {1 1}

# Invalid kernel size list length
test lppool2d-4.10 {Error: invalid kernel size list length} {
    set input [torch::randn -shape {1 4 8 8}]
    set code [catch {torch::lppool2d -input $input -normType 2.0 -kernelSize {3 2 1}} msg]
    list $code [string match "*2*" $msg]
} {1 1}

# -----------------------------------------------------------------------------
# 5. Functionality tests
# -----------------------------------------------------------------------------

# Output shape verification
test lppool2d-5.1 {Output shape verification} {
    set input [torch::randn -shape {2 4 8 8}]
    set output [torch::lppool2d -input $input -normType 2.0 -kernelSize 2 -stride 2]
    set shape [torch::tensor_shape $output]
    set expected {2 4 4 4}
    expr {$shape == $expected}
} {1}

# Non-square kernels
test lppool2d-5.2 {Non-square kernels} {
    set input [torch::randn -shape {1 3 10 12}]
    set output [torch::lppool2d -input $input -normType 2.0 -kernelSize {3 2}]
    string match "tensor*" $output
} {1}

# Different norm types
test lppool2d-5.3 {L1 norm pooling} {
    set input [torch::randn -shape {1 3 6 6}]
    set output [torch::lppool2d -input $input -normType 1.0 -kernelSize 2]
    string match "tensor*" $output
} {1}

# Large norm type (approximates max pooling)
test lppool2d-5.4 {Large norm type} {
    set input [torch::randn -shape {1 3 6 6}]
    set output [torch::lppool2d -input $input -normType 100.0 -kernelSize 2]
    string match "tensor*" $output
} {1}

# Custom stride effects
test lppool2d-5.5 {Custom stride effects on output size} {
    set input [torch::randn -shape {1 4 12 12}]
    set output1 [torch::lppool2d -input $input -normType 2.0 -kernelSize 3 -stride 1]
    set output2 [torch::lppool2d -input $input -normType 2.0 -kernelSize 3 -stride 3]
    
    set shape1 [torch::tensor_shape $output1]
    set shape2 [torch::tensor_shape $output2]
    
    # Stride 1 should give larger output than stride 3
    set h1 [lindex $shape1 2]
    set w1 [lindex $shape1 3]
    set h2 [lindex $shape2 2]
    set w2 [lindex $shape2 3]
    
    expr {($h1 > $h2) && ($w1 > $w2)}
} {1}

# Non-square stride
test lppool2d-5.6 {Non-square stride} {
    set input [torch::randn -shape {1 4 12 16}]
    set output [torch::lppool2d -input $input -normType 2.0 -kernelSize {3 4} -stride {2 3}]
    string match "tensor*" $output
} {1}

# -----------------------------------------------------------------------------
# 6. Mathematical correctness tests
# -----------------------------------------------------------------------------

# L2 norm behavior
test lppool2d-6.1 {L2 norm pooling produces valid output} {
    set input [torch::ones -shape {1 1 4 4}]
    set output [torch::lppool2d -input $input -normType 2.0 -kernelSize 2 -stride 2]
    string match "tensor*" $output
} {1}

# L1 norm behavior
test lppool2d-6.2 {L1 norm pooling produces valid output} {
    set input [torch::ones -shape {1 1 4 4}]
    set output [torch::lppool2d -input $input -normType 1.0 -kernelSize 2 -stride 2]
    string match "tensor*" $output
} {1}

# Ceil mode effect
test lppool2d-6.3 {Ceil mode effect on output size} {
    set input [torch::randn -shape {1 4 9 9}]
    set output_floor [torch::lppool2d -input $input -normType 2.0 -kernelSize 3 -stride 2 -ceilMode false]
    set output_ceil [torch::lppool2d -input $input -normType 2.0 -kernelSize 3 -stride 2 -ceilMode true]
    
    set shape_floor [torch::tensor_shape $output_floor]
    set shape_ceil [torch::tensor_shape $output_ceil]
    
    # Ceil mode might give larger or equal output size
    set h_floor [lindex $shape_floor 2]
    set w_floor [lindex $shape_floor 3]
    set h_ceil [lindex $shape_ceil 2]
    set w_ceil [lindex $shape_ceil 3]
    
    expr {($h_ceil >= $h_floor) && ($w_ceil >= $w_floor)}
} {1}

# -----------------------------------------------------------------------------
# 7. Edge cases and boundary conditions
# -----------------------------------------------------------------------------

# Minimum input size
test lppool2d-7.1 {Minimum input size} {
    set input [torch::randn -shape {1 1 3 3}]
    set output [torch::lppool2d -input $input -normType 2.0 -kernelSize 1]
    string match "tensor*" $output
} {1}

# Single channel
test lppool2d-7.2 {Single channel input} {
    set input [torch::randn -shape {1 1 8 8}]
    set output [torch::lppool2d -input $input -normType 2.0 -kernelSize 2]
    string match "tensor*" $output
} {1}

# Multiple batch dimensions
test lppool2d-7.3 {Multiple batch dimensions} {
    set input [torch::randn -shape {3 2 8 8}]
    set output [torch::lppool2d -input $input -normType 2.0 -kernelSize 2]
    set shape [torch::tensor_shape $output]
    
    # Should preserve batch and channel dimensions
    expr {[lindex $shape 0] == 3 && [lindex $shape 1] == 2}
} {1}

# Fractional norm type
test lppool2d-7.4 {Fractional norm type} {
    set input [torch::randn -shape {1 4 8 8}]
    set output [torch::lppool2d -input $input -normType 1.5 -kernelSize 2]
    string match "tensor*" $output
} {1}

# Boolean parameters as integers
test lppool2d-7.5 {Boolean parameters as integers} {
    set input [torch::randn -shape {1 4 8 8}]
    set output1 [torch::lppool2d $input 2.0 3 2 1]
    set output2 [torch::lppool2d $input 2.0 3 2 0]
    
    set r1 [string match "tensor*" $output1]
    set r2 [string match "tensor*" $output2]
    
    expr {$r1 && $r2}
} {1}

# Large kernel size
test lppool2d-7.6 {Large kernel size} {
    set input [torch::randn -shape {1 4 20 20}]
    set output [torch::lppool2d -input $input -normType 2.0 -kernelSize 10]
    string match "tensor*" $output
} {1}

# Asymmetric input dimensions
test lppool2d-7.7 {Asymmetric input dimensions} {
    set input [torch::randn -shape {1 4 8 12}]
    set output [torch::lppool2d -input $input -normType 2.0 -kernelSize {2 3}]
    string match "tensor*" $output
} {1}

# Small input with large kernel
test lppool2d-7.8 {Small input with large kernel} {
    set input [torch::randn -shape {1 2 4 4}]
    set output [torch::lppool2d -input $input -normType 2.0 -kernelSize 4]
    string match "tensor*" $output
} {1}

# Different data types behavior
test lppool2d-7.9 {Different norm values} {
    set input [torch::randn -shape {1 4 8 8}]
    
    set output1 [torch::lppool2d -input $input -normType 1.0 -kernelSize 2]
    set output2 [torch::lppool2d -input $input -normType 2.0 -kernelSize 2]
    set output3 [torch::lppool2d -input $input -normType 3.0 -kernelSize 2]
    
    set r1 [string match "tensor*" $output1]
    set r2 [string match "tensor*" $output2]
    set r3 [string match "tensor*" $output3]
    
    expr {$r1 && $r2 && $r3}
} {1}

# Very small norm value
test lppool2d-7.10 {Very small norm value} {
    set input [torch::randn -shape {1 4 8 8}]
    set output [torch::lppool2d -input $input -normType 0.5 -kernelSize 2]
    string match "tensor*" $output
} {1}

# -----------------------------------------------------------------------------
cleanupTests 