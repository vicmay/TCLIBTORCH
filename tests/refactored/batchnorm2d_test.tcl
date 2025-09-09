#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension (relative to this script)
if {[catch {load ../../build/libtorchtcl.so} err]} {
    puts "Failed to load libtorchtcl.so: $err"
    exit 1
}

configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Create input tensors
proc create_input {} {
    return [torch::ones {1 4 6 6}] ;# N,C,H,W
}

proc makeInput {} {
    # NCHW 2x64x8x8 filled with ones
    return [torch::ones -shape {2 64 8 8} -dtype float32]
}

# Test 1: Basic positional syntax

test batchnorm2d-1.1 {positional create basic} {
    set bn [torch::batchnorm2d 4]
    set x [create_input]
    set y [torch::layer_forward $bn $x]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 4 6 6"}
} {1}

test batchnorm2d-1.2 {positional with eps} {
    set bn [torch::batchnorm2d 4 0.001]
    set x [create_input]
    set y [torch::layer_forward $bn $x]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 4 6 6"}
} {1}

test batchnorm2d-1.3 {positional with eps and momentum} {
    set bn [torch::batchnorm2d 4 0.001 0.2]
    set x [create_input]
    set y [torch::layer_forward $bn $x]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 4 6 6"}
} {1}

test batchnorm2d-1.4 {positional with all parameters} {
    set bn [torch::batchnorm2d 4 0.001 0.2 0 0]
    set x [create_input]
    set y [torch::layer_forward $bn $x]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 4 6 6"}
} {1}

# Test 2: Named parameter syntax

test batchnorm2d-2.1 {named create basic} {
    set bn [torch::batchnorm2d -numFeatures 4]
    set x [create_input]
    set y [torch::layer_forward $bn $x]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 4 6 6"}
} {1}

test batchnorm2d-2.2 {named with eps} {
    set bn [torch::batchnorm2d -numFeatures 4 -eps 0.001]
    set x [create_input]
    set y [torch::layer_forward $bn $x]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 4 6 6"}
} {1}

test batchnorm2d-2.3 {named with momentum} {
    set bn [torch::batchnorm2d -numFeatures 4 -momentum 0.2]
    set x [create_input]
    set y [torch::layer_forward $bn $x]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 4 6 6"}
} {1}

test batchnorm2d-2.4 {named with affine} {
    set bn [torch::batchnorm2d -numFeatures 4 -affine false]
    set x [create_input]
    set y [torch::layer_forward $bn $x]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 4 6 6"}
} {1}

test batchnorm2d-2.5 {named with trackRunningStats} {
    set bn [torch::batchnorm2d -numFeatures 4 -trackRunningStats false]
    set x [create_input]
    set y [torch::layer_forward $bn $x]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 4 6 6"}
} {1}

test batchnorm2d-2.6 {named with all parameters} {
    set bn [torch::batchnorm2d -numFeatures 4 -eps 0.001 -momentum 0.2 -affine false -trackRunningStats false]
    set x [create_input]
    set y [torch::layer_forward $bn $x]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 4 6 6"}
} {1}

test batchnorm2d-2.7 {named with snake_case parameters} {
    set bn [torch::batchnorm2d -num_features 4 -track_running_stats false]
    set x [create_input]
    set y [torch::layer_forward $bn $x]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 4 6 6"}
} {1}

# Test 3: camelCase alias

test batchnorm2d-3.1 {camelCase alias basic} {
    set bn [torch::batchNorm2d -numFeatures 4]
    set x [create_input]
    set y [torch::layer_forward $bn $x]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 4 6 6"}
} {1}

test batchnorm2d-3.2 {camelCase with multiple parameters} {
    set bn [torch::batchNorm2d -numFeatures 4 -eps 0.001 -momentum 0.2]
    set x [create_input]
    set y [torch::layer_forward $bn $x]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 4 6 6"}
} {1}

# Test 4: Mathematical correctness

test batchnorm2d-4.1 {mathematical correctness - shape preservation} {
    # Different shape to test generalization
    set x [torch::ones {2 3 4 5}]
    set bn [torch::batchnorm2d 3]
    set y [torch::layer_forward $bn $x]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "2 3 4 5"}
} {1}

# Test 5: Consistency between syntaxes

test batchnorm2d-5.1 {consistency - positional vs named} {
    set x [create_input]
    set bn1 [torch::batchnorm2d 4 0.001 0.2]
    set bn2 [torch::batchnorm2d -numFeatures 4 -eps 0.001 -momentum 0.2]
    set y1 [torch::layer_forward $bn1 $x]
    set y2 [torch::layer_forward $bn2 $x]
    set shape1 [torch::tensor_shape $y1]
    set shape2 [torch::tensor_shape $y2]
    expr {$shape1 eq $shape2}
} {1}

test batchnorm2d-5.2 {consistency - snake_case vs camelCase} {
    set x [create_input]
    set bn1 [torch::batchnorm2d -numFeatures 4]
    set bn2 [torch::batchNorm2d -numFeatures 4]
    set y1 [torch::layer_forward $bn1 $x]
    set y2 [torch::layer_forward $bn2 $x]
    set shape1 [torch::tensor_shape $y1]
    set shape2 [torch::tensor_shape $y2]
    expr {$shape1 eq $shape2}
} {1}

# Test 6: Error handling

test batchnorm2d-6.1 {missing numFeatures} -body {
    torch::batchnorm2d
} -returnCodes error -match glob -result *numFeatures*

test batchnorm2d-6.2 {invalid numFeatures} -body {
    torch::batchnorm2d -numFeatures -1
} -returnCodes error -match glob -result *numFeatures*

test batchnorm2d-6.3 {invalid eps} -body {
    torch::batchnorm2d -numFeatures 4 -eps invalid
} -returnCodes error -match glob -result *eps*

test batchnorm2d-6.4 {invalid momentum} -body {
    torch::batchnorm2d -numFeatures 4 -momentum invalid
} -returnCodes error -match glob -result *momentum*

test batchnorm2d-6.5 {unknown parameter} -body {
    torch::batchnorm2d -numFeatures 4 -unknown value
} -returnCodes error -match glob -result *Unknown*parameter*

# Test 7: Positional syntax with new input

set bn1 [torch::batchnorm2d 64]
set x1  [makeInput]
set y1  [torch::layer_forward $bn1 $x1]

test batchnorm2d-7.1 {Positional returns tensor} {
    expr {[string match "tensor*" $y1]}
} {1}

set s1 [torch::tensor_shape $y1]

test batchnorm2d-7.2 {Output shape matches input} {
    set sx [torch::tensor_shape $x1]
    expr {$s1 eq $sx}
} {1}

# Test 8: Named parameters with new input

set bn2 [torch::batchnorm2d -numFeatures 64 -eps 1e-4 -momentum 0.05 -affine 1 -trackRunningStats 1]
set y2  [torch::layer_forward $bn2 $x1]

test batchnorm2d-8.1 {Named returns tensor} {
    expr {[string match "tensor*" $y2]}
} {1}

# Test 9: camelCase alias with new input

set bn3 [torch::batchNorm2d -numFeatures 64]
set y3  [torch::layer_forward $bn3 $x1]

test batchnorm2d-9.1 {camelCase returns tensor} {
    expr {[string match "tensor*" $y3]}
} {1}

cleanupTests
