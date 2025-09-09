#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

if {[catch {load ../../build/libtorchtcl.so} err]} {
    puts "Failed to load libtorchtcl.so: $err"
    exit 1
}

configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Create input tensors
proc create_input {} {
    return [torch::ones {1 3 6 6 6}] ;# N,C,D,H,W
}

# Test 1: Basic positional syntax

test avgpool3d-1.1 {positional create basic} {
    set x [create_input]
    set y [torch::avgpool3d $x 2]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 3 3 3 3"}
} {1}

test avgpool3d-1.2 {positional with stride} {
    set x [create_input]
    set y [torch::avgpool3d $x 2 1]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 3 5 5 5"}
} {1}

test avgpool3d-1.3 {positional with stride and padding} {
    set x [create_input]
    set y [torch::avgpool3d $x 2 2 1]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 3 4 4 4"}
} {1}

test avgpool3d-1.4 {positional with list kernel_size} {
    set x [create_input]
    set y [torch::avgpool3d $x {2 2 2}]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 3 3 3 3"}
} {1}

test avgpool3d-1.5 {positional with list stride} {
    set x [create_input]
    set y [torch::avgpool3d $x 2 {1 1 1}]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 3 5 5 5"}
} {1}

test avgpool3d-1.6 {positional with list padding} {
    set x [create_input]
    set y [torch::avgpool3d $x 2 2 {1 1 1}]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 3 4 4 4"}
} {1}

# Test 2: Named parameter syntax

test avgpool3d-2.1 {named create basic} {
    set x [create_input]
    set y [torch::avgpool3d -input $x -kernelSize 2]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 3 3 3 3"}
} {1}

test avgpool3d-2.2 {named with stride} {
    set x [create_input]
    set y [torch::avgpool3d -input $x -kernelSize 2 -stride 1]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 3 5 5 5"}
} {1}

test avgpool3d-2.3 {named with padding} {
    set x [create_input]
    set y [torch::avgpool3d -input $x -kernelSize 2 -padding 1]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 3 4 4 4"}
} {1}

test avgpool3d-2.4 {named with all parameters} {
    set x [create_input]
    set y [torch::avgpool3d -input $x -kernelSize 2 -stride 2 -padding 1 -countIncludePad 0]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 3 4 4 4"}
} {1}

test avgpool3d-2.5 {named with list kernelSize} {
    set x [create_input]
    set y [torch::avgpool3d -input $x -kernelSize {2 2 2}]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 3 3 3 3"}
} {1}

test avgpool3d-2.6 {named with list stride} {
    set x [create_input]
    set y [torch::avgpool3d -input $x -kernelSize 2 -stride {1 1 1}]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 3 5 5 5"}
} {1}

test avgpool3d-2.7 {named with list padding} {
    set x [create_input]
    set y [torch::avgpool3d -input $x -kernelSize 2 -padding {1 1 1}]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 3 4 4 4"}
} {1}

# Test 3: camelCase alias

test avgpool3d-3.1 {camelCase alias basic} {
    set x [create_input]
    set y [torch::avgPool3d -input $x -kernelSize 2]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 3 3 3 3"}
} {1}

test avgpool3d-3.2 {camelCase with multiple parameters} {
    set x [create_input]
    set y [torch::avgPool3d -input $x -kernelSize 2 -stride 1 -padding 1]
    set shape [torch::tensor_shape $y]
    expr {$shape eq "1 3 7 7 7"}
} {1}

# Test 4: Mathematical correctness

test avgpool3d-4.1 {mathematical correctness - shape} {
    # Create a 2x2x2 input tensor
    set input [torch::ones {1 1 2 2 2}]
    set output [torch::avgpool3d $input 2]
    set shape [torch::tensor_shape $output]
    expr {$shape eq "1 1 1 1 1"}
} {1}

# Test 5: Consistency between syntaxes

test avgpool3d-5.1 {consistency - positional vs named} {
    set x [create_input]
    set y1 [torch::avgpool3d $x 2 2 1]
    set y2 [torch::avgpool3d -input $x -kernelSize 2 -stride 2 -padding 1]
    set shape1 [torch::tensor_shape $y1]
    set shape2 [torch::tensor_shape $y2]
    expr {$shape1 eq $shape2}
} {1}

test avgpool3d-5.2 {consistency - snake_case vs camelCase} {
    set x [create_input]
    set y1 [torch::avgpool3d -input $x -kernelSize 2]
    set y2 [torch::avgPool3d -input $x -kernelSize 2]
    set shape1 [torch::tensor_shape $y1]
    set shape2 [torch::tensor_shape $y2]
    expr {$shape1 eq $shape2}
} {1}

# Test 6: Error handling

test avgpool3d-6.1 {missing input} -body {
    torch::avgpool3d -kernelSize 2
} -returnCodes error -match glob -result *input*

test avgpool3d-6.2 {missing kernelSize} -body {
    set x [create_input]
    torch::avgpool3d -input $x -stride 2
} -returnCodes error -match glob -result *kernelSize*

test avgpool3d-6.3 {invalid input tensor} -body {
    torch::avgpool3d -input "nonexistent" -kernelSize 2
} -returnCodes error -match glob -result *Invalid*input*

test avgpool3d-6.4 {invalid kernelSize} -body {
    set x [create_input]
    torch::avgpool3d -input $x -kernelSize {1 2}
} -returnCodes error -match glob -result *List*length*

test avgpool3d-6.5 {wrong number of args} -body {
    torch::avgpool3d
} -returnCodes error -match glob -result *Required*parameters*

cleanupTests