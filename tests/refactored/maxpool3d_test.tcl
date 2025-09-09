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

# Test cases for positional syntax
test maxpool3d-1.1 {Basic positional syntax} {
    set layer [torch::maxpool3d 3]
    expr {[string match "maxpool3d*" $layer]}
} {1}

test maxpool3d-1.2 {Positional syntax with stride} {
    set layer [torch::maxpool3d 3 2]
    expr {[string match "maxpool3d*" $layer]}
} {1}

test maxpool3d-1.3 {Positional syntax with stride and padding} {
    set layer [torch::maxpool3d 3 2 1]
    expr {[string match "maxpool3d*" $layer]}
} {1}

test maxpool3d-1.4 {Full positional syntax} {
    set layer [torch::maxpool3d 3 2 1 1]
    expr {[string match "maxpool3d*" $layer]}
} {1}

# Test cases for named parameter syntax
test maxpool3d-2.1 {Basic named parameter syntax} {
    set layer [torch::maxpool3d -kernelSize 3]
    expr {[string match "maxpool3d*" $layer]}
} {1}

test maxpool3d-2.2 {Named parameter syntax with stride} {
    set layer [torch::maxpool3d -kernelSize 3 -stride 2]
    expr {[string match "maxpool3d*" $layer]}
} {1}

test maxpool3d-2.3 {Named parameter syntax with padding} {
    set layer [torch::maxpool3d -kernelSize 3 -padding 1]
    expr {[string match "maxpool3d*" $layer]}
} {1}

test maxpool3d-2.4 {Full named parameter syntax} {
    set layer [torch::maxpool3d -kernelSize 3 -stride 2 -padding 1 -ceilMode 1]
    expr {[string match "maxpool3d*" $layer]}
} {1}

# Test cases for camelCase alias
test maxpool3d-3.1 {CamelCase alias basic} {
    set layer [torch::maxPool3d -kernelSize 3]
    expr {[string match "maxpool3d*" $layer]}
} {1}

test maxpool3d-3.2 {CamelCase alias full parameters} {
    set layer [torch::maxPool3d -kernelSize 3 -stride 2 -padding 1 -ceilMode 1]
    expr {[string match "maxpool3d*" $layer]}
} {1}

# Error handling tests
test maxpool3d-4.1 {Error on missing kernel size} -body {
    torch::maxpool3d
} -returnCodes error -result {Usage: torch::maxpool3d kernel_size ?stride? ?padding? ?ceil_mode? | torch::maxpool3d -kernelSize value ?-stride value? ?-padding value? ?-ceilMode value?}

test maxpool3d-4.2 {Error on invalid kernel size} -body {
    torch::maxpool3d 0
} -returnCodes error -result {kernelSize must be > 0}

test maxpool3d-4.3 {Error on invalid parameter name} -body {
    torch::maxpool3d -invalid 3
} -returnCodes error -result {Unknown parameter: -invalid}

test maxpool3d-4.4 {Error on missing parameter value} -body {
    torch::maxpool3d -kernelSize
} -returnCodes error -result {Missing value for parameter}

# Functional tests
test maxpool3d-5.1 {Forward pass with 3D input} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32]
    set input [torch::tensor_reshape $input {1 1 2 2 2}]
    set layer [torch::maxpool3d 2]
    set output [torch::layer_forward $layer $input]
    set shape [torch::tensor_shape $output]
    set result [torch::tensor_print $output]
    # Extract just the numeric value from the tensor string
    set value [regsub -all {[^0-9.]} $result ""]
    list $shape $value
} {{1 1 1 1 1} 8.}

test maxpool3d-5.2 {Forward pass with stride and padding} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32]
    set input [torch::tensor_reshape $input {1 1 2 2 2}]
    set layer [torch::maxpool3d -kernelSize 2 -stride 1 -padding 1]
    set output [torch::layer_forward $layer $input]
    set shape [torch::tensor_shape $output]
    set result [torch::tensor_print $output]
    # Extract just the numeric values from the tensor string
    set values [regsub -all {[^0-9., ]} $result ""]
    set values [regsub -all {[, ]+} $values " "]
    set values [string trim $values]
    list $shape $values
} {{1 1 2 2 2} {1. 2. 3. 4. 5. 6. 7. 8.}}

cleanupTests