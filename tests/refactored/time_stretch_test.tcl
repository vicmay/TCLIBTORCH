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

# Helper to create tensors
proc create_tensor {data {dtype float32}} {
    return [torch::tensor_create -data $data -dtype $dtype]
}

# Helper to create STFT matrix (simplified)
proc create_stft_matrix {rows cols} {
    set data {}
    for {set i 0} {$i < $rows} {incr i} {
        set row {}
        for {set j 0} {$j < $cols} {incr j} {
            lappend row [expr {sin($i * 0.1) * cos($j * 0.1)}]
        }
        lappend data $row
    }
    return [create_tensor $data]
}

# Test cases for positional syntax
test time_stretch-1.1 {Basic positional syntax} {
    set stft [create_stft_matrix 4 8]
    set out [torch::time_stretch $stft 2.0]
    set result [torch::tensor_shape $out]
    set result
} {4 4}

test time_stretch-1.2 {Positional syntax with rate 0.5} {
    set stft [create_stft_matrix 4 8]
    set out [torch::time_stretch $stft 0.5]
    set result [torch::tensor_shape $out]
    set result
} {4 16}

test time_stretch-1.3 {Positional syntax with rate 1.5} {
    set stft [create_stft_matrix 4 8]
    set out [torch::time_stretch $stft 1.5]
    set result [torch::tensor_shape $out]
    set result
} {4 5}

# Test cases for named syntax
test time_stretch-2.1 {Named parameter syntax} {
    set stft [create_stft_matrix 4 8]
    set out [torch::time_stretch -input $stft -rate 2.0]
    set result [torch::tensor_shape $out]
    set result
} {4 4}

test time_stretch-2.2 {Named syntax with rate 0.5} {
    set stft [create_stft_matrix 4 8]
    set out [torch::time_stretch -input $stft -rate 0.5]
    set result [torch::tensor_shape $out]
    set result
} {4 16}

test time_stretch-2.3 {Named syntax with rate 1.5} {
    set stft [create_stft_matrix 4 8]
    set out [torch::time_stretch -input $stft -rate 1.5]
    set result [torch::tensor_shape $out]
    set result
} {4 5}

# Test cases for camelCase alias
test time_stretch-3.1 {CamelCase alias} {
    set stft [create_stft_matrix 4 8]
    set out [torch::timeStretch -input $stft -rate 2.0]
    set result [torch::tensor_shape $out]
    set result
} {4 4}

test time_stretch-3.2 {CamelCase with rate 0.5} {
    set stft [create_stft_matrix 4 8]
    set out [torch::timeStretch -input $stft -rate 0.5]
    set result [torch::tensor_shape $out]
    set result
} {4 16}

# Error handling tests
test time_stretch-4.1 {Missing argument error} -returnCodes error -body {
    torch::time_stretch
} -match glob -result {*Required parameters missing: input tensor required*}

test time_stretch-4.2 {Invalid tensor} -returnCodes error -body {
    torch::time_stretch invalid_tensor 2.0
} -match glob -result {*Invalid tensor*}

test time_stretch-4.3 {Invalid rate value} -returnCodes error -body {
    set stft [create_stft_matrix 4 8]
    torch::time_stretch $stft invalid
} -match glob -result {*Invalid rate value*}

test time_stretch-4.4 {Zero rate} -returnCodes error -body {
    set stft [create_stft_matrix 4 8]
    torch::time_stretch $stft 0.0
} -match glob -result {*rate must be positive*}

test time_stretch-4.5 {Negative rate} -returnCodes error -body {
    set stft [create_stft_matrix 4 8]
    torch::time_stretch $stft -1.0
} -match glob -result {*rate must be positive*}

test time_stretch-4.6 {Unknown parameter} -returnCodes error -body {
    set stft [create_stft_matrix 4 8]
    torch::time_stretch -input $stft -invalid 2.0
} -match glob -result {*Unknown parameter: -invalid*}

test time_stretch-4.7 {Missing parameter value} -returnCodes error -body {
    set stft [create_stft_matrix 4 8]
    torch::time_stretch -input $stft -rate
} -match glob -result {*Missing value for parameter*}

# Edge cases and different rates
test time_stretch-5.1 {Very small rate} {
    set stft [create_stft_matrix 4 8]
    set out [torch::time_stretch $stft 0.1]
    set result [torch::tensor_shape $out]
    set result
} {4 80}

test time_stretch-5.2 {Very large rate} -returnCodes error -body {
    set stft [create_stft_matrix 4 8]
    torch::time_stretch $stft 10.0
} -match glob -result {*Input and output sizes should be greater than 0*}

test time_stretch-5.3 {Rate of 1.0 (no change)} {
    set stft [create_stft_matrix 4 8]
    set out [torch::time_stretch $stft 1.0]
    set result [torch::tensor_shape $out]
    set result
} {4 8}

test time_stretch-5.4 {Large input matrix} {
    set stft [create_stft_matrix 8 16]
    set out [torch::time_stretch $stft 2.0]
    set result [torch::tensor_shape $out]
    set result
} {8 8}

test time_stretch-5.5 {Small input matrix} {
    set stft [create_stft_matrix 2 4]
    set out [torch::time_stretch $stft 0.5]
    set result [torch::tensor_shape $out]
    set result
} {2 8}

# Test with different data types
test time_stretch-6.1 {Float64 input} {
    set stft [create_stft_matrix 4 8]
    set stft64 [torch::tensor_to -input $stft -device cpu -dtype float64]
    set out [torch::time_stretch $stft64 2.0]
    set result [torch::tensor_shape $out]
    set result
} {4 4}

# Test magnitude preservation (basic check)
test time_stretch-7.1 {Magnitude values are preserved} {
    set stft [create_stft_matrix 4 8]
    set out [torch::time_stretch $stft 1.0]
    set input_mag [torch::tensor_to_list [torch::tensor_abs $stft]]
    set output_mag [torch::tensor_to_list [torch::tensor_abs $out]]
    set input_sum [expr [join $input_mag +]]
    set output_sum [expr [join $output_mag +]]
    expr {abs($input_sum - $output_sum) < 1e-6}
} {1}

cleanupTests 