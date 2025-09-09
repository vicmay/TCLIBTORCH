#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# Test cases for positional syntax
test zero-pad1d-1.1 {Basic positional syntax} {
    set t [torch::arange -start 1 -end 5 -dtype float32]
    set padded [torch::zero_pad1d $t {2 3}]
    set shape [torch::tensor_shape $padded]
    return $shape
} {9}

test zero-pad1d-1.2 {Positional syntax, no padding} {
    set t [torch::arange -start 1 -end 5 -dtype float32]
    set padded [torch::zero_pad1d $t {0 0}]
    set shape [torch::tensor_shape $padded]
    return $shape
} {4}

;# Test cases for named parameter syntax
test zero-pad1d-2.1 {Named parameter syntax -input/-padding} {
    set t [torch::arange -start 1 -end 5 -dtype float32]
    set padded [torch::zero_pad1d -input $t -padding {1 2}]
    set shape [torch::tensor_shape $padded]
    return $shape
} {7}

test zero-pad1d-2.2 {Named parameter syntax -tensor/-pad} {
    set t [torch::arange -start 1 -end 5 -dtype float32]
    set padded [torch::zero_pad1d -tensor $t -pad {3 1}]
    set shape [torch::tensor_shape $padded]
    return $shape
} {8}

;# Test cases for camelCase alias
test zero-pad1d-3.1 {CamelCase alias basic} {
    set t [torch::arange -start 1 -end 5 -dtype float32]
    set padded [torch::zeroPad1d $t {1 1}]
    set shape [torch::tensor_shape $padded]
    return $shape
} {6}

test zero-pad1d-3.2 {CamelCase alias named parameters} {
    set t [torch::arange -start 1 -end 5 -dtype float32]
    set padded [torch::zeroPad1d -input $t -padding {2 0}]
    set shape [torch::tensor_shape $padded]
    return $shape
} {6}

;# Mathematical correctness
test zero-pad1d-4.1 {Zero padding values are zeros} {
    set t [torch::arange -start 1 -end 4 -dtype float32]
    set padded [torch::zero_pad1d $t {2 1}]
    set data [torch::tensorToList $padded]
    return [lrange $data 0 1]
} {0.0 0.0}

test zero-pad1d-4.2 {Original values preserved} {
    set t [torch::arange -start 1 -end 4 -dtype float32]
    set padded [torch::zero_pad1d $t {2 1}]
    set data [torch::tensorToList $padded]
    return [lrange $data 2 4]
} {1.0 2.0 3.0}

test zero-pad1d-4.3 {Right padding values are zeros} {
    set t [torch::arange -start 1 -end 4 -dtype float32]
    set padded [torch::zero_pad1d $t {2 2}]
    set data [torch::tensorToList $padded]
    return [lrange $data end-1 end]
} {0.0 0.0}

;# Error handling
test zero-pad1d-5.1 {Missing arguments} {
    catch {torch::zero_pad1d} result
    return [string match "*Usage*" $result]
} {1}

test zero-pad1d-5.2 {Invalid padding length} {
    set t [torch::arange -start 1 -end 5 -dtype float32]
    catch {torch::zero_pad1d $t {1 2 3}} result
    return [string match "*Padding must be a list of 2 values*" $result]
} {1}

test zero-pad1d-5.3 {Unknown named parameter} {
    set t [torch::arange -start 1 -end 5 -dtype float32]
    catch {torch::zero_pad1d -foo $t -padding {1 1}} result
    return [string match "*Unknown parameter*" $result]
} {1}

test zero-pad1d-5.4 {Missing value for parameter} {
    set t [torch::arange -start 1 -end 5 -dtype float32]
    catch {torch::zero_pad1d -input $t -padding} result
    return [string match "*Missing value for parameter*" $result]
} {1}

;# Edge cases
test zero-pad1d-6.1 {Zero-length tensor} {
    set t [torch::empty -shape {0} -dtype float32]
    catch {torch::zero_pad1d $t {1 1}} result
    return [string match "*Input tensor is empty*" $result]
} {1}

test zero-pad1d-6.2 {Negative padding} {
    set t [torch::arange -start 1 -end 5 -dtype float32]
    catch {torch::zero_pad1d $t {-1 2}} result
    return [string match "*must be non-negative*" $result]
} {0}

cleanupTests 