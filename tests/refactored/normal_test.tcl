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
test normal-1.1 {Basic positional syntax} {
    set result [torch::normal 0.0 1.0]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "1"}
} {1}

test normal-1.2 {Positional syntax with size} {
    set result [torch::normal 0.0 1.0 {2 3}]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 3"}
} {1}

test normal-1.3 {Positional syntax with size and dtype} {
    set result [torch::normal 0.0 1.0 {2 3} float64]
    set dtype [torch::tensor_dtype $result]
    expr {[string match "*Float64*" $dtype]}
} {1}

test normal-1.4 {Positional syntax with all parameters} {
    set result [torch::normal 0.0 1.0 {2 3} float32 cpu]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    expr {$shape eq "2 3" && [string match "*Float32*" $dtype]}
} {1}

# Test cases for named parameter syntax
test normal-2.1 {Named parameter syntax} {
    set result [torch::normal -mean 0.0 -std 1.0]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "1"}
} {1}

test normal-2.2 {Named parameter syntax with size} {
    set result [torch::normal -mean 0.0 -std 1.0 -size {2 3}]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 3"}
} {1}

test normal-2.3 {Named parameter syntax with size and dtype} {
    set result [torch::normal -mean 0.0 -std 1.0 -size {2 3} -dtype float64]
    set dtype [torch::tensor_dtype $result]
    expr {[string match "*Float64*" $dtype]}
} {1}

test normal-2.4 {Named parameter syntax with all parameters} {
    set result [torch::normal -mean 0.0 -std 1.0 -size {2 3} -dtype float32 -device cpu]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    expr {$shape eq "2 3" && [string match "*Float32*" $dtype]}
} {1}

# Test cases for camelCase alias
test normal-3.1 {CamelCase alias with positional syntax} {
    set result [torch::Normal 0.0 1.0 {2 3}]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 3"}
} {1}

test normal-3.2 {CamelCase alias with named parameters} {
    set result [torch::Normal -mean 0.0 -std 1.0 -size {2 3}]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 3"}
} {1}

# Statistical tests
test normal-4.1 {Mean close to specified value} {
    set samples [torch::normal 5.0 1.0 {10000}]
    set mean [torch::tensor_mean $samples]
    set mean_val [torch::tensor_item $mean]
    expr {abs($mean_val - 5.0) < 0.1}
} {1}

test normal-4.2 {Standard deviation close to specified value} {
    set samples [torch::normal 0.0 2.0 {10000}]
    set std [torch::tensor_std $samples]
    set std_val [torch::tensor_item $std]
    expr {abs($std_val - 2.0) < 0.1}
} {1}

# Error cases
test normal-5.1 {Error on missing arguments} {
    catch {torch::normal} err
    set err
} {wrong # args: should be "torch::normal mean std ?size? ?dtype? ?device?"}

test normal-5.2 {Error on invalid size} {
    catch {torch::normal 0.0 1.0 invalid_size} err
    expr {[string match "*Error*" $err]}
} {1}

test normal-5.3 {Error on invalid dtype} {
    catch {torch::normal 0.0 1.0 {2 3} invalid_dtype} err
    expr {[string match "*Error*" $err]}
} {1}

test normal-5.4 {Error on invalid device} {
    catch {torch::normal 0.0 1.0 {2 3} float32 invalid_device} err
    expr {[string match "*Error*" $err]}
} {1}

cleanupTests 