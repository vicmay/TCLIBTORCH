#!/usr/bin/env tclsh

package require tcltest
namespace import tcltest::*

# Load the library
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test cases for positional syntax
test poisson-1.1 {Basic poisson sampling with positional syntax} {
    set handle [torch::poisson {2 3} 2.0]
    set shape [torch::tensor_shape $handle]
    expr {$shape eq {2 3}}
} {1}

test poisson-1.2 {Poisson sampling with custom dtype} {
    set handle [torch::poisson {3 2} 5.0 float32]
    expr {[torch::tensor_dtype $handle] eq "Float32" && [torch::tensor_shape $handle] eq {3 2}}
} {1}

test poisson-1.3 {Poisson sampling with custom device} {
    set handle [torch::poisson {2 2} 3.0 float32 cpu]
    expr {[torch::tensor_shape $handle] eq {2 2}}
} {1}

# Test cases for named parameter syntax
test poisson-2.1 {Basic poisson sampling with named parameters} {
    set handle [torch::poisson -size {2 3} -lambda 2.0]
    set shape [torch::tensor_shape $handle]
    expr {$shape eq {2 3}}
} {1}

test poisson-2.2 {Poisson sampling with custom dtype using named parameters} {
    set handle [torch::poisson -size {3 2} -lambda 5.0 -dtype float32]
    expr {[torch::tensor_dtype $handle] eq "Float32" && [torch::tensor_shape $handle] eq {3 2}}
} {1}

test poisson-2.3 {Poisson sampling with custom device using named parameters} {
    set handle [torch::poisson -size {2 2} -lambda 3.0 -dtype float32 -device cpu]
    expr {[torch::tensor_shape $handle] eq {2 2}}
} {1}

# Test cases for error handling
test poisson-3.1 {Error on invalid size} {
    catch {torch::poisson -size {-1 2} -lambda 2.0} err
    set err
} {Invalid size: dimensions must be positive}

test poisson-3.2 {Error on invalid lambda} {
    catch {torch::poisson {2 2} -1.0} err
    set err
} {Invalid lambda: must be non-negative}

test poisson-3.3 {Error on invalid dtype} {
    catch {torch::poisson {2 2} 2.0 int32} err
    set err
} {Invalid dtype: must be float32 or float64}

# Test cases for camelCase alias
test poisson-4.1 {Basic poisson sampling with camelCase alias} {
    set handle [torch::Poisson {2 3} 2.0]
    set shape [torch::tensor_shape $handle]
    expr {$shape eq {2 3}}
} {1}

test poisson-4.2 {Poisson sampling with named parameters using camelCase alias} {
    set handle [torch::Poisson -size {3 2} -lambda 5.0 -dtype float32]
    expr {[torch::tensor_dtype $handle] eq "Float32" && [torch::tensor_shape $handle] eq {3 2}}
} {1}

cleanupTests 