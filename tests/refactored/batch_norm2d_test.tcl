#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

if {[catch {load ../../build/libtorchtcl.so} err]} {
    puts "Failed to load libtorchtcl.so: $err"
    exit 1
}

configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Positional syntax
test batchnorm2d-1.1 {Positional syntax layer creation} {
    set bn [torch::batchnorm2d 32]
    expr {[string length $bn] > 0}
} 1

# Named parameter syntax
test batchnorm2d-2.1 {Named parameter syntax} {
    set bn [torch::batchNorm2d -numFeatures 32 -eps 1e-4 -momentum 0.05 -affine false -trackRunningStats false]
    expr {[string length $bn] > 0}
} 1

# camelCase alias (already tested above) but include simple creation
test batchnorm2d-3.1 {CamelCase alias minimal} {
    set bn [torch::batchNorm2d -numFeatures 16]
    expr {[string length $bn] > 0}
} 1

# Error handling missing numFeatures
test batchnorm2d-4.1 {Error missing numFeatures} -body {
    torch::batchNorm2d -eps 1e-3
} -returnCodes error -match glob -result *

cleanupTests 