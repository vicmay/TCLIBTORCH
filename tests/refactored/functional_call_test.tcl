#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Create helper tensors
set param [torch::randn [list 2 2]]
set input [torch::zeros [list 2 2]]
set paramShape [torch::tensor_shape $param]

# Positional
test functional_call-1.1 {positional syntax} -body {
    set res [torch::functional_call myFunc $param $input]
    set resShape [torch::tensor_shape $res]
    expr {$resShape eq $paramShape}
} -result {1}

test functional_call-1.2 {wrong arg count} -body {
    catch {torch::functional_call} r
    set r
} -match glob -result {*Required parameters*}

# Named
test functional_call-2.1 {named syntax} -body {
    set res [torch::functional_call -func myFunc -parameters $param -args $input]
    set resShape [torch::tensor_shape $res]
    expr {$resShape eq $paramShape}
} -result {1}

test functional_call-2.2 {missing value} -body {
    catch {torch::functional_call -func myFunc -parameters} r
    set r
} -match glob -result {*Missing value*}

# camelCase
test functional_call-3.1 {camelCase positional} -body {
    set res [torch::functionalCall myFunc $param]
    set resShape [torch::tensor_shape $res]
    expr {$resShape eq $paramShape}
} -result {1}

test functional_call-3.2 {camelCase named} -body {
    set res [torch::functionalCall -func myFunc -parameters $param]
    set resShape [torch::tensor_shape $res]
    expr {$resShape eq $paramShape}
} -result {1}

cleanupTests 