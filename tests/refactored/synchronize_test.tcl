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
test synchronize-1.1 {Basic positional syntax} {
    set result [torch::synchronize]
    expr {$result eq "synchronized" || $result eq "cuda_not_available"}
} {1}

test synchronize-1.2 {Positional syntax with device argument} {
    set result [torch::synchronize cpu]
    expr {$result eq "synchronized" || $result eq "cuda_not_available"}
} {1}

;# Test cases for named parameter syntax
test synchronize-2.1 {Named parameter syntax with -device} {
    set result [torch::synchronize -device cpu]
    expr {$result eq "synchronized" || $result eq "cuda_not_available"}
} {1}

;# Test cases for camelCase alias
test synchronize-3.1 {CamelCase alias torch::synchronize} {
    set result [torch::synchronize]
    expr {$result eq "synchronized" || $result eq "cuda_not_available"}
} {1}

test synchronize-3.2 {CamelCase alias with named parameter} {
    set result [torch::synchronize -device cpu]
    expr {$result eq "synchronized" || $result eq "cuda_not_available"}
} {1}

;# Error handling
test synchronize-4.1 {Error: Too many positional arguments} {
    catch {torch::synchronize cpu extra} result
    expr {[string match *Invalid* $result] || [string match *wrong* $result]}
} {1}

test synchronize-4.2 {Error: Named parameter without value} {
    catch {torch::synchronize -device} result
    expr {[string first "Missing value" $result] >= 0}
} {1}

test synchronize-4.3 {Error: Unknown named parameter} {
    catch {torch::synchronize -unknown foo} result
    expr {[string first "Unknown parameter" $result] >= 0}
} {1}

cleanupTests 