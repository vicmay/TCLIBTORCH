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

# Helper function to check if two tensors are approximately equal
proc tensorsEqual {t1 t2} {
    set diff [torch::tensor_sub $t1 $t2]
    set max_diff [torch::tensor_max [torch::tensor_abs $diff]]
    set max_val [expr {[torch::tensor_item $max_diff] < 1e-5}]
    return $max_val
}

# Test cases for positional syntax
test rad2deg-1.1 {Basic positional syntax - convert radians to degrees} {
    set input [torch::tensor_create -data {0.0 1.57079633 3.14159265} -shape {3} -dtype float32]
    set result [torch::rad2deg $input]
    set expected [torch::tensor_create -data {0.0 90.0 180.0} -shape {3} -dtype float32]
    tensorsEqual $result $expected
} {1}

test rad2deg-1.2 {Positional syntax - error on wrong number of arguments} {
    catch {torch::rad2deg} msg
    set msg
} {Usage: torch::rad2deg tensor | torch::rad2deg -input tensor}

test rad2deg-1.3 {Positional syntax - error on too many arguments} {
    catch {torch::rad2deg tensor1 tensor2} msg
    set msg
} {Usage: torch::rad2deg tensor}

# Test cases for named parameter syntax
test rad2deg-2.1 {Named parameter syntax - convert radians to degrees} {
    set input [torch::tensor_create -data {0.0 1.57079633 3.14159265} -shape {3} -dtype float32]
    set result [torch::rad2deg -input $input]
    set expected [torch::tensor_create -data {0.0 90.0 180.0} -shape {3} -dtype float32]
    tensorsEqual $result $expected
} {1}

test rad2deg-2.2 {Named parameter syntax - error on missing value} {
    catch {torch::rad2deg -input} msg
    set msg
} {Missing value for parameter}

test rad2deg-2.3 {Named parameter syntax - error on unknown parameter} {
    catch {torch::rad2deg -invalid tensor1} msg
    set msg
} {Unknown parameter: -invalid}

test rad2deg-2.4 {Named parameter syntax - error on missing input} {
    catch {torch::rad2deg} msg
    set msg
} {Usage: torch::rad2deg tensor | torch::rad2deg -input tensor}

# Test cases for camelCase alias
test rad2deg-3.1 {CamelCase alias - convert radians to degrees} {
    set input [torch::tensor_create -data {0.0 1.57079633 3.14159265} -shape {3} -dtype float32]
    set result [torch::radToDeg -input $input]
    set expected [torch::tensor_create -data {0.0 90.0 180.0} -shape {3} -dtype float32]
    tensorsEqual $result $expected
} {1}

# Test edge cases
test rad2deg-4.1 {Edge case - zero tensor} {
    set input [torch::zeros {3} float32]
    set result [torch::rad2deg $input]
    set expected [torch::zeros {3} float32]
    tensorsEqual $result $expected
} {1}

test rad2deg-4.2 {Edge case - negative values} {
    set input [torch::tensor_create -data {-3.14159265 -1.57079633} -shape {2} -dtype float32]
    set result [torch::rad2deg $input]
    set expected [torch::tensor_create -data {-180.0 -90.0} -shape {2} -dtype float32]
    tensorsEqual $result $expected
} {1}

cleanupTests 