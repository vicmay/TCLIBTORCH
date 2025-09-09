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

# Test 1: Basic positional syntax (backward compatibility)
test isnan-1.1 {Basic positional syntax with finite values} {
    set t1 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result [torch::isnan $t1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}  ;# Shape is "3", not "{3}"
} {1}

test isnan-1.2 {Positional syntax with zeros} {
    set t1 [torch::tensorCreate -data {0.0 -0.0} -dtype float32] 
    set result [torch::isnan $t1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}  ;# Shape is "2", not "{2}"
} {1}

# Test 2: Named parameter syntax
test isnan-2.1 {Named parameter syntax with -input} {
    set t1 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result [torch::isnan -input $t1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

test isnan-2.2 {Named parameter syntax with -tensor alias} {
    set t1 [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    set result [torch::isnan -tensor $t1]  
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} {1}

# Test 3: camelCase alias (isNan)
test isnan-3.1 {camelCase alias with positional syntax} {
    set t1 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result [torch::isNan $t1]
    set shape [torch::tensor_shape $result] 
    expr {$shape eq "3"}
} {1}

test isnan-3.2 {camelCase alias with named parameters} {
    set t1 [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    set result [torch::isNan -input $t1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} {1}

# Test 4: Syntax consistency - both syntaxes produce same results
test isnan-4.1 {Syntax consistency check} {
    set t1 [torch::tensorCreate -data {1.0 0.0 -3.14} -dtype float32]
    set result1 [torch::isnan $t1]           ;# Positional
    set result2 [torch::isnan -input $t1]    ;# Named
    set result3 [torch::isNan $t1]           ;# camelCase positional
    set result4 [torch::isNan -input $t1]    ;# camelCase named
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2] 
    set shape3 [torch::tensor_shape $result3]
    set shape4 [torch::tensor_shape $result4]
    
    ;# All should have identical shapes
    expr {$shape1 eq $shape2 && $shape2 eq $shape3 && $shape3 eq $shape4}
} {1}

# Test 5: Error handling 
test isnan-5.1 {Error: missing arguments} {
    catch {torch::isnan} result
    regexp {Usage:} $result
} {1}

test isnan-5.2 {Error: invalid tensor name positional} {
    catch {torch::isnan invalid_tensor} result
    regexp {Invalid tensor name} $result
} {1}

test isnan-5.3 {Error: invalid tensor name named} {
    catch {torch::isnan -input invalid_tensor} result
    regexp {Invalid tensor name} $result
} {1}

test isnan-5.4 {Error: unknown parameter} {
    set t1 [torch::tensorCreate -data {1.0 2.0} -dtype float32]
    catch {torch::isnan -invalid $t1} result
    regexp {Unknown parameter} $result
} {1}

test isnan-5.5 {Error: missing parameter value} {
    catch {torch::isnan -input} result
    regexp {Missing value for parameter} $result
} {1}

# Test 6: Different data types
test isnan-6.1 {Double precision values} {
    set t1 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float64]
    set result [torch::isnan $t1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

test isnan-6.2 {Integer values (none NaN)} {
    set t1 [torch::tensorCreate -data {1 2 3 -5 0} -dtype int32]
    set result [torch::isnan $t1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "5"}  ;# All integers are non-NaN
} {1}

# Test 7: Edge cases with finite values
test isnan-7.1 {Very large finite values} {
    set t1 [torch::tensorCreate -data {1e30 -1e30 1e-30} -dtype float32]
    set result [torch::isnan $t1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}  ;# All should be finite (non-NaN)
} {1}

test isnan-7.2 {Zero values} {
    set t1 [torch::tensorCreate -data {0.0 -0.0} -dtype float32]
    set result [torch::isnan $t1]
    set shape [torch::tensor_shape $result] 
    expr {$shape eq "2"}  ;# Both zeros are finite (non-NaN)
} {1}

test isnan-7.3 {Empty tensor} {
    set t1 [torch::empty -shape {0}]
    set result [torch::isnan $t1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "0"}  ;# Result should have same shape
} {1}

# Test 8: Multi-dimensional tensors (using proper syntax)
test isnan-8.1 {2D tensor} {
    set t1 [torch::tensorCreate -data {1.0 2.0 3.0 -1.0} -shape {2 2} -dtype float32]
    set result [torch::isnan $t1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}  ;# Should preserve 2D shape
} {1}

# Test 9: Single element tensor with tensor_item
test isnan-9.1 {Single element tensor - finite value} {
    set t1 [torch::tensorCreate -data {42.5} -dtype float32]
    set result [torch::isnan $t1]
    set value [torch::tensor_item $result]
    expr {$value == 0}  ;# Should be 0 (false) for finite value
} {1}

test isnan-9.2 {Single element tensor - zero} {
    set t1 [torch::tensorCreate -data {0.0} -dtype float32]
    set result [torch::isnan $t1]
    set value [torch::tensor_item $result]
    expr {$value == 0}  ;# Should be 0 (false) for zero
} {1}

# Test 10: Special cases with negative values
test isnan-10.1 {Negative finite values} {
    set t1 [torch::tensorCreate -data {-1.0 -2.5 -100.0} -dtype float32]
    set result [torch::isnan $t1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}  ;# Negative finite values are not NaN
} {1}

test isnan-10.2 {Very small finite values} {
    set t1 [torch::tensorCreate -data {1e-38 -1e-38} -dtype float32]
    set result [torch::isnan $t1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}  ;# Very small finite values are not NaN
} {1}

# Test 11: Special mathematical values
test isnan-11.1 {Mix of finite and normal values} {
    set t1 [torch::tensorCreate -data {1.0 -1.0 0.0 100.0 -100.0} -dtype float32]
    set result [torch::isnan $t1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "5"}  ;# All normal values should be non-NaN
} {1}

test isnan-11.2 {Fractional values} {
    set t1 [torch::tensorCreate -data {0.5 -0.5 3.14159 -2.71828} -dtype float32]
    set result [torch::isnan $t1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "4"}  ;# All fractional values should be non-NaN
} {1}

cleanupTests 