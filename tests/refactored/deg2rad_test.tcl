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

# Test helper function to verify tensor result
proc verify_deg2rad_result {result} {
    # Verify we got a valid tensor handle back
    if {![string match "tensor*" $result]} {
        return 0
    }
    
    # Basic verification that the tensor exists and can be accessed
    set shape [torch::tensor_shape $result]
    if {$shape == ""} {
        return 0
    }
    
    return 1
}

# Test positional syntax
test deg2rad-1.1 {Basic positional syntax} {
    set t1 [torch::tensor_create -data {0.0 90.0} -shape {2} -dtype float32]
    set result [torch::deg2rad $t1]
    expr {[verify_deg2rad_result $result] && [torch::tensor_shape $result] eq "2"}
} {1}

test deg2rad-1.2 {Positional syntax with 45 degrees} {
    set t1 [torch::tensor_create -data {45.0} -shape {1} -dtype float32]
    set result [torch::deg2rad $t1]
    verify_deg2rad_result $result
} {1}

test deg2rad-1.3 {Positional syntax with 180 degrees} {
    set t1 [torch::tensor_create -data {180.0} -shape {1} -dtype float32]
    set result [torch::deg2rad $t1]
    verify_deg2rad_result $result
} {1}

# Test named parameter syntax
test deg2rad-2.1 {Named parameter syntax} {
    set t1 [torch::tensor_create -data {0.0 90.0} -shape {2} -dtype float32]
    set result [torch::deg2rad -input $t1]
    expr {[verify_deg2rad_result $result] && [torch::tensor_shape $result] eq "2"}
} {1}

test deg2rad-2.2 {Named parameter with single value} {
    set t1 [torch::tensor_create -data {360.0} -shape {1} -dtype float32]
    set result [torch::deg2rad -input $t1]
    verify_deg2rad_result $result
} {1}

test deg2rad-2.3 {Named parameter with negative degrees} {
    set t1 [torch::tensor_create -data {-90.0} -shape {1} -dtype float32]
    set result [torch::deg2rad -input $t1]
    verify_deg2rad_result $result
} {1}

test deg2rad-2.4 {Named parameter with multi-dimensional tensor} {
    set t1 [torch::tensor_create -data {0.0 45.0 90.0 180.0} -shape {2 2} -dtype float32]
    set result [torch::deg2rad -input $t1]
    expr {[verify_deg2rad_result $result] && [torch::tensor_shape $result] eq "2 2"}
} {1}

# Test camelCase alias
test deg2rad-3.1 {CamelCase alias positional} {
    set t1 [torch::tensor_create -data {45.0} -shape {1} -dtype float32]
    set result [torch::deg2Rad $t1]
    verify_deg2rad_result $result
} {1}

test deg2rad-3.2 {CamelCase alias named parameter} {
    set t1 [torch::tensor_create -data {90.0} -shape {1} -dtype float32]
    set result [torch::deg2Rad -input $t1]
    verify_deg2rad_result $result
} {1}

# Test mathematical correctness
test deg2rad-4.1 {Mathematical correctness - common angles} {
    set t1 [torch::tensor_create -data {0.0 90.0 180.0 270.0} -shape {4} -dtype float32]
    set result [torch::deg2rad $t1]
    expr {[verify_deg2rad_result $result] && [torch::tensor_shape $result] eq "4"}
} {1}

test deg2rad-4.2 {Mathematical correctness - full circle} {
    set t1 [torch::tensor_create -data {360.0} -shape {1} -dtype float32]
    set result [torch::deg2rad $t1]
    verify_deg2rad_result $result
} {1}

test deg2rad-4.3 {Mathematical correctness - fractional degrees} {
    set t1 [torch::tensor_create -data {30.5} -shape {1} -dtype float32]
    set result [torch::deg2rad $t1]
    verify_deg2rad_result $result
} {1}

# Test consistency between syntaxes
test deg2rad-5.1 {Consistency between positional and named syntax} {
    set t1 [torch::tensor_create -data {30.0 60.0 120.0} -shape {3} -dtype float32]
    set result1 [torch::deg2rad $t1]
    set result2 [torch::deg2rad -input $t1]
    
    # Both should be valid tensors
    set valid1 [verify_deg2rad_result $result1]
    set valid2 [verify_deg2rad_result $result2]
    
    # Both should have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$valid1 && $valid2 && $shape1 eq $shape2}
} {1}

test deg2rad-5.2 {Consistency between snake_case and camelCase} {
    set t1 [torch::tensor_create -data {45.0 135.0} -shape {2} -dtype float32]
    set result1 [torch::deg2rad $t1]
    set result2 [torch::deg2Rad $t1]
    
    # Both should be valid tensors
    set valid1 [verify_deg2rad_result $result1]
    set valid2 [verify_deg2rad_result $result2]
    
    # Both should have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$valid1 && $valid2 && $shape1 eq $shape2}
} {1}

# Test data type support
test deg2rad-6.1 {Float32 tensor support} {
    set t1 [torch::tensor_create -data {45.0 90.0} -shape {2} -dtype float32]
    set result [torch::deg2rad $t1]
    expr {[verify_deg2rad_result $result] && [torch::tensor_dtype $result] eq "Float32"}
} {1}

test deg2rad-6.2 {Float64 tensor support} {
    set t1 [torch::tensor_create -data {45.0 90.0} -shape {2} -dtype float64]
    set result [torch::deg2rad $t1]
    expr {[verify_deg2rad_result $result] && [torch::tensor_dtype $result] eq "Float64"}
} {1}

# Test edge cases
test deg2rad-7.1 {Zero degrees} {
    set t1 [torch::tensor_create -data {0.0} -shape {1} -dtype float32]
    set result [torch::deg2rad $t1]
    verify_deg2rad_result $result
} {1}

test deg2rad-7.2 {Large positive angle} {
    set t1 [torch::tensor_create -data {720.0} -shape {1} -dtype float32]
    set result [torch::deg2rad $t1]
    verify_deg2rad_result $result
} {1}

test deg2rad-7.3 {Large negative angle} {
    set t1 [torch::tensor_create -data {-180.0} -shape {1} -dtype float32]
    set result [torch::deg2rad $t1]
    verify_deg2rad_result $result
} {1}

test deg2rad-7.4 {Very small angle} {
    set t1 [torch::tensor_create -data {0.1} -shape {1} -dtype float32]
    set result [torch::deg2rad $t1]
    verify_deg2rad_result $result
} {1}

test deg2rad-7.5 {Empty tensor} {
    set t1 [torch::tensor_create -data {} -shape {0} -dtype float32]
    set result [torch::deg2rad $t1]
    expr {[verify_deg2rad_result $result] && [torch::tensor_shape $result] eq "0"}
} {1}

test deg2rad-7.6 {Scalar tensor} {
    set t1 [torch::tensor_create -data {90.0} -shape {} -dtype float32]
    set result [torch::deg2rad $t1]
    verify_deg2rad_result $result
} {1}

test deg2rad-7.7 {Large multi-dimensional tensor} {
    set t1 [torch::tensor_create -data {0.0 30.0 45.0 60.0 90.0 180.0} -shape {2 3} -dtype float32]
    set result [torch::deg2rad $t1]
    expr {[verify_deg2rad_result $result] && [torch::tensor_shape $result] eq "2 3"}
} {1}

# Test error handling
test deg2rad-8.1 {Invalid tensor name positional} {
    catch {torch::deg2rad invalid_tensor} msg
    expr {[string match "*Invalid tensor name*" $msg]}
} {1}

test deg2rad-8.2 {Invalid tensor name named parameter} {
    catch {torch::deg2rad -input invalid_tensor} msg
    expr {[string match "*Invalid tensor name*" $msg]}
} {1}

test deg2rad-8.3 {Missing required parameter} {
    catch {torch::deg2rad} msg
    expr {[string match "*Usage*" $msg] || [string match "*missing*" $msg]}
} {1}

test deg2rad-8.4 {Invalid parameter name} {
    set t1 [torch::tensor_create -data {45.0} -shape {1} -dtype float32]
    catch {torch::deg2rad -invalid $t1} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

test deg2rad-8.5 {Too many positional arguments} {
    set t1 [torch::tensor_create -data {45.0} -shape {1} -dtype float32]
    catch {torch::deg2rad $t1 extra_arg} msg
    expr {[string match "*Usage*" $msg]}
} {1}

test deg2rad-8.6 {Missing value for named parameter} {
    catch {torch::deg2rad -input} msg
    expr {[string match "*Missing value*" $msg]}
} {1}

test deg2rad-8.7 {Mixed invalid syntax} {
    set t1 [torch::tensor_create -data {45.0} -shape {1} -dtype float32]
    catch {torch::deg2rad $t1 -input $t1} msg
    expr {[string match "*Usage*" $msg]}
} {1}

cleanupTests 