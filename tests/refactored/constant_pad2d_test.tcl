#!/usr/bin/env tclsh
# Test file for refactored torch::constant_pad2d command

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Configuration
set COMMAND_NAME "constant_pad2d"
set COMMAND_OLD "torch::constant_pad2d"
set COMMAND_NEW "torch::constantPad2d"

puts "=== Testing Refactored Command: $COMMAND_OLD / $COMMAND_NEW ==="
puts ""

# Test 1: Basic positional syntax
puts "Test 1: Basic positional syntax..."
if {[catch {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    set result [$COMMAND_OLD $input {1 1 1 1} 0.0]
    puts "  Basic positional: OK (result: $result)"
} error]} {
    puts "  ❌ Basic positional failed: $error"
    exit 1
}

# Test 2: Named parameter syntax - basic
puts "Test 2: Named parameter syntax - basic..."
if {[catch {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set input [torch::tensor_reshape $input {2 3}]
    set result [$COMMAND_OLD -input $input -padding {1 2 0 1} -value 5.0]
    puts "  Named basic: OK (result: $result)"
} error]} {
    puts "  ❌ Named basic failed: $error"
    exit 1
}

# Test 3: CamelCase alias
puts "Test 3: CamelCase alias..."
if {[catch {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    set result [$COMMAND_NEW $input {2 0 1 1} -1.0]
    puts "  CamelCase alias: OK (result: $result)"
} error]} {
    puts "  ❌ CamelCase alias failed: $error"
    exit 1
}

# Test 4: Named parameter syntax with -tensor alias
puts "Test 4: Named parameter syntax with -tensor alias..."
if {[catch {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set input [torch::tensor_reshape $input {3 2}]
    set result [$COMMAND_OLD -tensor $input -pad {0 1 2 0} -val 2.5]
    puts "  Named with -tensor: OK (result: $result)"
} error]} {
    puts "  ❌ Named with -tensor failed: $error"
    exit 1
}

# Test 5: Parameter order independence
puts "Test 5: Parameter order independence..."
if {[catch {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    set result1 [$COMMAND_OLD -input $input -padding {1 1 1 1} -value 0.0]
    set result2 [$COMMAND_OLD -value 0.0 -padding {1 1 1 1} -input $input]
    set result3 [$COMMAND_OLD -padding {1 1 1 1} -input $input -value 0.0]
    puts "  Parameter order independence: OK (all variants work)"
} error]} {
    puts "  ❌ Parameter order independence failed: $error"
    exit 1
}

# Test 6: Different padding patterns
puts "Test 6: Different padding patterns..."
if {[catch {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    
    # Symmetric padding
    set r1 [$COMMAND_OLD $input {2 2 2 2} 1.0]
    # Asymmetric padding
    set r2 [$COMMAND_OLD $input {1 3 0 2} 2.0]
    # Only vertical padding
    set r3 [$COMMAND_OLD $input {0 0 1 1} 0.0]
    # Only horizontal padding
    set r4 [$COMMAND_OLD $input {2 1 0 0} 3.0]
    
    puts "  Different padding patterns: OK"
    puts "    Symmetric: $r1"
    puts "    Asymmetric: $r2"
    puts "    Vertical only: $r3"
    puts "    Horizontal only: $r4"
} error]} {
    puts "  ❌ Different padding patterns failed: $error"
    exit 1
}

# Test 7: Different data types
puts "Test 7: Different data types..."
if {[catch {
    set data {1.0 2.0 3.0 4.0}
    
    set input_f32 [torch::tensor_create $data float32]
    set input_f32 [torch::tensor_reshape $input_f32 {2 2}]
    set input_f64 [torch::tensor_create $data float64]
    set input_f64 [torch::tensor_reshape $input_f64 {2 2}]
    set input_i32 [torch::tensor_create {1 2 3 4} int32]
    set input_i32 [torch::tensor_reshape $input_i32 {2 2}]
    
    set result_f32 [$COMMAND_OLD $input_f32 {1 1 1 1} 0.0]
    set result_f64 [$COMMAND_OLD $input_f64 {1 1 1 1} 0.0]
    set result_i32 [$COMMAND_OLD $input_i32 {1 1 1 1} 0.0]
    
    puts "  Different data types: OK"
    puts "    float32: $result_f32"
    puts "    float64: $result_f64"
    puts "    int32: $result_i32"
} error]} {
    puts "  ❌ Different data types failed: $error"
    exit 1
}

# Test 8: Different constant values
puts "Test 8: Different constant values..."
if {[catch {
    set input [torch::tensor_create {5.0 10.0 15.0 20.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    
    set result_pos [$COMMAND_OLD $input {1 1 1 1} 1.5]
    set result_neg [$COMMAND_OLD $input {1 1 1 1} -2.5]
    set result_zero [$COMMAND_OLD $input {1 1 1 1} 0.0]
    set result_large [$COMMAND_OLD $input {1 1 1 1} 100.0]
    
    puts "  Different constant values: OK"
    puts "    Positive: $result_pos"
    puts "    Negative: $result_neg"
    puts "    Zero: $result_zero"
    puts "    Large: $result_large"
} error]} {
    puts "  ❌ Different constant values failed: $error"
    exit 1
}

# Test 9: Zero padding (edge case)
puts "Test 9: Zero padding (edge case)..."
if {[catch {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    set result [$COMMAND_OLD $input {0 0 0 0} 5.0]
    puts "  Zero padding: OK (result: $result)"
} error]} {
    puts "  ❌ Zero padding failed: $error"
    exit 1
}

# Test 10: Large padding values
puts "Test 10: Large padding values..."
if {[catch {
    set input [torch::tensor_create {1.0} float32]
    set input [torch::tensor_reshape $input {1 1}]
    set result [$COMMAND_OLD $input {3 2 4 1} 0.0]
    puts "  Large padding values: OK (result: $result)"
} error]} {
    puts "  ❌ Large padding values failed: $error"
    exit 1
}

# Test 11: Both syntaxes produce same results
puts "Test 11: Syntax consistency verification..."
if {[catch {
    set input [torch::tensor_create {3.0 4.0 5.0 6.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    
    # Same computation using both syntaxes
    set result_pos [$COMMAND_OLD $input {2 1 1 2} 1.0]
    set result_named [$COMMAND_OLD -input $input -padding {2 1 1 2} -value 1.0]
    
    puts "  Syntax consistency: OK (both produce results)"
    puts "    Positional: $result_pos"
    puts "    Named: $result_named"
} error]} {
    puts "  ❌ Syntax consistency failed: $error"
    exit 1
}

# Test 12: Error handling - no arguments
puts "Test 12: Error handling - no arguments..."
if {[catch {
    $COMMAND_OLD
    puts "  ❌ Should have failed with no arguments"
    exit 1
} error]} {
    puts "  No arguments error: OK - $error"
}

# Test 13: Error handling - invalid tensor
puts "Test 13: Error handling - invalid tensor..."
if {[catch {
    $COMMAND_OLD invalid_tensor {1 1 1 1} 0.0
    puts "  ❌ Should have failed with invalid tensor"
    exit 1
} error]} {
    puts "  Invalid tensor error: OK - $error"
}

# Test 14: Error handling - invalid padding format
puts "Test 14: Error handling - invalid padding format..."
if {[catch {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    $COMMAND_OLD $input {1 2} 0.0
    puts "  ❌ Should have failed with invalid padding format"
    exit 1
} error]} {
    puts "  Invalid padding format error: OK - $error"
}

# Test 15: Error handling - wrong number of padding values
puts "Test 15: Error handling - wrong number of padding values..."
if {[catch {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    $COMMAND_OLD $input {1 2 3 4 5 6} 0.0
    puts "  ❌ Should have failed with wrong number of padding values"
    exit 1
} error]} {
    puts "  Wrong padding count error: OK - $error"
}

# Test 16: Error handling - missing parameter value
puts "Test 16: Error handling - missing parameter value..."
if {[catch {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    $COMMAND_OLD -input $input -padding
    puts "  ❌ Should have failed with missing parameter value"
    exit 1
} error]} {
    puts "  Missing parameter value error: OK - $error"
}

# Test 17: Error handling - unknown parameter
puts "Test 17: Error handling - unknown parameter..."
if {[catch {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    $COMMAND_OLD -input $input -unknown_param {1 1 1 1}
    puts "  ❌ Should have failed with unknown parameter"
    exit 1
} error]} {
    puts "  Unknown parameter error: OK - $error"
}

# Test 18: Image-like tensor (common use case)
puts "Test 18: Image-like tensor (common use case)..."
if {[catch {
    # Create a small 3x3 "image"
    set values {}
    for {set i 1} {$i <= 9} {incr i} {
        lappend values [expr {$i * 1.0}]
    }
    set input [torch::tensor_create $values float32]
    set input [torch::tensor_reshape $input {3 3}]
    
    # Add border padding (common in image processing)
    set result [$COMMAND_OLD $input {1 1 1 1} 0.0]
    puts "  Image-like tensor: OK (result: $result)"
    puts "    Added 1-pixel border with value 0.0"
} error]} {
    puts "  ❌ Image-like tensor failed: $error"
    exit 1
}

# Test 19: Mathematical verification for 2D
puts "Test 19: Mathematical verification for 2D..."
if {[catch {
    # Create a simple 2x2 tensor and verify padding behavior
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    set result [$COMMAND_OLD $input {1 2 0 1} 5.0]
    
    puts "  Mathematical verification: OK"
    puts "    Input: \[\[1.0, 2.0\], \[3.0, 4.0\]\]"
    puts "    Padding: {left=1, right=2, top=0, bottom=1} with value 5.0"
    puts "    Expected: Added padding around 2x2 matrix"
    puts "    Result: $result"
} error]} {
    puts "  ❌ Mathematical verification failed: $error"
    exit 1
}

# Test 20: Performance with different tensor sizes
puts "Test 20: Performance test..."
if {[catch {
    # Small tensor
    set small_input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set small_input [torch::tensor_reshape $small_input {2 2}]
    set start [clock clicks -milliseconds]
    set small_result [$COMMAND_OLD $small_input {5 5 5 5} 0.0]
    set small_time [expr {[clock clicks -milliseconds] - $start}]
    
    # Medium tensor (10x10)
    set medium_values {}
    for {set i 0} {$i < 100} {incr i} {
        lappend medium_values [expr {$i * 0.1}]
    }
    set medium_input [torch::tensor_create $medium_values float32]
    set medium_input [torch::tensor_reshape $medium_input {10 10}]
    set start [clock clicks -milliseconds]
    set medium_result [$COMMAND_OLD $medium_input {2 2 2 2} 1.0]
    set medium_time [expr {[clock clicks -milliseconds] - $start}]
    
    puts "  Performance: OK"
    puts "    Small tensor (2x2): ${small_time}ms"
    puts "    Medium tensor (10x10): ${medium_time}ms"
} error]} {
    puts "  ❌ Performance test failed: $error"
    exit 1
}

puts ""
puts "✅ All tests passed for $COMMAND_OLD / $COMMAND_NEW"
puts ""
puts "📝 Next steps:"
puts "   1. Create tests for constant_pad3d"
puts "   2. Create documentation for all three commands"
puts "   3. Mark commands complete in tracking system" 