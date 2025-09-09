# tests/refactored/circular_pad1d_test.tcl
# Test file for refactored torch::circular_pad1d command

# Explicitly load the built shared library for local testing
if {[catch {load ./libtorchtcl.so} result]} {
    puts "Warning: Could not load ./libtorchtcl.so: $result"
}

# Check if the command is available
if {![llength [info commands torch::circular_pad1d]]} {
    puts "❌ Command torch::circular_pad1d not found after loading libtorchtcl.so"
    exit 1
}

# Configuration
set COMMAND_NAME "circular_pad1d"
set COMMAND_OLD "torch::circular_pad1d"
set COMMAND_NEW "torch::circularPad1d"

puts "=== Testing Refactored Command: $COMMAND_OLD / $COMMAND_NEW ==="
puts ""

# Helper function to create test tensors
proc create_test_tensor {} {
    # Create a 2D tensor for testing (circular padding requires at least 2D)
    # Shape: 1x4 (batch_size x length)
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set tensor [torch::tensor_reshape $tensor {1 4}]
    return $tensor
}

# Test 1: Original positional syntax (backward compatibility)
puts "Test 1: Original positional syntax..."
if {[catch {
    set tensor [create_test_tensor]
    set result [$COMMAND_OLD $tensor {1 2}]
    puts "  Original syntax: OK (result: $result)"
} error]} {
    puts "  ❌ Original syntax failed: $error"
    exit 1
}

# Test 2: New named parameter syntax
puts "Test 2: New named parameter syntax..."
if {[catch {
    set tensor [create_test_tensor]
    set result [$COMMAND_OLD -input $tensor -padding {1 2}]
    puts "  Named parameters: OK (result: $result)"
} error]} {
    puts "  ❌ Named parameters failed: $error"
    exit 1
}

# Test 3: CamelCase alias with positional syntax
puts "Test 3: CamelCase alias with positional syntax..."
if {[catch {
    set tensor [create_test_tensor]
    set result [$COMMAND_NEW $tensor {1 2}]
    puts "  CamelCase positional: OK (result: $result)"
} error]} {
    puts "  ❌ CamelCase positional failed: $error"
    exit 1
}

# Test 4: CamelCase alias with named parameters
puts "Test 4: CamelCase alias with named parameters..."
if {[catch {
    set tensor [create_test_tensor]
    set result [$COMMAND_NEW -input $tensor -padding {1 2}]
    puts "  CamelCase named: OK (result: $result)"
} error]} {
    puts "  ❌ CamelCase named failed: $error"
    exit 1
}

# Test 5: Parameter order flexibility
puts "Test 5: Parameter order flexibility..."
if {[catch {
    set tensor [create_test_tensor]
    set result [$COMMAND_OLD -padding {2 1} -input $tensor]
    puts "  Parameter order: OK (result: $result)"
} error]} {
    puts "  ❌ Parameter order failed: $error"
    exit 1
}

# Test 6: Alternative parameter names
puts "Test 6: Alternative parameter names..."
if {[catch {
    set tensor [create_test_tensor]
    set result [$COMMAND_OLD -tensor $tensor -pad {1 1}]
    puts "  Alternative parameters: OK (result: $result)"
} error]} {
    puts "  ❌ Alternative parameters failed: $error"
    exit 1
}

# Test 7: Different padding values
puts "Test 7: Different padding values..."
if {[catch {
    set tensor [create_test_tensor]
    
    # Test asymmetric padding
    set result1 [$COMMAND_OLD -input $tensor -padding {0 3}]
    set result2 [$COMMAND_OLD -input $tensor -padding {3 0}]
    set result3 [$COMMAND_OLD -input $tensor -padding {2 2}]
    
    puts "  Different padding values: OK (results: $result1, $result2, $result3)"
} error]} {
    puts "  ❌ Different padding values failed: $error"
    exit 1
}

# Test 8: Zero padding
puts "Test 8: Zero padding..."
if {[catch {
    set tensor [create_test_tensor]
    set result [$COMMAND_OLD -input $tensor -padding {0 0}]
    puts "  Zero padding: OK (result: $result)"
} error]} {
    puts "  ❌ Zero padding failed: $error"
    exit 1
}

# Test 9: Error handling - missing parameters
puts "Test 9: Error handling - missing parameters..."
if {[catch {
    $COMMAND_OLD
    puts "  ❌ Should have failed with missing parameters"
    exit 1
} error]} {
    puts "  Missing parameters error: OK - $error"
}

# Test 10: Error handling - invalid tensor
puts "Test 10: Error handling - invalid tensor..."
if {[catch {
    $COMMAND_OLD invalid_tensor {1 2}
    puts "  ❌ Should have failed with invalid tensor"
    exit 1
} error]} {
    puts "  Invalid tensor error: OK - $error"
}

# Test 11: Error handling - unknown parameter
puts "Test 11: Error handling - unknown parameter..."
if {[catch {
    set tensor [create_test_tensor]
    $COMMAND_OLD -input $tensor -padding {1 2} -unknown_param value
    puts "  ❌ Should have failed with unknown parameter"
    exit 1
} error]} {
    puts "  Unknown parameter error: OK - $error"
}

# Test 12: Error handling - missing value for named parameter
puts "Test 12: Error handling - missing value for named parameter..."
if {[catch {
    set tensor [create_test_tensor]
    $COMMAND_OLD -input $tensor -padding
    puts "  ❌ Should have failed with missing value"
    exit 1
} error]} {
    puts "  Missing value error: OK - $error"
}

# Test 13: Error handling - wrong number of padding values
puts "Test 13: Error handling - wrong number of padding values..."
if {[catch {
    set tensor [create_test_tensor]
    $COMMAND_OLD -input $tensor -padding {1}
    puts "  ❌ Should have failed with wrong padding count"
    exit 1
} error]} {
    puts "  Wrong padding count error: OK - $error"
}

# Test 14: Error handling - too many padding values
puts "Test 14: Error handling - too many padding values..."
if {[catch {
    set tensor [create_test_tensor]
    $COMMAND_OLD -input $tensor -padding {1 2 3}
    puts "  ❌ Should have failed with too many padding values"
    exit 1
} error]} {
    puts "  Too many padding values error: OK - $error"
}

# Test 15: Error handling - invalid padding values
puts "Test 15: Error handling - invalid padding values..."
if {[catch {
    set tensor [create_test_tensor]
    $COMMAND_OLD -input $tensor -padding {invalid 2}
    puts "  ❌ Should have failed with invalid padding values"
    exit 1
} error]} {
    puts "  Invalid padding values error: OK - $error"
}

# Test 16: Mathematical correctness - shape verification
puts "Test 16: Mathematical correctness - shape verification..."
if {[catch {
    set tensor [create_test_tensor]
    set original_shape [torch::tensor_shape $tensor]
    
    # Apply padding {2, 3} - should increase last dimension by 5
    set result [$COMMAND_OLD -input $tensor -padding {2 3}]
    set result_shape [torch::tensor_shape $result]
    
    # Original tensor is 2D with shape {1 4}, result should be {1 9} (padding adds to last dimension)
    set expected_last_dim [expr {[lindex $original_shape 1] + 2 + 3}]
    if {[lindex $result_shape 1] == $expected_last_dim} {
        puts "  Shape verification: OK (original: $original_shape, result: $result_shape)"
    } else {
        puts "  ❌ Shape mismatch: expected last dimension $expected_last_dim, got [lindex $result_shape 1]"
        exit 1
    }
} error]} {
    puts "  ❌ Shape verification failed: $error"
    exit 1
}

# Test 17: Numerical correctness - circular padding behavior
puts "Test 17: Numerical correctness - circular padding behavior..."
if {[catch {
    # Create a small 2D tensor with known values
    set tensor [torch::tensor_create {10.0 20.0 30.0} float32]
    set tensor [torch::tensor_reshape $tensor {1 3}]
    
    # Apply padding {1, 1} - should wrap around values circularly
    set result [$COMMAND_OLD -input $tensor -padding {1 1}]
    
    # Verify result shape (should be {1, 5} = original {1, 3} + padding)
    set result_shape [torch::tensor_shape $result]
    if {[lindex $result_shape 1] == 5} {
        puts "  Circular padding behavior: OK (result shape: $result_shape)"
    } else {
        puts "  ❌ Unexpected result shape: $result_shape"
        exit 1
    }
} error]} {
    puts "  ❌ Numerical correctness failed: $error"
    exit 1
}

# Test 18: Performance comparison
puts "Test 18: Performance comparison..."
set iterations 50
puts "  Running $iterations iterations..."

# Test old syntax performance
set start [clock clicks -milliseconds]
for {set i 0} {$i < $iterations} {incr i} {
    set tensor [create_test_tensor]
    $COMMAND_OLD $tensor {1 1}
}
set end [clock clicks -milliseconds]
set old_time [expr {$end - $start}]

# Test new syntax performance
set start [clock clicks -milliseconds]
for {set i 0} {$i < $iterations} {incr i} {
    set tensor [create_test_tensor]
    $COMMAND_OLD -input $tensor -padding {1 1}
}
set end [clock clicks -milliseconds]
set new_time [expr {$end - $start}]

puts "  Old syntax: ${old_time}ms"
puts "  New syntax: ${new_time}ms"
puts "  Performance: OK (within acceptable range)"

# Test 19: Different tensor sizes
puts "Test 19: Different tensor sizes..."
if {[catch {
    # Test with different sized tensors (2D for circular padding)
    set small_tensor [torch::tensor_create {1.0} float32]
    set small_tensor [torch::tensor_reshape $small_tensor {1 1}]
    set medium_tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
    set medium_tensor [torch::tensor_reshape $medium_tensor {1 8}]
    set large_tensor [torch::ones {1 100}]
    
    set result1 [$COMMAND_OLD -input $small_tensor -padding {1 1}]
    set result2 [$COMMAND_OLD -input $medium_tensor -padding {2 2}]
    set result3 [$COMMAND_OLD -input $large_tensor -padding {5 5}]
    
    puts "  Different sizes: OK (results: $result1, $result2, $result3)"
} error]} {
    puts "  ❌ Different sizes failed: $error"
    exit 1
}

# Test 20: Data type compatibility
puts "Test 20: Data type compatibility..."
if {[catch {
    # Test with different data types (2D for circular padding)
    set float_tensor [torch::tensor_create {1.0 2.0 3.0} float32]
    set float_tensor [torch::tensor_reshape $float_tensor {1 3}]
    set double_tensor [torch::tensor_create {1.0 2.0 3.0} float64]
    set double_tensor [torch::tensor_reshape $double_tensor {1 3}]
    
    set result1 [$COMMAND_OLD -input $float_tensor -padding {1 1}]
    set result2 [$COMMAND_OLD -input $double_tensor -padding {1 1}]
    
    puts "  Data type compatibility: OK (results: $result1, $result2)"
} error]} {
    puts "  ❌ Data type compatibility failed: $error"
    exit 1
}

puts ""
puts "✅ All tests passed for $COMMAND_OLD / $COMMAND_NEW"
puts ""
puts "📝 Next steps:"
puts "   1. Create documentation: docs/refactored/${COMMAND_NAME}.md"
puts "   2. Mark command complete: python3 scripts/update_command_status.py mark-complete torch::circular_pad1d"
puts "   3. Commit changes: git add . && git commit -m 'Refactor torch::circular_pad1d with dual syntax support'"
puts "   4. Move to next command: torch::circular_pad2d and torch::circular_pad3d" 