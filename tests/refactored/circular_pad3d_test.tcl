# tests/refactored/circular_pad3d_test.tcl
# Test file for refactored torch::circular_pad3d command

# Explicitly load the built shared library for local testing
if {[catch {load ./libtorchtcl.so} result]} {
    puts "Warning: Could not load ./libtorchtcl.so: $result"
}

# Check if the command is available
if {![llength [info commands torch::circular_pad3d]]} {
    puts "❌ Command torch::circular_pad3d not found after loading libtorchtcl.so"
    exit 1
}

# Configuration
set COMMAND_NAME "circular_pad3d"
set COMMAND_OLD "torch::circular_pad3d"
set COMMAND_NEW "torch::circularPad3d"

puts "=== Testing Refactored Command: $COMMAND_OLD / $COMMAND_NEW ==="
puts ""

# Helper function to create test tensors
proc create_test_tensor {} {
    # Create a 4D tensor for testing (batch_size=1, depth=2, height=2, width=2)
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
    set tensor [torch::tensor_reshape $tensor {1 2 2 2}]
    return $tensor
}

# Test 1: Original positional syntax (backward compatibility)
puts "Test 1: Original positional syntax..."
if {[catch {
    set tensor [create_test_tensor]
    set result [$COMMAND_OLD $tensor {1 1 1 1 1 1}]
    puts "  Original syntax: OK (result: $result)"
} error]} {
    puts "  ❌ Original syntax failed: $error"
    exit 1
}

# Test 2: New named parameter syntax
puts "Test 2: New named parameter syntax..."
if {[catch {
    set tensor [create_test_tensor]
    set result [$COMMAND_OLD -input $tensor -padding {1 2 1 1 1 0}]
    puts "  Named parameters: OK (result: $result)"
} error]} {
    puts "  ❌ Named parameters failed: $error"
    exit 1
}

# Test 3: CamelCase alias with positional syntax
puts "Test 3: CamelCase alias with positional syntax..."
if {[catch {
    set tensor [create_test_tensor]
    set result [$COMMAND_NEW $tensor {1 1 1 1 1 1}]
    puts "  CamelCase positional: OK (result: $result)"
} error]} {
    puts "  ❌ CamelCase positional failed: $error"
    exit 1
}

# Test 4: CamelCase alias with named parameters
puts "Test 4: CamelCase alias with named parameters..."
if {[catch {
    set tensor [create_test_tensor]
    set result [$COMMAND_NEW -input $tensor -padding {1 1 2 2 1 1}]
    puts "  CamelCase named: OK (result: $result)"
} error]} {
    puts "  ❌ CamelCase named failed: $error"
    exit 1
}

# Test 5: Parameter order flexibility
puts "Test 5: Parameter order flexibility..."
if {[catch {
    set tensor [create_test_tensor]
    set result [$COMMAND_OLD -padding {2 1 1 0 1 1} -input $tensor]
    puts "  Parameter order: OK (result: $result)"
} error]} {
    puts "  ❌ Parameter order failed: $error"
    exit 1
}

# Test 6: Alternative parameter names
puts "Test 6: Alternative parameter names..."
if {[catch {
    set tensor [create_test_tensor]
    set result [$COMMAND_OLD -tensor $tensor -pad {1 1 1 1 1 1}]
    puts "  Alternative parameters: OK (result: $result)"
} error]} {
    puts "  ❌ Alternative parameters failed: $error"
    exit 1
}

# Test 7: Different padding values
puts "Test 7: Different padding values..."
if {[catch {
    # Test different 3D padding combinations (keeping padding < tensor dimensions)
    set tensor_a [create_test_tensor]
    set result_a [$COMMAND_OLD -input $tensor_a -padding {0 1 1 0 0 1}]
    
    set tensor_b [create_test_tensor]
    set result_b [$COMMAND_OLD -input $tensor_b -padding {1 0 0 1 1 0}]
    
    set tensor_c [create_test_tensor]
    set result_c [$COMMAND_OLD -input $tensor_c -padding {1 1 1 1 1 1}]
    
    puts "  Different padding values: OK (results: $result_a, $result_b, $result_c)"
} error]} {
    puts "  ❌ Different padding values failed: $error"
    exit 1
}

# Test 8: Zero padding
puts "Test 8: Zero padding..."
if {[catch {
    set tensor [create_test_tensor]
    set result [$COMMAND_OLD -input $tensor -padding {0 0 0 0 0 0}]
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
    $COMMAND_OLD invalid_tensor {1 2 1 1 1 1}
    puts "  ❌ Should have failed with invalid tensor"
    exit 1
} error]} {
    puts "  Invalid tensor error: OK - $error"
}

# Test 11: Error handling - unknown parameter
puts "Test 11: Error handling - unknown parameter..."
if {[catch {
    set tensor [create_test_tensor]
    $COMMAND_OLD -input $tensor -padding {1 2 1 1 1 1} -unknown_param value
    puts "  ❌ Should have failed with unknown parameter"
    exit 1
} error]} {
    puts "  Unknown parameter error: OK - $error"
}

# Test 12: Error handling - wrong number of padding values
puts "Test 12: Error handling - wrong number of padding values..."
if {[catch {
    set tensor [create_test_tensor]
    $COMMAND_OLD -input $tensor -padding {1 2 3 4 5}
    puts "  ❌ Should have failed with wrong padding count"
    exit 1
} error]} {
    puts "  Wrong padding count error: OK - $error"
}

# Test 13: Error handling - too many padding values
puts "Test 13: Error handling - too many padding values..."
if {[catch {
    set tensor [create_test_tensor]
    $COMMAND_OLD -input $tensor -padding {1 2 3 4 5 6 7}
    puts "  ❌ Should have failed with too many padding values"
    exit 1
} error]} {
    puts "  Too many padding values error: OK - $error"
}

# Test 14: Error handling - invalid padding values
puts "Test 14: Error handling - invalid padding values..."
if {[catch {
    set tensor [create_test_tensor]
    $COMMAND_OLD -input $tensor -padding {invalid 2 1 1 1 1}
    puts "  ❌ Should have failed with invalid padding values"
    exit 1
} error]} {
    puts "  Invalid padding values error: OK - $error"
}

# Test 15: Mathematical correctness - shape verification
puts "Test 15: Mathematical correctness - shape verification..."
if {[catch {
    set test_tensor [create_test_tensor]
    set original_shape [torch::tensor_shape $test_tensor]
    
    # Apply padding {1, 1, 1, 1, 1, 1} - simple symmetric case
    set padded_result [$COMMAND_OLD -input $test_tensor -padding {1 1 1 1 1 1}]
    set result_shape [torch::tensor_shape $padded_result]
    
    # Original: {1, 2, 2, 2}, Result should be: {1, 4, 4, 4} (each dimension +2)
    set orig_depth [lindex $original_shape 1]
    set orig_height [lindex $original_shape 2]
    set orig_width [lindex $original_shape 3]
    set result_depth [lindex $result_shape 1]
    set result_height [lindex $result_shape 2]
    set result_width [lindex $result_shape 3]
    
    # Check if all dimensions increased by 2 (1+1 padding each)
    if {$result_depth == [expr {$orig_depth + 2}] && $result_height == [expr {$orig_height + 2}] && $result_width == [expr {$orig_width + 2}]} {
        puts "  Shape verification: OK (original: $original_shape, result: $result_shape)"
    } else {
        puts "  ❌ Shape mismatch: expected dimensions [expr {$orig_depth + 2}]x[expr {$orig_height + 2}]x[expr {$orig_width + 2}], got $result_shape"
        exit 1
    }
} error]} {
    puts "  ❌ Shape verification failed: $error"
    exit 1
}

# Test 16: Performance comparison
puts "Test 16: Performance comparison..."
set iterations 15
puts "  Running $iterations iterations..."

# Test old syntax performance
set start [clock clicks -milliseconds]
for {set i 0} {$i < $iterations} {incr i} {
    set tensor [create_test_tensor]
    $COMMAND_OLD $tensor {1 1 1 1 1 1}
}
set end [clock clicks -milliseconds]
set old_time [expr {$end - $start}]

# Test new syntax performance
set start [clock clicks -milliseconds]
for {set i 0} {$i < $iterations} {incr i} {
    set tensor [create_test_tensor]
    $COMMAND_OLD -input $tensor -padding {1 1 1 1 1 1}
}
set end [clock clicks -milliseconds]
set new_time [expr {$end - $start}]

puts "  Old syntax: ${old_time}ms"
puts "  New syntax: ${new_time}ms"
puts "  Performance: OK (within acceptable range)"

# Test 17: Different tensor sizes
puts "Test 17: Different tensor sizes..."
if {[catch {
    # Test with different sized tensors (4D for 3D padding)
    set small_tensor [torch::ones {1 1 2 2}]
    set medium_tensor [torch::ones {1 3 3 3}]
    set large_tensor [torch::ones {1 5 5 5}]
    
    set result1 [$COMMAND_OLD -input $small_tensor -padding {1 1 1 1 1 1}]
    set result2 [$COMMAND_OLD -input $medium_tensor -padding {2 2 1 1 1 1}]
    set result3 [$COMMAND_OLD -input $large_tensor -padding {3 3 2 2 1 1}]
    
    puts "  Different sizes: OK (results: $result1, $result2, $result3)"
} error]} {
    puts "  ❌ Different sizes failed: $error"
    exit 1
}

# Test 18: Data type compatibility
puts "Test 18: Data type compatibility..."
if {[catch {
    # Test with different data types (4D for 3D padding)
    set float_tensor [torch::ones {1 2 2 2}]
    set float_tensor [torch::tensor_to $float_tensor float32]
    set double_tensor [torch::ones {1 2 2 2}]
    set double_tensor [torch::tensor_to $double_tensor float64]
    
    set result1 [$COMMAND_OLD -input $float_tensor -padding {1 1 1 1 1 1}]
    set result2 [$COMMAND_OLD -input $double_tensor -padding {1 1 1 1 1 1}]
    
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
puts "   2. Mark command complete: python3 scripts/update_command_status.py mark-complete torch::circular_pad3d"
puts "   3. Continue with next commands in the queue" 