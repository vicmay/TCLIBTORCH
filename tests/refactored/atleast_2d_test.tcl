# tests/refactored/atleast_2d_test.tcl
# Test file for refactored torch::atleast_2d command

# Explicitly load the built shared library for local testing
if {[catch {load ../../build/libtorchtcl.so} result]} {
    puts "Warning: Could not load ../../build/libtorchtcl.so: $result"
}

# Check if the command is available
if {![llength [info commands torch::atleast_2d]]} {
    puts "❌ Command torch::atleast_2d not found after loading libtorchtcl.so"
    exit 1
}

# Configuration
set COMMAND_NAME "atleast_2d"
set COMMAND_OLD "torch::atleast_2d"
set COMMAND_NEW "torch::atleast2d"

puts "=== Testing Refactored Command: $COMMAND_OLD / $COMMAND_NEW ==="
puts ""

# Test 1: Original positional syntax (backward compatibility)
puts "Test 1: Original positional syntax..."
if {[catch {
    set t1 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result1 [$COMMAND_OLD $t1]
    puts "  Original syntax: OK"
} result]} {
    puts "  ❌ Original syntax failed: $result"
    exit 1
}

# Test 2: New named parameter syntax with -input
puts "Test 2: New named parameter syntax with -input..."
if {[catch {
    set t2 [torch::tensorCreate -data {4.0 5.0 6.0} -dtype float32]
    set result2 [$COMMAND_OLD -input $t2]
    puts "  Named parameters (-input): OK"
} result]} {
    puts "  ❌ Named parameters (-input) failed: $result"
    exit 1
}

# Test 3: New named parameter syntax with -tensor
puts "Test 3: New named parameter syntax with -tensor..."
if {[catch {
    set t3 [torch::tensorCreate -data {7.0 8.0 9.0} -dtype float32]
    set result3 [$COMMAND_OLD -tensor $t3]
    puts "  Named parameters (-tensor): OK"
} result]} {
    puts "  ❌ Named parameters (-tensor) failed: $result"
    exit 1
}

# Test 4: camelCase alias with positional syntax
puts "Test 4: camelCase alias with positional syntax..."
if {[catch {
    set t4 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result4 [$COMMAND_NEW $t4]
    puts "  camelCase alias (positional): OK"
} result]} {
    puts "  ❌ camelCase alias (positional) failed: $result"
    exit 1
}

# Test 5: camelCase alias with named parameters
puts "Test 5: camelCase alias with named parameters..."
if {[catch {
    set t5 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result5 [$COMMAND_NEW -input $t5]
    puts "  camelCase alias (named): OK"
} result]} {
    puts "  ❌ camelCase alias (named) failed: $result"
    exit 1
}

# Test 6: Different tensor shapes - scalar
puts "Test 6: Scalar tensor (should become 2D)..."
if {[catch {
    set scalar [torch::tensorCreate -data 5.0 -dtype float32]
    set result6 [$COMMAND_OLD -input $scalar]
    set shape [torch::tensor_shape $result6]
    puts "  Scalar -> 2D: OK (shape: $shape)"
} result]} {
    puts "  ❌ Scalar tensor test failed: $result"
    exit 1
}

# Test 7: Different tensor shapes - 1D tensor  
puts "Test 7: 1D tensor (should become 2D)..."
if {[catch {
    set t1d [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result7 [$COMMAND_OLD -input $t1d]
    set shape [torch::tensor_shape $result7]
    puts "  1D -> 2D: OK (shape: $shape)"
} result]} {
    puts "  ❌ 1D tensor test failed: $result"
    exit 1
}

# Test 8: Different tensor shapes - already 2D tensor
puts "Test 8: 2D tensor (should remain unchanged)..."
if {[catch {
    set t2d [torch::zeros -shape {2 3} -dtype float32]
    set result8 [$COMMAND_OLD -input $t2d]
    set shape [torch::tensor_shape $result8]
    puts "  2D -> 2D: OK (shape: $shape)"
} result]} {
    puts "  ❌ 2D tensor test failed: $result"
    exit 1
}

# Test 9: Different tensor shapes - 3D tensor
puts "Test 9: 3D tensor (should remain unchanged)..."
if {[catch {
    set t3d [torch::zeros -shape {2 2 2} -dtype float32]
    set result9 [$COMMAND_OLD -input $t3d]
    set shape [torch::tensor_shape $result9]
    puts "  3D -> 3D: OK (shape: $shape)"
} result]} {
    puts "  ❌ 3D tensor test failed: $result"
    exit 1
}

# Test 10: Error handling - invalid parameter names
puts "Test 10: Error handling - invalid parameter names..."
if {[catch {
    set t10 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    $COMMAND_OLD -invalid_param $t10
    puts "  ❌ Should have failed with invalid parameter"
    exit 1
} result]} {
    puts "  Error handling: OK - $result"
}

# Test 11: Error handling - missing required parameters
puts "Test 11: Error handling - missing required parameters..."
if {[catch {
    $COMMAND_OLD
    puts "  ❌ Should have failed with missing tensor"
    exit 1
} result]} {
    puts "  Error handling: OK - $result"
}

# Test 12: Error handling - nonexistent tensor
puts "Test 12: Error handling - nonexistent tensor..."
if {[catch {
    $COMMAND_OLD -input nonexistent_tensor
    puts "  ❌ Should have failed with nonexistent tensor"
    exit 1
} result]} {
    puts "  Error handling: OK - $result"
}

# Test 13: Consistency between syntaxes
puts "Test 13: Consistency between syntaxes..."
if {[catch {
    set t13 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    
    # Same operation using different syntaxes - all should succeed
    set result_pos [$COMMAND_OLD $t13]
    set result_named [$COMMAND_OLD -input $t13]
    set result_camel [$COMMAND_NEW -input $t13]
    
    # Check that all results have the same shape (should be 2D)
    set shape_pos [torch::tensor_shape $result_pos]
    set shape_named [torch::tensor_shape $result_named]
    set shape_camel [torch::tensor_shape $result_camel]
    
    if {$shape_pos eq $shape_named && $shape_named eq $shape_camel} {
        puts "  Consistency between syntaxes: OK (all shapes: $shape_pos)"
    } else {
        puts "  ❌ Shapes differ between syntaxes"
        exit 1
    }
} result]} {
    puts "  ❌ Consistency test failed: $result"
    exit 1
}

# Test 14: Performance comparison
puts "Test 14: Performance comparison..."
set iterations 1000
puts "  Running $iterations iterations..."

if {[catch {
    set t14 [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32]
    
    # Test old syntax performance
    set start [clock clicks -milliseconds]
    for {set i 0} {$i < $iterations} {incr i} {
        $COMMAND_OLD $t14
    }
    set end [clock clicks -milliseconds]
    set old_time [expr {$end - $start}]
    
    # Test new syntax performance
    set start [clock clicks -milliseconds]
    for {set i 0} {$i < $iterations} {incr i} {
        $COMMAND_OLD -input $t14
    }
    set end [clock clicks -milliseconds]
    set new_time [expr {$end - $start}]
    
    puts "  Old syntax: ${old_time}ms"
    puts "  New syntax: ${new_time}ms"
    
    # Check that performance is reasonable (less than 5x difference)
    if {$new_time <= [expr {$old_time * 5}] && $old_time <= [expr {$new_time * 5}]} {
        puts "  Performance: OK (within acceptable range)"
    } else {
        puts "  ⚠️  Performance difference may be too large"
    }
} result]} {
    puts "  ❌ Performance test failed: $result"
    exit 1
}

# Test 15: Different data types
puts "Test 15: Different data types..."
if {[catch {
    # Test with int32
    set t_int [torch::tensorCreate -data {1 2 3} -dtype int32]
    set result_int [$COMMAND_OLD -input $t_int]
    
    # Test with float64
    set t_float64 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float64]
    set result_float64 [$COMMAND_OLD -input $t_float64]
    
    puts "  Different data types: OK"
} result]} {
    puts "  ❌ Different data types test failed: $result"
    exit 1
}

puts ""
puts "✅ All tests passed for $COMMAND_OLD / $COMMAND_NEW"
puts ""
puts "📝 Next steps:"
puts "   1. Create documentation: docs/refactored/${COMMAND_NAME}.md"
puts "   2. Update tracking: mark command complete in database"
puts "   3. Commit changes: git add ."
puts "   4. Move to next command: python3 scripts/query_next_commands.py" 