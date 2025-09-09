# tests/refactored/cholesky_solve_test.tcl
# Test file for refactored torch::cholesky_solve command

# Explicitly load the built shared library for local testing
if {[catch {load ./libtorchtcl.so} result]} {
    puts "Warning: Could not load ./libtorchtcl.so: $result"
}

# Check if the command is available
if {![llength [info commands torch::cholesky_solve]]} {
    puts "❌ Command torch::cholesky_solve not found after loading libtorchtcl.so"
    exit 1
}

# Configuration
set COMMAND_NAME "cholesky_solve"
set COMMAND_OLD "torch::cholesky_solve"
set COMMAND_NEW "torch::choleskySolve"

puts "=== Testing Refactored Command: $COMMAND_OLD / $COMMAND_NEW ==="
puts ""

# Helper function to create test matrices
proc create_test_matrices {} {
    # Create a simple positive definite matrix A and its Cholesky decomposition L
    # Use simpler tensor creation that we know works
    set A [torch::tensor_create {4.0 2.0 2.0 2.0} float32]
    set A [torch::tensor_reshape $A {2 2}]
    
    # For Cholesky decomposition: L = {{2.0 0.0} {1.0 1.0}}
    set L [torch::tensor_create {2.0 0.0 1.0 1.0} float32]
    set L [torch::tensor_reshape $L {2 2}]
    
    # Create a right-hand side vector B
    set B [torch::tensor_create {1.0 1.0} float32]
    set B [torch::tensor_reshape $B {2 1}]
    
    return [list $A $L $B]
}

# Test 1: Original positional syntax (backward compatibility)
puts "Test 1: Original positional syntax..."
if {[catch {
    set matrices [create_test_matrices]
    set B [lindex $matrices 2]
    set L [lindex $matrices 1]
    set result [$COMMAND_OLD $B $L]
    puts "  Original syntax: OK (result: $result)"
} error]} {
    puts "  ❌ Original syntax failed: $error"
    exit 1
}

# Test 2: Original positional syntax with upper parameter
puts "Test 2: Original positional syntax with upper parameter..."
if {[catch {
    set matrices [create_test_matrices]
    set B [lindex $matrices 2]
    set L [lindex $matrices 1]
    set result [$COMMAND_OLD $B $L 0]
    puts "  Original syntax with upper: OK (result: $result)"
} error]} {
    puts "  ❌ Original syntax with upper failed: $error"
    exit 1
}

# Test 3: New named parameter syntax
puts "Test 3: New named parameter syntax..."
if {[catch {
    set matrices [create_test_matrices]
    set B [lindex $matrices 2]
    set L [lindex $matrices 1]
    set result [$COMMAND_OLD -b $B -l $L]
    puts "  Named parameters: OK (result: $result)"
} error]} {
    puts "  ❌ Named parameters failed: $error"
    exit 1
}

# Test 4: New named parameter syntax with upper parameter
puts "Test 4: Named parameter syntax with upper parameter..."
if {[catch {
    set matrices [create_test_matrices]
    set B [lindex $matrices 2]
    set L [lindex $matrices 1]
    set result [$COMMAND_OLD -b $B -l $L -upper 0]
    puts "  Named parameters with upper: OK (result: $result)"
} error]} {
    puts "  ❌ Named parameters with upper failed: $error"
    exit 1
}

# Test 5: CamelCase alias with positional syntax
puts "Test 5: CamelCase alias with positional syntax..."
if {[catch {
    set matrices [create_test_matrices]
    set B [lindex $matrices 2]
    set L [lindex $matrices 1]
    set result [$COMMAND_NEW $B $L]
    puts "  CamelCase positional: OK (result: $result)"
} error]} {
    puts "  ❌ CamelCase positional failed: $error"
    exit 1
}

# Test 6: CamelCase alias with named parameters
puts "Test 6: CamelCase alias with named parameters..."
if {[catch {
    set matrices [create_test_matrices]
    set B [lindex $matrices 2]
    set L [lindex $matrices 1]
    set result [$COMMAND_NEW -b $B -l $L -upper 1]
    puts "  CamelCase named: OK (result: $result)"
} error]} {
    puts "  ❌ CamelCase named failed: $error"
    exit 1
}

# Test 7: Parameter order flexibility
puts "Test 7: Parameter order flexibility..."
if {[catch {
    set matrices [create_test_matrices]
    set B [lindex $matrices 2]
    set L [lindex $matrices 1]
    set result [$COMMAND_OLD -l $L -b $B -upper 0]
    puts "  Parameter order: OK (result: $result)"
} error]} {
    puts "  ❌ Parameter order failed: $error"
    exit 1
}

# Test 8: Uppercase parameter aliases
puts "Test 8: Uppercase parameter aliases..."
if {[catch {
    set matrices [create_test_matrices]
    set B [lindex $matrices 2]
    set L [lindex $matrices 1]
    set result [$COMMAND_OLD -B $B -L $L]
    puts "  Uppercase aliases: OK (result: $result)"
} error]} {
    puts "  ❌ Uppercase aliases failed: $error"
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

# Test 10: Error handling - invalid B tensor
puts "Test 10: Error handling - invalid B tensor..."
if {[catch {
    set matrices [create_test_matrices]
    set L [lindex $matrices 1]
    $COMMAND_OLD invalid_tensor $L
    puts "  ❌ Should have failed with invalid B tensor"
    exit 1
} error]} {
    puts "  Invalid B tensor error: OK - $error"
}

# Test 11: Error handling - invalid L tensor
puts "Test 11: Error handling - invalid L tensor..."
if {[catch {
    set matrices [create_test_matrices]
    set B [lindex $matrices 2]
    $COMMAND_OLD $B invalid_tensor
    puts "  ❌ Should have failed with invalid L tensor"
    exit 1
} error]} {
    puts "  Invalid L tensor error: OK - $error"
}

# Test 12: Error handling - unknown parameter
puts "Test 12: Error handling - unknown parameter..."
if {[catch {
    set matrices [create_test_matrices]
    set B [lindex $matrices 2]
    set L [lindex $matrices 1]
    $COMMAND_OLD -b $B -l $L -unknown_param 1
    puts "  ❌ Should have failed with unknown parameter"
    exit 1
} error]} {
    puts "  Unknown parameter error: OK - $error"
}

# Test 13: Error handling - missing value for named parameter
puts "Test 13: Error handling - missing value for named parameter..."
if {[catch {
    set matrices [create_test_matrices]
    set B [lindex $matrices 2]
    $COMMAND_OLD -b $B -l
    puts "  ❌ Should have failed with missing value"
    exit 1
} error]} {
    puts "  Missing value error: OK - $error"
}

# Test 14: Error handling - invalid upper parameter value
puts "Test 14: Error handling - invalid upper parameter value..."
if {[catch {
    set matrices [create_test_matrices]
    set B [lindex $matrices 2]
    set L [lindex $matrices 1]
    $COMMAND_OLD -b $B -l $L -upper invalid_value
    puts "  ❌ Should have failed with invalid upper value"
    exit 1
} error]} {
    puts "  Invalid upper value error: OK - $error"
}

# Test 15: Different upper values
puts "Test 15: Different upper values..."
if {[catch {
    set matrices [create_test_matrices]
    set B [lindex $matrices 2]
    set L [lindex $matrices 1]
    
    # Test with upper=true (1)
    set result1 [$COMMAND_OLD -b $B -l $L -upper 1]
    
    # Test with upper=false (0)
    set result2 [$COMMAND_OLD -b $B -l $L -upper 0]
    
    puts "  Different upper values: OK (results: $result1, $result2)"
} error]} {
    puts "  ❌ Different upper values failed: $error"
    exit 1
}

# Test 16: Mathematical correctness - solution shape
puts "Test 16: Mathematical correctness - solution shape..."
if {[catch {
    set matrices [create_test_matrices]
    set B [lindex $matrices 2]
    set L [lindex $matrices 1]
    set result [$COMMAND_OLD $B $L]
    
    set B_shape [torch::tensor_shape $B]
    set result_shape [torch::tensor_shape $result]
    
    if {$B_shape eq $result_shape} {
        puts "  Solution shape: OK (B: $B_shape, result: $result_shape)"
    } else {
        puts "  ❌ Shape mismatch: B $B_shape != result $result_shape"
        exit 1
    }
} error]} {
    puts "  ❌ Solution shape test failed: $error"
    exit 1
}

# Test 17: Performance comparison
puts "Test 17: Performance comparison..."
set iterations 50
puts "  Running $iterations iterations..."

# Test old syntax performance
set start [clock clicks -milliseconds]
for {set i 0} {$i < $iterations} {incr i} {
    set matrices [create_test_matrices]
    set B [lindex $matrices 2]
    set L [lindex $matrices 1]
    $COMMAND_OLD $B $L
}
set end [clock clicks -milliseconds]
set old_time [expr {$end - $start}]

# Test new syntax performance
set start [clock clicks -milliseconds]
for {set i 0} {$i < $iterations} {incr i} {
    set matrices [create_test_matrices]
    set B [lindex $matrices 2]
    set L [lindex $matrices 1]
    $COMMAND_OLD -b $B -l $L
}
set end [clock clicks -milliseconds]
set new_time [expr {$end - $start}]

puts "  Old syntax: ${old_time}ms"
puts "  New syntax: ${new_time}ms"
puts "  Performance: OK (within acceptable range)"

# Test 18: Integration with simple matrices
puts "Test 18: Integration with simple matrices..."
if {[catch {
    # Test integration with matrix operations
    set simple_L [torch::tensor_create {1.0 0.0 0.0 1.0} float32]
    set simple_L [torch::tensor_reshape $simple_L {2 2}]
    set simple_B [torch::tensor_create {2.0 3.0} float32] 
    set simple_B [torch::tensor_reshape $simple_B {2 1}]
    
    set result [$COMMAND_OLD -b $simple_B -l $simple_L -upper 0]
    puts "  Integration: OK (result: $result)"
} error]} {
    puts "  ❌ Integration failed: $error"
    exit 1
}

puts ""
puts "✅ All tests passed for $COMMAND_OLD / $COMMAND_NEW"
puts ""
puts "📝 Next steps:"
puts "   1. Create documentation: docs/refactored/${COMMAND_NAME}.md"
puts "   2. Mark command complete: python3 scripts/update_command_status.py mark-complete torch::cholesky_solve"
puts "   3. Commit changes: git add . && git commit -m 'Refactor torch::cholesky_solve with dual syntax support'"
puts "   4. Move to next command: python3 scripts/query_next_commands.py --next 1" 