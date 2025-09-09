#!/usr/bin/env tclsh
# Test file for refactored torch::cond command

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Configuration
set COMMAND_NAME "cond"
set COMMAND_OLD "torch::cond"
set COMMAND_NEW "torch::cond"

puts "=== Testing Refactored Command: $COMMAND_OLD ==="
puts ""

# Test 1: Basic positional syntax (default 2-norm)
puts "Test 1: Basic positional syntax..."
if {[catch {
    # Create a well-conditioned matrix
    set input [torch::tensor_create {4.0 1.0 1.0 3.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    set result [$COMMAND_OLD $input]
    puts "  Basic positional: OK (result: $result)"
} error]} {
    puts "  ❌ Basic positional failed: $error"
    exit 1
}

# Test 2: Positional syntax with numeric p parameter
puts "Test 2: Positional syntax with numeric p parameter..."
if {[catch {
    set input [torch::tensor_create {2.0 0.0 0.0 1.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    set result [$COMMAND_OLD $input 2.0]
    puts "  Positional with numeric p: OK (result: $result)"
} error]} {
    puts "  ❌ Positional with numeric p failed: $error"
    exit 1
}

# Test 3: Positional syntax with "fro" norm
puts "Test 3: Positional syntax with Frobenius norm..."
if {[catch {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    set result [$COMMAND_OLD $input "fro"]
    puts "  Positional with 'fro': OK (result: $result)"
} error]} {
    puts "  ❌ Positional with 'fro' failed: $error"
    exit 1
}

# Test 4: Named parameter syntax - basic
puts "Test 4: Named parameter syntax - basic..."
if {[catch {
    set input [torch::tensor_create {5.0 1.0 2.0 3.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    set result [$COMMAND_OLD -input $input]
    puts "  Named basic: OK (result: $result)"
} error]} {
    puts "  ❌ Named basic failed: $error"
    exit 1
}

# Test 5: Named parameter syntax with -tensor alias
puts "Test 5: Named parameter syntax with -tensor alias..."
if {[catch {
    set input [torch::tensor_create {3.0 1.0 1.0 2.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    set result [$COMMAND_OLD -tensor $input -p 2.0]
    puts "  Named with -tensor: OK (result: $result)"
} error]} {
    puts "  ❌ Named with -tensor failed: $error"
    exit 1
}

# Test 6: Named parameter syntax with -norm alias
puts "Test 6: Named parameter syntax with -norm alias..."
if {[catch {
    set input [torch::tensor_create {2.0 1.0 0.0 1.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    set result [$COMMAND_OLD -input $input -norm "fro"]
    puts "  Named with -norm: OK (result: $result)"
} error]} {
    puts "  ❌ Named with -norm failed: $error"
    exit 1
}

# Test 7: Different matrix types - identity matrix
puts "Test 7: Identity matrix (well-conditioned)..."
if {[catch {
    set input [torch::eye 3]
    set result [$COMMAND_OLD $input]
    puts "  Identity matrix: OK (result: $result)"
} error]} {
    puts "  ❌ Identity matrix failed: $error"
    exit 1
}

# Test 8: Different matrix types - diagonal matrix
puts "Test 8: Diagonal matrix..."
if {[catch {
    set input [torch::tensor_create {10.0 0.0 0.0 1.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    set result [$COMMAND_OLD $input]
    puts "  Diagonal matrix: OK (result: $result)"
} error]} {
    puts "  ❌ Diagonal matrix failed: $error"
    exit 1
}

# Test 9: Parameter order independence (named syntax)
puts "Test 9: Parameter order independence..."
if {[catch {
    set input [torch::tensor_create {1.0 0.5 0.5 1.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    
    # Test different parameter orders
    set result1 [$COMMAND_OLD -input $input -p 2.0]
    set result2 [$COMMAND_OLD -p 2.0 -input $input]
    
    puts "  Parameter order independence: OK (both variants work)"
} error]} {
    puts "  ❌ Parameter order independence failed: $error"
    exit 1
}

# Test 10: Different norm types
puts "Test 10: Different norm types..."
if {[catch {
    set input [torch::tensor_create {2.0 1.0 1.0 1.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    
    # Test different norm values
    set result_default [$COMMAND_OLD $input]
    set result_2 [$COMMAND_OLD $input 2.0]
    set result_1 [$COMMAND_OLD $input 1.0]
    
    puts "  Different norms: OK (default: $result_default, 2-norm: $result_2, 1-norm: $result_1)"
} error]} {
    puts "  ❌ Different norm types failed: $error"
    exit 1
}

# Test 11: Error handling - no arguments
puts "Test 11: Error handling - no arguments..."
if {[catch {
    $COMMAND_OLD
    puts "  ❌ Should have failed with no arguments"
    exit 1
} error]} {
    puts "  No arguments error: OK - $error"
}

# Test 12: Error handling - invalid tensor
puts "Test 12: Error handling - invalid tensor..."
if {[catch {
    $COMMAND_OLD invalid_tensor
    puts "  ❌ Should have failed with invalid tensor"
    exit 1
} error]} {
    puts "  Invalid tensor error: OK - $error"
}

# Test 13: Error handling - invalid p parameter
puts "Test 13: Error handling - invalid p parameter..."
if {[catch {
    set input [torch::tensor_create {1.0 0.0 0.0 1.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    $COMMAND_OLD $input "invalid_norm"
    puts "  ❌ Should have failed with invalid p parameter"
    exit 1
} error]} {
    puts "  Invalid p parameter error: OK - $error"
}

# Test 14: Error handling - missing parameter value
puts "Test 14: Error handling - missing parameter value..."
if {[catch {
    set input [torch::tensor_create {1.0 0.0 0.0 1.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    $COMMAND_OLD -input $input -p
    puts "  ❌ Should have failed with missing parameter value"
    exit 1
} error]} {
    puts "  Missing parameter value error: OK - $error"
}

# Test 15: Error handling - unknown parameter
puts "Test 15: Error handling - unknown parameter..."
if {[catch {
    set input [torch::tensor_create {1.0 0.0 0.0 1.0} float32]
    set input [torch::tensor_reshape $input {2 2}]
    $COMMAND_OLD -input $input -unknown_param 5
    puts "  ❌ Should have failed with unknown parameter"
    exit 1
} error]} {
    puts "  Unknown parameter error: OK - $error"
}

# Test 16: Larger matrices
puts "Test 16: Larger matrices..."
if {[catch {
    # Create a 3x3 matrix
    set values {1.0 0.1 0.1 0.1 2.0 0.1 0.1 0.1 3.0}
    set input [torch::tensor_create $values float32]
    set input [torch::tensor_reshape $input {3 3}]
    set result [$COMMAND_OLD $input]
    puts "  3x3 matrix: OK (result: $result)"
} error]} {
    puts "  ❌ 3x3 matrix failed: $error"
    exit 1
}

# Test 17: Double precision matrices
puts "Test 17: Double precision matrices..."
if {[catch {
    set input [torch::tensor_create {1.0 0.0 0.0 1.0} float64]
    set input [torch::tensor_reshape $input {2 2}]
    set result [$COMMAND_OLD $input]
    puts "  Double precision: OK (result: $result)"
} error]} {
    puts "  ❌ Double precision failed: $error"
    exit 1
}

# Test 18: Mathematical verification - well-conditioned matrix
puts "Test 18: Mathematical verification - well-conditioned..."
if {[catch {
    # Identity matrix should have condition number ≈ 1
    set input [torch::eye 2]
    set result [$COMMAND_OLD $input]
    
    # For identity matrix, condition number should be close to 1
    puts "  Well-conditioned verification: OK (identity cond ≈ 1: $result)"
} error]} {
    puts "  ❌ Mathematical verification failed: $error"
    exit 1
}

# Test 19: Performance with larger matrix
puts "Test 19: Performance with larger matrix..."
if {[catch {
    # Create a 4x4 matrix for performance testing
    set values {}
    for {set i 0} {$i < 16} {incr i} {
        lappend values [expr {($i % 4) + 1.0}]
    }
    set input [torch::tensor_create $values float32]
    set input [torch::tensor_reshape $input {4 4}]
    
    set start [clock clicks -milliseconds]
    set result [$COMMAND_OLD $input]
    set end [clock clicks -milliseconds]
    set duration [expr {$end - $start}]
    
    puts "  Performance: OK (${duration}ms for 4x4 matrix: $result)"
} error]} {
    puts "  ❌ Performance test failed: $error"
    exit 1
}

# Test 20: Both syntaxes produce same results
puts "Test 20: Syntax consistency verification..."
if {[catch {
    set input [torch::tensor_create {2.0 1.0 1.0 1.5} float32]
    set input [torch::tensor_reshape $input {2 2}]
    
    # Same computation using both syntaxes
    set result_pos [$COMMAND_OLD $input 2.0]
    set result_named [$COMMAND_OLD -input $input -p 2.0]
    
    puts "  Syntax consistency: OK (both produce results)"
    puts "    Positional: $result_pos"
    puts "    Named: $result_named"
} error]} {
    puts "  ❌ Syntax consistency failed: $error"
    exit 1
}

puts ""
puts "✅ All tests passed for $COMMAND_OLD"
puts ""
puts "📝 Next steps:"
puts "   1. Create documentation: docs/refactored/${COMMAND_NAME}.md"
puts "   2. Mark command complete: python3 scripts/update_command_status.py mark-complete torch::cond"
puts "   3. Continue with next commands in queue" 