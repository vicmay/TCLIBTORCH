# tests/refactored/all_reduce_test.tcl
# Test file for refactored torch::all_reduce command

# Explicitly load the built shared library for local testing
if {[catch {load ../../build/libtorchtcl.so} result]} {
    puts "Warning: Could not load ../../build/libtorchtcl.so: $result"
}

# Check if the command is available
if {![llength [info commands torch::all_reduce]]} {
    puts "❌ Command torch::all_reduce not found after loading libtorchtcl.so"
    exit 1
}

# Configuration
set COMMAND_NAME "all_reduce"
set COMMAND_OLD "torch::all_reduce"
set COMMAND_NEW "torch::allReduce"

puts "=== Testing Refactored Command: $COMMAND_OLD / $COMMAND_NEW ==="
puts ""

# Initialize distributed training (required for all_reduce)
puts "Setup: Initializing distributed training..."
if {[catch {
    torch::distributed_init 0 1 localhost 29500
    puts "  Distributed training initialized: OK"
} result]} {
    puts "  ❌ Distributed initialization failed: $result"
    exit 1
}

# Test 1: Original positional syntax (backward compatibility)
puts "Test 1: Original positional syntax..."
if {[catch {
    set t1 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result1 [$COMMAND_OLD $t1]
    puts "  Original syntax (default operation): OK"
} result]} {
    puts "  ❌ Original syntax failed: $result"
    exit 1
}

# Test 2: Original positional syntax with explicit operation
puts "Test 2: Original positional syntax with operation..."
if {[catch {
    set t2 [torch::tensorCreate -data {2.0 4.0 6.0} -dtype float32]
    set result2 [$COMMAND_OLD $t2 mean]
    puts "  Original syntax (mean operation): OK"
} result]} {
    puts "  ❌ Original syntax with operation failed: $result"
    exit 1
}

# Test 3: New named parameter syntax
puts "Test 3: New named parameter syntax..."
if {[catch {
    set t3 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result3 [$COMMAND_OLD -tensor $t3 -operation sum]
    puts "  Named parameters: OK"
} result]} {
    puts "  ❌ Named parameters failed: $result"
    exit 1
}

# Test 4: camelCase alias
puts "Test 4: camelCase alias..."
if {[catch {
    set t4 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result4 [$COMMAND_NEW -tensor $t4 -operation sum]
    puts "  camelCase alias: OK"
} result]} {
    puts "  ❌ camelCase alias failed: $result"
    exit 1
}

# Test 5: Default parameters in named syntax
puts "Test 5: Default parameters in named syntax..."
if {[catch {
    set t5 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result5 [$COMMAND_OLD -tensor $t5]
    puts "  Default parameters: OK"
} result]} {
    puts "  ❌ Default parameters failed: $result"
    exit 1
}

# Test 6: Parameter order independence
puts "Test 6: Parameter order independence..."
if {[catch {
    set t6 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    set result6 [$COMMAND_OLD -operation sum -tensor $t6]
    puts "  Parameter order independence: OK"
} result]} {
    puts "  ❌ Parameter order independence failed: $result"
    exit 1
}

# Test 7: Error handling - invalid parameter names
puts "Test 7: Error handling - invalid parameter names..."
if {[catch {
    set t7 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    $COMMAND_OLD -tensor $t7 -invalid_param value
    puts "  ❌ Should have failed with invalid parameter"
    exit 1
} result]} {
    puts "  Error handling: OK - $result"
}

# Test 8: Error handling - missing required parameters
puts "Test 8: Error handling - missing required parameters..."
if {[catch {
    $COMMAND_OLD -operation sum
    puts "  ❌ Should have failed with missing tensor"
    exit 1
} result]} {
    puts "  Error handling: OK - $result"
}

# Test 9: Error handling - invalid operation
puts "Test 9: Error handling - invalid operation..."
if {[catch {
    set t9 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    $COMMAND_OLD -tensor $t9 -operation invalid_operation
    puts "  ❌ Should have failed with invalid operation"
    exit 1
} result]} {
    puts "  Error handling: OK - $result"
}

# Test 10: Different reduction operations
puts "Test 10: Different reduction operations..."
if {[catch {
    set t10 [torch::tensorCreate -data {1.0 5.0 3.0} -dtype float32]
    
    # Test sum
    set sum_result [$COMMAND_OLD -tensor $t10 -operation sum]
    
    # Test mean
    set mean_result [$COMMAND_OLD -tensor $t10 -operation mean]
    
    # Test max
    set max_result [$COMMAND_OLD -tensor $t10 -operation max]
    
    # Test min
    set min_result [$COMMAND_OLD -tensor $t10 -operation min]
    
    puts "  All reduction operations: OK"
} result]} {
    puts "  ❌ Reduction operations failed: $result"
    exit 1
}

# Test 11: Consistency between syntaxes
puts "Test 11: Consistency between syntaxes..."
if {[catch {
    set t11 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
    
    # Same operation using different syntaxes - all should succeed
    set result_pos [$COMMAND_OLD $t11 sum]
    set result_named [$COMMAND_OLD -tensor $t11 -operation sum]
    set result_camel [$COMMAND_NEW -tensor $t11 -operation sum]
    
    # If all three succeeded, the syntaxes are consistent
    puts "  Consistency between syntaxes: OK"
} result]} {
    puts "  ❌ Consistency test failed: $result"
    exit 1
}

# Test 12: Performance comparison
puts "Test 12: Performance comparison..."
set iterations 100
puts "  Running $iterations iterations..."

if {[catch {
    set t12 [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32]
    
    # Test old syntax performance
    set start [clock clicks -milliseconds]
    for {set i 0} {$i < $iterations} {incr i} {
        $COMMAND_OLD $t12 sum
    }
    set end [clock clicks -milliseconds]
    set old_time [expr {$end - $start}]
    
    # Test new syntax performance
    set start [clock clicks -milliseconds]
    for {set i 0} {$i < $iterations} {incr i} {
        $COMMAND_OLD -tensor $t12 -operation sum
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

puts ""
puts "✅ All tests passed for $COMMAND_OLD / $COMMAND_NEW"
puts ""
puts "📝 Next steps:"
puts "   1. Create documentation: docs/refactored/${COMMAND_NAME}.md"
puts "   2. Update tracking: mark command complete in database"
puts "   3. Commit changes: git add ."
puts "   4. Move to next command: python3 scripts/query_next_commands.py" 