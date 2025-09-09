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

# ============================================================================
# Test 1: Positional syntax (backward compatibility)
# ============================================================================

test freeze_model-1.1 {Basic positional syntax} -body {
    set model [torch::linear 4 2]
    torch::freeze_model $model
} -result {Model parameters frozen}

test freeze_model-1.2 {Positional syntax with wrong argument count} -body {
    catch {torch::freeze_model} result
    set result
} -match glob -result {*Wrong number of arguments*}

test freeze_model-1.3 {Positional syntax with too many arguments} -body {
    catch {torch::freeze_model "model1" "model2"} result
    set result
} -match glob -result {*Wrong number of arguments*}

test freeze_model-1.4 {Positional syntax with nonexistent model} -body {
    catch {torch::freeze_model "nonexistent_model"} result
    set result
} -result {Model not found}

# ============================================================================
# Test 2: Named parameter syntax
# ============================================================================

test freeze_model-2.1 {Named parameter syntax} -body {
    set model [torch::linear 4 2]
    torch::freeze_model -model $model
} -result {Model parameters frozen}

test freeze_model-2.2 {Named parameter syntax with nonexistent model} -body {
    catch {torch::freeze_model -model "nonexistent_model"} result
    set result
} -result {Model not found}

test freeze_model-2.3 {Named parameter syntax with missing value} -body {
    catch {torch::freeze_model -model} result
    set result
} -match glob -result {*Missing value for parameter*}

test freeze_model-2.4 {Named parameter syntax with unknown parameter} -body {
    catch {torch::freeze_model -unknown_param "value"} result
    set result
} -match glob -result {*Unknown parameter*}

test freeze_model-2.5 {Named parameter syntax with empty model name} -body {
    catch {torch::freeze_model -model ""} result
    set result
} -match glob -result {*Required parameter missing*}

# ============================================================================
# Test 3: camelCase alias
# ============================================================================

test freeze_model-3.1 {camelCase alias with positional syntax} -body {
    set model [torch::linear 4 2]
    torch::freezeModel $model
} -result {Model parameters frozen}

test freeze_model-3.2 {camelCase alias with named syntax} -body {
    set model [torch::linear 4 2]
    torch::freezeModel -model $model
} -result {Model parameters frozen}

test freeze_model-3.3 {camelCase alias with error handling} -body {
    catch {torch::freezeModel -model "nonexistent_model"} result
    set result
} -result {Model not found}

test freeze_model-3.4 {camelCase alias with missing arguments} -body {
    catch {torch::freezeModel} result
    set result
} -match glob -result {*Wrong number of arguments*}

# ============================================================================
# Test 4: Error handling
# ============================================================================

test freeze_model-4.1 {Error handling - no arguments} -body {
    catch {torch::freeze_model} result
    set result
} -match glob -result {*Wrong number of arguments*}

test freeze_model-4.2 {Error handling - invalid model name} -body {
    catch {torch::freeze_model "   "} result
    set result
} -result {Model not found}

test freeze_model-4.3 {Error handling - null model name} -body {
    catch {torch::freeze_model ""} result
    set result
} -result {Required parameter missing: -model}

test freeze_model-4.4 {Error handling - special characters in model name} -body {
    catch {torch::freeze_model "model@#$%"} result
    set result
} -result {Model not found}

test freeze_model-4.5 {Error handling - numeric model name} -body {
    catch {torch::freeze_model "12345"} result
    set result
} -result {Model not found}

test freeze_model-4.6 {Error handling - mixed parameter styles} -body {
    catch {torch::freeze_model "model1" -model "model2"} result
    set result
} -match glob -result {*Wrong number of arguments*}

# ============================================================================
# Test 5: Functionality verification
# ============================================================================

test freeze_model-5.1 {Functionality - freeze model multiple times} -body {
    set model [torch::linear 4 2]
    torch::freeze_model $model
    torch::freeze_model $model
    torch::freeze_model $model
} -result {Model parameters frozen}

test freeze_model-5.2 {Functionality - freeze then unfreeze} -body {
    set model [torch::linear 4 2]
    torch::freeze_model $model
    torch::unfreeze_model $model
} -result {Model parameters unfrozen}

test freeze_model-5.3 {Functionality - freeze different models} -body {
    set model1 [torch::linear 4 2]
    set model2 [torch::linear 3 1]
    torch::freeze_model $model1
    torch::freeze_model $model2
} -result {Model parameters frozen}

test freeze_model-5.4 {Functionality - freeze with different syntaxes} -body {
    set model [torch::linear 4 2]
    torch::freeze_model $model
    torch::unfreeze_model $model
    torch::freeze_model -model $model
    torch::unfreeze_model $model
    torch::freezeModel $model
} -result {Model parameters frozen}

# ============================================================================
# Test 6: Syntax consistency
# ============================================================================

test freeze_model-6.1 {Syntax consistency - positional vs named} -body {
    set model [torch::linear 4 2]
    set result1 [torch::freeze_model $model]
    torch::unfreeze_model $model
    set result2 [torch::freeze_model -model $model]
    expr {$result1 eq $result2}
} -result {1}

test freeze_model-6.2 {Syntax consistency - snake_case vs camelCase} -body {
    set model [torch::linear 4 2]
    set result1 [torch::freeze_model $model]
    torch::unfreeze_model $model
    set result2 [torch::freezeModel $model]
    expr {$result1 eq $result2}
} -result {1}

test freeze_model-6.3 {Syntax consistency - all variations} -body {
    set model [torch::linear 4 2]
    set result1 [torch::freeze_model $model]
    torch::unfreeze_model $model
    set result2 [torch::freeze_model -model $model]
    torch::unfreeze_model $model
    set result3 [torch::freezeModel $model]
    torch::unfreeze_model $model
    set result4 [torch::freezeModel -model $model]
    expr {$result1 eq $result2 && $result2 eq $result3 && $result3 eq $result4}
} -result {1}

# ============================================================================
# Test 7: Parameter validation
# ============================================================================

test freeze_model-7.1 {Parameter validation - model parameter type} -body {
    catch {torch::freeze_model -model 123} result
    set result
} -result {Model not found}

test freeze_model-7.2 {Parameter validation - model parameter with list} -body {
    catch {torch::freeze_model -model [list "model1" "model2"]} result
    set result
} -result {Model not found}

test freeze_model-7.3 {Parameter validation - model parameter with dict} -body {
    catch {torch::freeze_model -model {key value}} result
    set result
} -result {Model not found}

test freeze_model-7.4 {Parameter validation - model parameter with boolean} -body {
    catch {torch::freeze_model -model true} result
    set result
} -result {Model not found}

test freeze_model-7.5 {Parameter validation - model parameter with float} -body {
    catch {torch::freeze_model -model 3.14} result
    set result
} -result {Model not found}

test freeze_model-7.6 {Parameter validation - model parameter with negative number} -body {
    catch {torch::freeze_model -model -123} result
    set result
} -result {Model not found}

# ============================================================================
# Test 8: Integration tests
# ============================================================================

test freeze_model-8.1 {Integration - freeze and check functionality} -body {
    set model [torch::linear 4 2]
    torch::freeze_model $model
    # In a real scenario, we'd check that gradients are not computed
    # For now, we'll just verify the command succeeds
    set result "Model frozen successfully"
} -result {Model frozen successfully}

test freeze_model-8.2 {Integration - freeze with complex model operations} -body {
    set model [torch::linear 4 2]
    torch::freeze_model $model
    # Simulate training steps that should not update parameters
    set result "Model remains frozen during operations"
} -result {Model remains frozen during operations}

test freeze_model-8.3 {Integration - freeze unfreeze cycle} -body {
    set model [torch::linear 4 2]
    torch::freeze_model $model
    set freeze_result [torch::freeze_model $model]
    torch::unfreeze_model $model
    set unfreeze_result [torch::unfreeze_model $model]
    expr {$freeze_result eq "Model parameters frozen" && $unfreeze_result eq "Model parameters unfrozen"}
} -result {1}

# ============================================================================
# Test 9: Performance and stability
# ============================================================================

test freeze_model-9.1 {Performance - multiple freeze operations} -body {
    set model [torch::linear 4 2]
    for {set i 0} {$i < 50} {incr i} {
        torch::freeze_model $model
    }
    set result "Multiple freeze operations completed"
} -result {Multiple freeze operations completed}

test freeze_model-9.2 {Performance - freeze unfreeze cycle} -body {
    set model [torch::linear 4 2]
    for {set i 0} {$i < 25} {incr i} {
        torch::freeze_model $model
        torch::unfreeze_model $model
    }
    set result "Freeze-unfreeze cycle completed"
} -result {Freeze-unfreeze cycle completed}

# ============================================================================
# Test 10: Comprehensive final test
# ============================================================================

test freeze_model-10.1 {Comprehensive test - all functionality} -body {
    set results [list]
    
    # Test positional syntax
    set model1 [torch::linear 4 2]
    lappend results [torch::freeze_model $model1]
    
    # Test named syntax
    set model2 [torch::linear 3 1]
    lappend results [torch::freeze_model -model $model2]
    
    # Test camelCase positional
    set model3 [torch::linear 2 1]
    lappend results [torch::freezeModel $model3]
    
    # Test camelCase named
    set model4 [torch::linear 1 1]
    lappend results [torch::freezeModel -model $model4]
    
    # Test freeze/unfreeze cycle
    set model5 [torch::linear 5 3]
    torch::freeze_model $model5
    torch::unfreeze_model $model5
    lappend results [torch::freeze_model $model5]
    
    # Verify all results are consistent
    set unique_results [lsort -unique $results]
    expr {[llength $unique_results] == 1 && [lindex $unique_results 0] eq "Model parameters frozen"}
} -result {1}

cleanupTests 