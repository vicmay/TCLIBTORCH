#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# ============================================================================
;# TORCH::GRAD_SCALER_UPDATE TESTS
;# Test both positional and named parameter syntax
;# ============================================================================

;# Test torch::grad_scaler_update - Positional Syntax (Backward Compatibility)
test grad_scaler_update-1.1 {Basic positional syntax} -body {
    set scaler [torch::grad_scaler_new]
    set result [torch::grad_scaler_update $scaler]
    expr {$result eq "scaler updated"}
} -result 1

test grad_scaler_update-1.2 {Positional syntax with custom init scale} -body {
    set scaler [torch::grad_scaler_new -initScale 2048.0]
    set result [torch::grad_scaler_update $scaler]
    expr {$result eq "scaler updated"}
} -result 1

test grad_scaler_update-1.3 {Positional syntax with custom growth factors} -body {
    set scaler [torch::grad_scaler_new -growthFactor 3.0 -backoffFactor 0.25]
    set result [torch::grad_scaler_update $scaler]
    expr {$result eq "scaler updated"}
} -result 1

test grad_scaler_update-1.4 {Positional syntax with custom growth interval} -body {
    set scaler [torch::grad_scaler_new -growthInterval 1000]
    set result [torch::grad_scaler_update $scaler]
    expr {$result eq "scaler updated"}
} -result 1

;# Test torch::grad_scaler_update - Named Parameter Syntax
test grad_scaler_update-2.1 {Named parameter syntax} -body {
    set scaler [torch::grad_scaler_new]
    set result [torch::grad_scaler_update -scaler $scaler]
    expr {$result eq "scaler updated"}
} -result 1

test grad_scaler_update-2.2 {Named parameter syntax with custom scaler} -body {
    set scaler [torch::grad_scaler_new -initScale 1024.0]
    set result [torch::grad_scaler_update -scaler $scaler]
    expr {$result eq "scaler updated"}
} -result 1

test grad_scaler_update-2.3 {Named parameter syntax with alternative parameter name} -body {
    set scaler [torch::grad_scaler_new]
    set result [torch::grad_scaler_update -gradScaler $scaler]
    expr {$result eq "scaler updated"}
} -result 1

;# Test torch::gradScalerUpdate - CamelCase Alias
test grad_scaler_update-3.1 {CamelCase alias with positional syntax} -body {
    set scaler [torch::grad_scaler_new]
    set result [torch::gradScalerUpdate $scaler]
    expr {$result eq "scaler updated"}
} -result 1

test grad_scaler_update-3.2 {CamelCase alias with named syntax} -body {
    set scaler [torch::grad_scaler_new]
    set result [torch::gradScalerUpdate -scaler $scaler]
    expr {$result eq "scaler updated"}
} -result 1

test grad_scaler_update-3.3 {CamelCase alias with custom scaler} -body {
    set scaler [torch::grad_scaler_new -initScale 512.0]
    set result [torch::gradScalerUpdate -scaler $scaler]
    expr {$result eq "scaler updated"}
} -result 1

;# Error handling tests
test grad_scaler_update-4.1 {Error: Invalid scaler handle} -body {
    catch {torch::grad_scaler_update -scaler invalid_scaler} result
    expr {[string match "*scaler not found*" $result]}
} -result 1

test grad_scaler_update-4.2 {Error: Missing scaler parameter} -body {
    catch {torch::grad_scaler_update -wrongparam value} result
    expr {[string match "*Unknown parameter*" $result]}
} -result 1

test grad_scaler_update-4.3 {Error: Unknown parameter} -body {
    set scaler [torch::grad_scaler_new]
    catch {torch::grad_scaler_update -invalidParam value -scaler $scaler} result
    expr {[string match "*Unknown parameter*" $result]}
} -result 1

test grad_scaler_update-4.4 {Error: Incomplete named parameter} -body {
    catch {torch::grad_scaler_update -scaler} result
    expr {[string match "*must come in pairs*" $result]}
} -result 1

test grad_scaler_update-4.5 {Error: Wrong number of positional arguments - too few} -body {
    catch {torch::grad_scaler_update} result
    expr {[string match "*Usage: torch::grad_scaler_update scaler*" $result]}
} -result 1

test grad_scaler_update-4.6 {Error: Wrong number of positional arguments - too many} -body {
    set scaler [torch::grad_scaler_new]
    catch {torch::grad_scaler_update $scaler extra} result
    expr {[string match "*Usage: torch::grad_scaler_update scaler*" $result]}
} -result 1

;# Integration tests
test grad_scaler_update-5.1 {Integration with multiple scalers} -body {
    set scaler1 [torch::grad_scaler_new]
    set scaler2 [torch::grad_scaler_new -initScale 2048.0]
    set scaler3 [torch::grad_scaler_new -growthFactor 4.0]
    
    set result1 [torch::grad_scaler_update -scaler $scaler1]
    set result2 [torch::grad_scaler_update -scaler $scaler2]
    set result3 [torch::grad_scaler_update -scaler $scaler3]
    
    expr {$result1 eq "scaler updated" && $result2 eq "scaler updated" && $result3 eq "scaler updated"}
} -result 1

test grad_scaler_update-5.2 {Integration with step operation} -body {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::randn -shape {2 3} -dtype float32]
    set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]
    
    ;# Step first
    set step_result [torch::grad_scaler_step -scaler $scaler -optimizer $optimizer]
    
    ;# Then update
    set update_result [torch::grad_scaler_update -scaler $scaler]
    
    expr {$step_result eq "scaler step completed" && $update_result eq "scaler updated"}
} -result 1

;# Mixed syntax consistency tests
test grad_scaler_update-6.1 {Consistency between positional and named syntax} -body {
    set scaler1 [torch::grad_scaler_new]
    set scaler2 [torch::grad_scaler_new]
    
    set result_pos [torch::grad_scaler_update $scaler1]
    set result_named [torch::grad_scaler_update -scaler $scaler2]
    
    expr {$result_pos eq $result_named}
} -result 1

;# Performance and edge case tests
test grad_scaler_update-7.1 {Performance with multiple updates} -body {
    set scaler [torch::grad_scaler_new]
    set results [list]
    
    for {set i 0} {$i < 10} {incr i} {
        lappend results [torch::grad_scaler_update -scaler $scaler]
    }
    
    set all_success 1
    foreach result $results {
        if {$result ne "scaler updated"} {
            set all_success 0
            break
        }
    }
    
    expr {$all_success}
} -result 1

test grad_scaler_update-7.2 {Edge case: Update after getting scale} -body {
    set scaler [torch::grad_scaler_new]
    set scale [torch::grad_scaler_get_scale -scaler $scaler]
    set result [torch::grad_scaler_update -scaler $scaler]
    expr {$result eq "scaler updated"}
} -result 1

;# Advanced integration tests
test grad_scaler_update-8.1 {Integration with scale and step operations} -body {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::randn -shape {2 3} -dtype float32]
    set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]
    
    ;# Scale tensor
    set scaled_tensor [torch::grad_scaler_scale -scaler $scaler -tensor $tensor]
    
    ;# Step optimizer
    set step_result [torch::grad_scaler_step -scaler $scaler -optimizer $optimizer]
    
    ;# Update scaler
    set update_result [torch::grad_scaler_update -scaler $scaler]
    
    expr {$step_result eq "scaler step completed" && $update_result eq "scaler updated"}
} -result 1

test grad_scaler_update-8.2 {Integration with different scaler configurations} -body {
    set scaler_default [torch::grad_scaler_new]
    set scaler_custom [torch::grad_scaler_new -initScale 4096.0 -growthFactor 1.5 -backoffFactor 0.75]
    
    set result1 [torch::grad_scaler_update -scaler $scaler_default]
    set result2 [torch::grad_scaler_update -scaler $scaler_custom]
    
    expr {$result1 eq "scaler updated" && $result2 eq "scaler updated"}
} -result 1

;# Full workflow test
test grad_scaler_update-9.1 {Complete AMP workflow with update} -body {
    ;# Create scaler and tensors
    set scaler [torch::grad_scaler_new -initScale 1024.0]
    set loss_tensor [torch::randn -shape {1} -dtype float32]
    set param_tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    
    ;# Create optimizer
    set optimizer [torch::optimizer_sgd -parameters [list $param_tensor] -lr 0.01]
    
    ;# Scale loss
    set scaled_loss [torch::grad_scaler_scale -scaler $scaler -tensor $loss_tensor]
    
    ;# Get initial scale
    set initial_scale [torch::grad_scaler_get_scale -scaler $scaler]
    
    ;# Step optimizer
    set step_result [torch::grad_scaler_step -scaler $scaler -optimizer $optimizer]
    
    ;# Update scaler
    set update_result [torch::grad_scaler_update -scaler $scaler]
    
    ;# Get final scale
    set final_scale [torch::grad_scaler_get_scale -scaler $scaler]
    
    expr {$step_result eq "scaler step completed" && $update_result eq "scaler updated"}
} -result 1

test grad_scaler_update-9.2 {Multiple training steps with updates} -body {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float32]
    set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]
    
    set all_success 1
    
    for {set step 0} {$step < 5} {incr step} {
        ;# Scale a dummy loss
        set loss [torch::randn -shape {1} -dtype float32]
        set scaled_loss [torch::grad_scaler_scale -scaler $scaler -tensor $loss]
        
        ;# Step optimizer
        set step_result [torch::grad_scaler_step -scaler $scaler -optimizer $optimizer]
        
        ;# Update scaler
        set update_result [torch::grad_scaler_update -scaler $scaler]
        
        if {$step_result ne "scaler step completed" || $update_result ne "scaler updated"} {
            set all_success 0
            break
        }
    }
    
    expr {$all_success}
} -result 1

;# CamelCase alias comprehensive test
test grad_scaler_update-10.1 {CamelCase alias functionality equivalence} -body {
    set scaler1 [torch::grad_scaler_new]
    set scaler2 [torch::grad_scaler_new]
    
    ;# Use snake_case version
    set result1 [torch::grad_scaler_update $scaler1]
    
    ;# Use camelCase version
    set result2 [torch::gradScalerUpdate $scaler2]
    
    expr {$result1 eq $result2 && $result1 eq "scaler updated"}
} -result 1

test grad_scaler_update-10.2 {CamelCase alias with named parameters} -body {
    set scaler1 [torch::grad_scaler_new]
    set scaler2 [torch::grad_scaler_new]
    
    ;# Use snake_case version with named params
    set result1 [torch::grad_scaler_update -scaler $scaler1]
    
    ;# Use camelCase version with named params
    set result2 [torch::gradScalerUpdate -scaler $scaler2]
    
    expr {$result1 eq $result2 && $result1 eq "scaler updated"}
} -result 1

cleanupTests 