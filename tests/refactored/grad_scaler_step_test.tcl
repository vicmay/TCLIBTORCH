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
;# TORCH::GRAD_SCALER_STEP TESTS
;# Test both positional and named parameter syntax
;# ============================================================================

;# Test torch::grad_scaler_step - Positional Syntax (Backward Compatibility)
test grad_scaler_step-1.1 {Basic positional syntax} -body {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::randn -shape {2 3} -dtype float32]
    set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]
    set result [torch::grad_scaler_step $scaler $optimizer]
    expr {$result eq "scaler step completed"}
} -result 1

test grad_scaler_step-1.2 {Positional syntax with float32 tensors} -body {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::randn -shape {2 3} -dtype float32]
    set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]
    set result [torch::grad_scaler_step $scaler $optimizer]
    expr {$result eq "scaler step completed"}
} -result 1

test grad_scaler_step-1.3 {Positional syntax with different optimizer} -body {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::randn -shape {3 3} -dtype float32]
    set optimizer [torch::optimizer_adam -parameters [list $tensor] -lr 0.001]
    set result [torch::grad_scaler_step $scaler $optimizer]
    expr {$result eq "scaler step completed"}
} -result 1

;# Test torch::grad_scaler_step - Named Parameter Syntax
test grad_scaler_step-2.1 {Named parameter syntax} -body {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::randn -shape {2 3} -dtype float32]
    set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]
    set result [torch::grad_scaler_step -scaler $scaler -optimizer $optimizer]
    expr {$result eq "scaler step completed"}
} -result 1

test grad_scaler_step-2.2 {Named parameter syntax with parameter order reversed} -body {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::randn -shape {2 3} -dtype float32]
    set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]
    set result [torch::grad_scaler_step -optimizer $optimizer -scaler $scaler]
    expr {$result eq "scaler step completed"}
} -result 1

test grad_scaler_step-2.3 {Named parameter syntax with alternative parameter names} -body {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::randn -shape {2 3} -dtype float32]
    set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]
    set result [torch::grad_scaler_step -gradScaler $scaler -optimizer $optimizer]
    expr {$result eq "scaler step completed"}
} -result 1

test grad_scaler_step-2.4 {Named parameter syntax with optim alias} -body {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::randn -shape {2 3} -dtype float32]
    set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]
    set result [torch::grad_scaler_step -scaler $scaler -optim $optimizer]
    expr {$result eq "scaler step completed"}
} -result 1

;# Test torch::gradScalerStep - CamelCase Alias
test grad_scaler_step-3.1 {CamelCase alias with positional syntax} -body {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::randn -shape {2 3} -dtype float32]
    set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]
    set result [torch::gradScalerStep $scaler $optimizer]
    expr {$result eq "scaler step completed"}
} -result 1

test grad_scaler_step-3.2 {CamelCase alias with named syntax} -body {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::randn -shape {2 3} -dtype float32]
    set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]
    set result [torch::gradScalerStep -scaler $scaler -optimizer $optimizer]
    expr {$result eq "scaler step completed"}
} -result 1

;# Error handling tests
test grad_scaler_step-4.1 {Error: Invalid scaler handle} -body {
    set tensor [torch::randn -shape {2 3} -dtype float32]
    set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]
    catch {torch::grad_scaler_step -scaler invalid_scaler -optimizer $optimizer} result
    expr {[string match "*scaler not found*" $result]}
} -result 1

test grad_scaler_step-4.2 {Error: Invalid optimizer handle} -body {
    set scaler [torch::grad_scaler_new]
    catch {torch::grad_scaler_step -scaler $scaler -optimizer invalid_optimizer} result
    expr {[string match "*Optimizer not found*" $result]}
} -result 1

test grad_scaler_step-4.3 {Error: Missing scaler parameter} -body {
    set tensor [torch::randn -shape {2 3} -dtype float32]
    set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]
    catch {torch::grad_scaler_step -optimizer $optimizer} result
    expr {[string match "*Required parameters missing*" $result]}
} -result 1

test grad_scaler_step-4.4 {Error: Missing optimizer parameter} -body {
    set scaler [torch::grad_scaler_new]
    catch {torch::grad_scaler_step -scaler $scaler} result
    expr {[string match "*Required parameters missing*" $result]}
} -result 1

test grad_scaler_step-4.5 {Error: Unknown parameter} -body {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::randn -shape {2 3} -dtype float32]
    set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]
    catch {torch::grad_scaler_step -invalidParam value -scaler $scaler -optimizer $optimizer} result
    expr {[string match "*Unknown parameter*" $result]}
} -result 1

test grad_scaler_step-4.6 {Error: Incomplete named parameter} -body {
    set scaler [torch::grad_scaler_new]
    catch {torch::grad_scaler_step -scaler} result
    expr {[string match "*must come in pairs*" $result]}
} -result 1

test grad_scaler_step-4.7 {Error: Wrong number of positional arguments - too few} -body {
    set scaler [torch::grad_scaler_new]
    catch {torch::grad_scaler_step $scaler} result
    expr {[string match "*Usage: torch::grad_scaler_step scaler optimizer*" $result]}
} -result 1

test grad_scaler_step-4.8 {Error: Wrong number of positional arguments - too many} -body {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::randn -shape {2 3} -dtype float32]
    set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]
    catch {torch::grad_scaler_step $scaler $optimizer extra} result
    expr {[string match "*Usage: torch::grad_scaler_step scaler optimizer*" $result]}
} -result 1

;# Integration tests
test grad_scaler_step-5.1 {Integration with different optimizers} -body {
    set scaler [torch::grad_scaler_new]
    set tensor1 [torch::randn -shape {2 3} -dtype float32]
    set tensor2 [torch::randn -shape {3 4} -dtype float32]
    set optimizer1 [torch::optimizer_sgd -parameters [list $tensor1] -lr 0.01]
    set optimizer2 [torch::optimizer_adam -parameters [list $tensor2] -lr 0.001]
    
    set result1 [torch::grad_scaler_step -scaler $scaler -optimizer $optimizer1]
    set result2 [torch::grad_scaler_step -scaler $scaler -optimizer $optimizer2]
    
    expr {$result1 eq "scaler step completed" && $result2 eq "scaler step completed"}
} -result 1

test grad_scaler_step-5.2 {Integration with scaled tensors} -body {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::randn -shape {2 3} -dtype float32]
    set scaled_tensor [torch::grad_scaler_scale -scaler $scaler -tensor $tensor]
    set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]
    
    set result [torch::grad_scaler_step -scaler $scaler -optimizer $optimizer]
    expr {$result eq "scaler step completed"}
} -result 1

;# Mixed syntax consistency tests
test grad_scaler_step-6.1 {Consistency between positional and named syntax} -body {
    set scaler [torch::grad_scaler_new]
    set tensor1 [torch::randn -shape {2 3} -dtype float32]
    set tensor2 [torch::randn -shape {2 3} -dtype float32]
    set optimizer1 [torch::optimizer_sgd -parameters [list $tensor1] -lr 0.01]
    set optimizer2 [torch::optimizer_sgd -parameters [list $tensor2] -lr 0.01]
    
    set result_pos [torch::grad_scaler_step $scaler $optimizer1]
    set result_named [torch::grad_scaler_step -scaler $scaler -optimizer $optimizer2]
    
    expr {$result_pos eq $result_named}
} -result 1

;# Full workflow test
test grad_scaler_step-7.1 {Complete AMP workflow} -body {
    ;# Create scaler and tensors  
    set scaler [torch::grad_scaler_new -initScale 1024.0]
    set loss_tensor [torch::randn -shape {1} -dtype float32]
    set param_tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    
    ;# Create optimizer
    set optimizer [torch::optimizer_sgd -parameters [list $param_tensor] -lr 0.01]
    
    ;# Scale loss
    set scaled_loss [torch::grad_scaler_scale -scaler $scaler -tensor $loss_tensor]
    
    ;# Step optimizer
    set result [torch::grad_scaler_step -scaler $scaler -optimizer $optimizer]
    
    expr {$result eq "scaler step completed"}
} -result 1

cleanupTests 