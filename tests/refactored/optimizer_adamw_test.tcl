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

# Helper procedure to create simple parameter tensors
proc createParams {} {
    set param1 [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32 -device cpu]
    set param2 [torch::tensor_create -data {7.0 8.0 9.0} -shape {3 1} -dtype float32 -device cpu]
    return [list $param1 $param2]
}

# Test 1: Original positional syntax
test optimizer_adamw-1.1 {Original positional syntax - basic} {
    set params [createParams]
    set optimizer [torch::optimizer_adamw $params 0.001]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adamw-1.2 {Original positional syntax - with weight decay} {
    set params [createParams]
    set optimizer [torch::optimizer_adamw $params 0.001 0.01]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

# Test 2: New named parameter syntax
test optimizer_adamw-2.1 {Named parameter syntax - basic} {
    set params [createParams]
    set optimizer [torch::optimizer_adamw -parameters $params -lr 0.001]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adamw-2.2 {Named parameter syntax - with beta1} {
    set params [createParams]
    set optimizer [torch::optimizer_adamw -parameters $params -lr 0.001 -beta1 0.8]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adamw-2.3 {Named parameter syntax - with beta2} {
    set params [createParams]
    set optimizer [torch::optimizer_adamw -parameters $params -lr 0.001 -beta2 0.99]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adamw-2.4 {Named parameter syntax - with eps} {
    set params [createParams]
    set optimizer [torch::optimizer_adamw -parameters $params -lr 0.001 -eps 1e-7]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adamw-2.5 {Named parameter syntax - with weightDecay} {
    set params [createParams]
    set optimizer [torch::optimizer_adamw -parameters $params -lr 0.001 -weightDecay 0.02]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adamw-2.6 {Named parameter syntax - with amsgrad true} {
    set params [createParams]
    set optimizer [torch::optimizer_adamw -parameters $params -lr 0.001 -amsgrad true]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adamw-2.7 {Named parameter syntax - with amsgrad false} {
    set params [createParams]
    set optimizer [torch::optimizer_adamw -parameters $params -lr 0.001 -amsgrad false]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adamw-2.8 {Named parameter syntax - all parameters} {
    set params [createParams]
    set optimizer [torch::optimizer_adamw -parameters $params -lr 0.001 -beta1 0.8 -beta2 0.99 -eps 1e-7 -weightDecay 0.02 -amsgrad true]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adamw-2.9 {Named parameter syntax - alternative parameter names} {
    set params [createParams]
    set optimizer [torch::optimizer_adamw -params $params -learningRate 0.002 -epsilon 1e-6 -weight_decay 0.005]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adamw-2.10 {Named parameter syntax - out of order parameters} {
    set params [createParams]
    set optimizer [torch::optimizer_adamw -beta1 0.85 -parameters $params -weightDecay 0.01 -lr 0.001 -eps 1e-7 -beta2 0.99 -amsgrad false]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

# Test 3: CamelCase alias
test optimizer_adamw-3.1 {CamelCase alias - positional syntax} {
    set params [createParams]
    set optimizer [torch::optimizerAdamW $params 0.001]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adamw-3.2 {CamelCase alias - named syntax} {
    set params [createParams]
    set optimizer [torch::optimizerAdamW -parameters $params -lr 0.001 -beta1 0.9 -beta2 0.999 -weightDecay 0.01]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

# Test 4: Error handling - positional syntax
test optimizer_adamw-4.1 {Error handling - missing arguments positional} {
    set params [createParams]
    catch {torch::optimizer_adamw $params} result
    expr {[string match "*Usage: torch::optimizer_adamw*" $result]}
} {1}

test optimizer_adamw-4.2 {Error handling - invalid learning rate positional} {
    set params [createParams]
    catch {torch::optimizer_adamw $params -0.001} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test optimizer_adamw-4.3 {Error handling - invalid weight_decay positional} {
    set params [createParams]
    catch {torch::optimizer_adamw $params 0.001 -0.01} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

# Test 5: Error handling - named syntax
test optimizer_adamw-5.1 {Error handling - missing required parameters named} {
    catch {torch::optimizer_adamw -lr 0.001} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test optimizer_adamw-5.2 {Error handling - invalid parameter name} {
    set params [createParams]
    catch {torch::optimizer_adamw -parameters $params -lr 0.001 -invalid_param value} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test optimizer_adamw-5.3 {Error handling - parameters must come in pairs} {
    set params [createParams]
    catch {torch::optimizer_adamw -parameters $params -lr} result
    expr {[string match "*Named parameters must come in pairs*" $result]}
} {1}

test optimizer_adamw-5.4 {Error handling - invalid learning rate named} {
    set params [createParams]
    catch {torch::optimizer_adamw -parameters $params -lr -0.001} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test optimizer_adamw-5.5 {Error handling - invalid beta1 named} {
    set params [createParams]
    catch {torch::optimizer_adamw -parameters $params -lr 0.001 -beta1 1.5} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test optimizer_adamw-5.6 {Error handling - invalid beta2 named} {
    set params [createParams]
    catch {torch::optimizer_adamw -parameters $params -lr 0.001 -beta2 -0.1} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test optimizer_adamw-5.7 {Error handling - invalid eps named} {
    set params [createParams]
    catch {torch::optimizer_adamw -parameters $params -lr 0.001 -eps -1e-8} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test optimizer_adamw-5.8 {Error handling - invalid weight decay named} {
    set params [createParams]
    catch {torch::optimizer_adamw -parameters $params -lr 0.001 -weightDecay -0.01} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test optimizer_adamw-5.9 {Error handling - invalid amsgrad value} {
    set params [createParams]
    catch {torch::optimizer_adamw -parameters $params -lr 0.001 -amsgrad invalid} result
    expr {[string match "*Invalid amsgrad value*" $result]}
} {1}

# Test 6: Parameter validation
test optimizer_adamw-6.1 {Parameter validation - empty parameter list} {
    catch {torch::optimizer_adamw {} 0.001} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test optimizer_adamw-6.2 {Parameter validation - invalid tensor reference} {
    catch {torch::optimizer_adamw {invalid_tensor} 0.001} result
    expr {[string match "*Invalid parameter tensor*" $result]}
} {1}

# Test 7: Default values
test optimizer_adamw-7.1 {Default values - positional} {
    set params [createParams]
    set optimizer1 [torch::optimizer_adamw $params 0.001]
    set optimizer2 [torch::optimizer_adamw $params 0.001 0.01]
    # Both should succeed (can't easily test internal values)
    expr {[string length $optimizer1] > 0 && [string length $optimizer2] > 0}
} {1}

test optimizer_adamw-7.2 {Default values - named} {
    set params [createParams]
    set optimizer1 [torch::optimizer_adamw -parameters $params -lr 0.001]
    set optimizer2 [torch::optimizer_adamw -parameters $params -lr 0.001 -beta1 0.9 -beta2 0.999 -eps 1e-8 -weightDecay 0.01 -amsgrad false]
    # Both should succeed (can't easily test internal values)
    expr {[string length $optimizer1] > 0 && [string length $optimizer2] > 0}
} {1}

# Test 8: Consistency between syntaxes
test optimizer_adamw-8.1 {Consistency - both syntaxes produce optimizers} {
    set params [createParams]
    set optimizer1 [torch::optimizer_adamw $params 0.001 0.01]
    set optimizer2 [torch::optimizer_adamw -parameters $params -lr 0.001 -weightDecay 0.01]
    # Both should be valid optimizer handles
    expr {[string match "optimizer*" $optimizer1] && [string match "optimizer*" $optimizer2]}
} {1}

# Test 9: Integration with other commands
test optimizer_adamw-9.1 {Integration - can be used with optimizer operations} {
    set params [createParams]
    set optimizer [torch::optimizer_adamw -parameters $params -lr 0.001]
    # The optimizer should be a valid handle for further operations
    # This just tests that we get a valid handle format
    expr {[string match "optimizer*" $optimizer]}
} {1}

# Test 10: Edge cases
test optimizer_adamw-10.1 {Edge case - minimum valid learning rate} {
    set params [createParams]
    set optimizer [torch::optimizer_adamw -parameters $params -lr 1e-10]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adamw-10.2 {Edge case - beta values at boundaries} {
    set params [createParams]
    set optimizer [torch::optimizer_adamw -parameters $params -lr 0.001 -beta1 0.0 -beta2 0.0]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adamw-10.3 {Edge case - zero weight decay} {
    set params [createParams]
    set optimizer [torch::optimizer_adamw -parameters $params -lr 0.001 -weightDecay 0.0]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adamw-10.4 {Edge case - large parameter lists} {
    # Create tensors with appropriate shapes
    set param1 [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32 -device cpu]
    set param2 [torch::tensor_create -data {7.0 8.0 9.0} -shape {3 1} -dtype float32 -device cpu] 
    set param3 [torch::tensor_create -data {10.0 11.0} -shape {1 2} -dtype float32 -device cpu]
    set large_params [list $param1 $param2 $param3]
    
    set optimizer [torch::optimizer_adamw -parameters $large_params -lr 0.001 -weightDecay 0.01]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adamw-10.5 {Edge case - very small eps value} {
    set params [createParams]
    set optimizer [torch::optimizer_adamw -parameters $params -lr 0.001 -eps 1e-15]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adamw-10.6 {Edge case - high weight decay} {
    set params [createParams]
    set optimizer [torch::optimizer_adamw -parameters $params -lr 0.001 -weightDecay 0.1]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

# Test 11: AdamW specific features
test optimizer_adamw-11.1 {AdamW specific - amsgrad boolean values} {
    set params [createParams]
    set optimizer1 [torch::optimizer_adamw -parameters $params -lr 0.001 -amsgrad 0]
    set optimizer2 [torch::optimizer_adamw -parameters $params -lr 0.001 -amsgrad 1]
    expr {[string match "optimizer*" $optimizer1] && [string match "optimizer*" $optimizer2]}
} {1}

test optimizer_adamw-11.2 {AdamW specific - higher default weight decay} {
    set params [createParams]
    # AdamW should work well with higher weight decay values
    set optimizer [torch::optimizer_adamw -parameters $params -lr 0.001 -weightDecay 0.05]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

cleanupTests 