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
test optimizer_adagrad-1.1 {Original positional syntax - basic} {
    set params [createParams]
    set optimizer [torch::optimizer_adagrad $params 0.01]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adagrad-1.2 {Original positional syntax - with eps} {
    set params [createParams]
    set optimizer [torch::optimizer_adagrad $params 0.01 1e-8]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

# Test 2: New named parameter syntax
test optimizer_adagrad-2.1 {Named parameter syntax - basic} {
    set params [createParams]
    set optimizer [torch::optimizer_adagrad -parameters $params -lr 0.01]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adagrad-2.2 {Named parameter syntax - with eps} {
    set params [createParams]
    set optimizer [torch::optimizer_adagrad -parameters $params -lr 0.01 -eps 1e-8]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adagrad-2.3 {Named parameter syntax - alternative parameter names} {
    set params [createParams]
    set optimizer [torch::optimizer_adagrad -params $params -learningRate 0.02 -epsilon 1e-9]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

# Test 3: CamelCase alias
test optimizer_adagrad-3.1 {CamelCase alias - positional syntax} {
    set params [createParams]
    set optimizer [torch::optimizerAdagrad $params 0.01]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

test optimizer_adagrad-3.2 {CamelCase alias - named syntax} {
    set params [createParams]
    set optimizer [torch::optimizerAdagrad -parameters $params -lr 0.01 -eps 1e-8]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

# Test 4: Error handling - positional syntax
test optimizer_adagrad-4.1 {Error handling - missing arguments positional} {
    set params [createParams]
    catch {torch::optimizer_adagrad $params} result
    expr {[string match "*Usage: torch::optimizer_adagrad*" $result]}
} {1}

test optimizer_adagrad-4.2 {Error handling - invalid learning rate positional} {
    set params [createParams]
    catch {torch::optimizer_adagrad $params -0.01} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test optimizer_adagrad-4.3 {Error handling - invalid eps positional} {
    set params [createParams]
    catch {torch::optimizer_adagrad $params 0.01 -1e-8} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

# Test 5: Error handling - named syntax
test optimizer_adagrad-5.1 {Error handling - missing required parameters named} {
    catch {torch::optimizer_adagrad -lr 0.01} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test optimizer_adagrad-5.2 {Error handling - invalid parameter name} {
    set params [createParams]
    catch {torch::optimizer_adagrad -parameters $params -lr 0.01 -invalid_param value} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test optimizer_adagrad-5.3 {Error handling - parameters must come in pairs} {
    set params [createParams]
    catch {torch::optimizer_adagrad -parameters $params -lr} result
    expr {[string match "*Named parameters must come in pairs*" $result]}
} {1}

test optimizer_adagrad-5.4 {Error handling - invalid learning rate named} {
    set params [createParams]
    catch {torch::optimizer_adagrad -parameters $params -lr -0.01} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test optimizer_adagrad-5.5 {Error handling - invalid eps named} {
    set params [createParams]
    catch {torch::optimizer_adagrad -parameters $params -lr 0.01 -eps -1e-8} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

# Test 6: Parameter validation
test optimizer_adagrad-6.1 {Parameter validation - empty parameter list} {
    catch {torch::optimizer_adagrad {} 0.01} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test optimizer_adagrad-6.2 {Parameter validation - invalid tensor reference} {
    catch {torch::optimizer_adagrad {invalid_tensor} 0.01} result
    expr {[string match "*Invalid parameter tensor*" $result]}
} {1}

# Test 7: Default values
test optimizer_adagrad-7.1 {Default eps value - positional} {
    set params [createParams]
    set optimizer1 [torch::optimizer_adagrad $params 0.01]
    set optimizer2 [torch::optimizer_adagrad $params 0.01 1e-10]
    # Both should succeed (can't easily test internal eps value)
    expr {[string length $optimizer1] > 0 && [string length $optimizer2] > 0}
} {1}

test optimizer_adagrad-7.2 {Default eps value - named} {
    set params [createParams]
    set optimizer1 [torch::optimizer_adagrad -parameters $params -lr 0.01]
    set optimizer2 [torch::optimizer_adagrad -parameters $params -lr 0.01 -eps 1e-10]
    # Both should succeed (can't easily test internal eps value)
    expr {[string length $optimizer1] > 0 && [string length $optimizer2] > 0}
} {1}

# Test 8: Consistency between syntaxes
test optimizer_adagrad-8.1 {Consistency - both syntaxes produce optimizers} {
    set params [createParams]
    set optimizer1 [torch::optimizer_adagrad $params 0.01 1e-8]
    set optimizer2 [torch::optimizer_adagrad -parameters $params -lr 0.01 -eps 1e-8]
    # Both should be valid optimizer handles
    expr {[string match "optimizer*" $optimizer1] && [string match "optimizer*" $optimizer2]}
} {1}

# Test 9: Integration with other commands
test optimizer_adagrad-9.1 {Integration - can be used with optimizer operations} {
    set params [createParams]
    set optimizer [torch::optimizer_adagrad -parameters $params -lr 0.01]
    # The optimizer should be a valid handle for further operations
    # This just tests that we get a valid handle format
    expr {[string match "optimizer*" $optimizer]}
} {1}

# Test 10: Large parameter lists
test optimizer_adagrad-10.1 {Large parameter list handling} {
    # Create tensors with appropriate sizes for large parameter list test
    # Using zeros to create tensors of the right size without needing explicit data
    set param1 [torch::zeros {100 200} float32 cpu]
    set param2 [torch::zeros {200 50} float32 cpu]
    set param3 [torch::zeros {50 10} float32 cpu]
    set large_params [list $param1 $param2 $param3]
    
    set optimizer [torch::optimizer_adagrad -parameters $large_params -lr 0.001]
    expr {[string length $optimizer] > 0 && [string match "optimizer*" $optimizer]}
} {1}

cleanupTests 