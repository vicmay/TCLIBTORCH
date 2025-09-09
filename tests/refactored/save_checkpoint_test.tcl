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

# Create test model and optimizer
set model [torch::linear 10 5]
set optimizer [torch::optimizer_adam [torch::layer_parameters $model] 0.01]

# Test cases for positional syntax
test save_checkpoint-1.1 {Basic positional syntax - minimal args} {
    set filename "test_checkpoint_1.pt"
    set result [torch::save_checkpoint $model $optimizer $filename]
    file exists $filename
} {1}

test save_checkpoint-1.2 {Basic positional syntax - with epoch} {
    set filename "test_checkpoint_2.pt"
    set result [torch::save_checkpoint $model $optimizer $filename 5]
    file exists $filename
} {1}

test save_checkpoint-1.3 {Basic positional syntax - with epoch and loss} {
    set filename "test_checkpoint_3.pt"
    set result [torch::save_checkpoint $model $optimizer $filename 5 0.123]
    file exists $filename
} {1}

test save_checkpoint-1.4 {Basic positional syntax - all args} {
    set filename "test_checkpoint_4.pt"
    set result [torch::save_checkpoint $model $optimizer $filename 5 0.123 0.001]
    file exists $filename
} {1}

# Test cases for named parameter syntax
test save_checkpoint-2.1 {Named parameter syntax - minimal args} {
    set filename "test_checkpoint_5.pt"
    set result [torch::save_checkpoint -model $model -optimizer $optimizer -filename $filename]
    file exists $filename
} {1}

test save_checkpoint-2.2 {Named parameter syntax - with epoch} {
    set filename "test_checkpoint_6.pt"
    set result [torch::save_checkpoint -model $model -optimizer $optimizer -filename $filename -epoch 5]
    file exists $filename
} {1}

test save_checkpoint-2.3 {Named parameter syntax - with epoch and loss} {
    set filename "test_checkpoint_7.pt"
    set result [torch::save_checkpoint -model $model -optimizer $optimizer -filename $filename -epoch 5 -loss 0.123]
    file exists $filename
} {1}

test save_checkpoint-2.4 {Named parameter syntax - all args} {
    set filename "test_checkpoint_8.pt"
    set result [torch::save_checkpoint -model $model -optimizer $optimizer -filename $filename -epoch 5 -loss 0.123 -lr 0.001]
    file exists $filename
} {1}

# Test cases for camelCase alias
test save_checkpoint-3.1 {CamelCase alias - all args} {
    set filename "test_checkpoint_9.pt"
    set result [torch::saveCheckpoint -model $model -optimizer $optimizer -filename $filename -epoch 5 -loss 0.123 -lr 0.001]
    file exists $filename
} {1}

# Error handling tests
test save_checkpoint-4.1 {Error - no arguments} {
    catch {torch::save_checkpoint} err
    set err
} {wrong # args: should be "torch::save_checkpoint model optimizer filename ?epoch? ?loss? ?lr?"}

test save_checkpoint-4.2 {Error - invalid model} {
    catch {torch::save_checkpoint invalid_model $optimizer "test.pt"} err
    set err
} {Model not found}

test save_checkpoint-4.3 {Error - invalid optimizer} {
    catch {torch::save_checkpoint $model invalid_optimizer "test.pt"} err
    set err
} {Optimizer not found}

test save_checkpoint-4.4 {Error - invalid epoch} {
    catch {torch::save_checkpoint -model $model -optimizer $optimizer -filename "test.pt" -epoch invalid} err
    set err
} {Error in save_checkpoint: Invalid epoch value}

test save_checkpoint-4.5 {Error - unknown parameter} {
    catch {torch::save_checkpoint -invalid $model} err
    set err
} {Error in save_checkpoint: Unknown parameter: -invalid}

# Cleanup test files
foreach file [glob -nocomplain test_checkpoint_*.pt] {
    file delete $file
}

cleanupTests 