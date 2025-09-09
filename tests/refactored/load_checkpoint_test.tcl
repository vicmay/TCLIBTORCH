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

;# Create temporary test files directory
set test_dir [file join [pwd] test_checkpoints]
if {![file exists $test_dir]} {
    file mkdir $test_dir
}

;# Helper procedure to create test parameters
proc create_test_parameters {} {
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    return [list $tensor1 $tensor2]
}

;# Helper procedure to create a test model
proc create_test_model {name} {
    set model [torch::linear -inFeatures 10 -outFeatures 5]
    return $model
}

;# Helper procedure to create a test optimizer
proc create_test_optimizer {name model} {
    set params [create_test_parameters]
    set optimizer [torch::optimizer_sgd $params 0.01]
    return $optimizer
}

;# Helper procedure to create a test checkpoint file
proc create_test_checkpoint {filename model optimizer} {
    ;# Create a test checkpoint using save_checkpoint
    torch::save_checkpoint -filename $filename -model $model -optimizer $optimizer -epoch 10 -loss 0.5 -lr 0.01
}

;# Test cases for positional syntax
test load_checkpoint-1.1 {Basic positional syntax - checkpoint loading} -setup {
    set model [create_test_model "test_model_1"]
    set optimizer [create_test_optimizer "test_opt_1" $model]
    set checkpoint_file [file join $test_dir "test1.pt"]
    create_test_checkpoint $checkpoint_file $model $optimizer
} -body {
    ;# Test positional syntax
    set result [torch::load_checkpoint $checkpoint_file $model $optimizer]
    
    ;# Should return success message containing checkpoint info
    if {[string match "*Checkpoint loaded*" $result]} {
        return "success"
    } else {
        return "failed: unexpected result format: $result"
    }
} -result "success"

test load_checkpoint-1.2 {Positional syntax - verify checkpoint metadata} -setup {
    set model [create_test_model "test_model_2"]
    set optimizer [create_test_optimizer "test_opt_2" $model]
    set checkpoint_file [file join $test_dir "test2.pt"]
    create_test_checkpoint $checkpoint_file $model $optimizer
} -body {
    ;# Test positional syntax with metadata verification
    set result [torch::load_checkpoint $checkpoint_file $model $optimizer]
    
    ;# Should contain epoch, loss, and learning rate info
    if {[string match "*epoch=10*" $result] && [string match "*loss=0.5*" $result]} {
        return "success"
    } else {
        return "failed: metadata not found in result: $result"
    }
} -result "success"

;# Test cases for named parameter syntax
test load_checkpoint-2.1 {Named parameter syntax - filename/model/optimizer} -setup {
    set model [create_test_model "test_model_3"]
    set optimizer [create_test_optimizer "test_opt_3" $model]
    set checkpoint_file [file join $test_dir "test3.pt"]
    create_test_checkpoint $checkpoint_file $model $optimizer
} -body {
    ;# Test named parameter syntax
    set result [torch::load_checkpoint -filename $checkpoint_file -model $model -optimizer $optimizer]
    
    ;# Should return success message
    if {[string match "*Checkpoint loaded*" $result]} {
        return "success"
    } else {
        return "failed: unexpected result: $result"
    }
} -result "success"

test load_checkpoint-2.2 {Named parameter syntax - file alias} -setup {
    set model [create_test_model "test_model_4"]
    set optimizer [create_test_optimizer "test_opt_4" $model]
    set checkpoint_file [file join $test_dir "test4.pt"]
    create_test_checkpoint $checkpoint_file $model $optimizer
} -body {
    ;# Test named parameter syntax with -file alias
    set result [torch::load_checkpoint -file $checkpoint_file -model $model -optimizer $optimizer]
    
    ;# Should return success message
    if {[string match "*Checkpoint loaded*" $result]} {
        return "success"
    } else {
        return "failed: unexpected result: $result"
    }
} -result "success"

;# Test cases for camelCase alias
test load_checkpoint-3.1 {camelCase alias - positional syntax} -setup {
    set model [create_test_model "test_model_5"]
    set optimizer [create_test_optimizer "test_opt_5" $model]
    set checkpoint_file [file join $test_dir "test5.pt"]
    create_test_checkpoint $checkpoint_file $model $optimizer
} -body {
    ;# Test camelCase alias with positional syntax
    set result [torch::loadCheckpoint $checkpoint_file $model $optimizer]
    
    ;# Should return success message
    if {[string match "*Checkpoint loaded*" $result]} {
        return "success"
    } else {
        return "failed: unexpected result: $result"
    }
} -result "success"

test load_checkpoint-3.2 {camelCase alias - named parameter syntax} -setup {
    set model [create_test_model "test_model_6"]
    set optimizer [create_test_optimizer "test_opt_6" $model]
    set checkpoint_file [file join $test_dir "test6.pt"]
    create_test_checkpoint $checkpoint_file $model $optimizer
} -body {
    ;# Test camelCase alias with named parameter syntax
    set result [torch::loadCheckpoint -filename $checkpoint_file -model $model -optimizer $optimizer]
    
    ;# Should return success message
    if {[string match "*Checkpoint loaded*" $result]} {
        return "success"
    } else {
        return "failed: unexpected result: $result"
    }
} -result "success"

;# Test cases for parameter validation
test load_checkpoint-4.1 {Parameter validation - both syntaxes produce same result} -setup {
    set model1 [create_test_model "test_model_7a"]
    set optimizer1 [create_test_optimizer "test_opt_7a" $model1]
    set model2 [create_test_model "test_model_7b"]
    set optimizer2 [create_test_optimizer "test_opt_7b" $model2]
    set checkpoint_file [file join $test_dir "test7.pt"]
    create_test_checkpoint $checkpoint_file $model1 $optimizer1
    
    ;# Reset models for loading
    set model1_fresh [create_test_model "test_model_7a"]
    set optimizer1_fresh [create_test_optimizer "test_opt_7a" $model1_fresh]
    set model2_fresh [create_test_model "test_model_7b"]
    set optimizer2_fresh [create_test_optimizer "test_opt_7b" $model2_fresh]
} -body {
    ;# Test both syntaxes
    set result1 [torch::load_checkpoint $checkpoint_file $model1_fresh $optimizer1_fresh]
    set result2 [torch::load_checkpoint -filename $checkpoint_file -model $model2_fresh -optimizer $optimizer2_fresh]
    
    ;# Both should return success messages
    if {[string match "*Checkpoint loaded*" $result1] && [string match "*Checkpoint loaded*" $result2]} {
        return "success"
    } else {
        return "failed: inconsistent results between syntaxes"
    }
} -result "success"

;# Error handling tests
test load_checkpoint-5.1 {Error handling - invalid model name} -body {
    set params [create_test_parameters]
    set optimizer [torch::optimizer_sgd $params 0.01]
    set checkpoint_file [file join $test_dir "test8.pt"]
    
    ;# Test with non-existent model
    if {[catch {torch::load_checkpoint $checkpoint_file "invalid_model" $optimizer} error]} {
        return "success"
    } else {
        return "failed: should have thrown error"
    }
} -result "success"

test load_checkpoint-5.2 {Error handling - invalid optimizer name} -setup {
    set model [create_test_model "test_model_9"]
    set checkpoint_file [file join $test_dir "test9.pt"]
} -body {
    ;# Test with non-existent optimizer
    if {[catch {torch::load_checkpoint $checkpoint_file $model "invalid_optimizer"} error]} {
        return "success"
    } else {
        return "failed: should have thrown error"
    }
} -result "success"

test load_checkpoint-5.3 {Error handling - missing required parameters} -body {
    ;# Test named syntax without required parameters
    if {[catch {torch::load_checkpoint -filename "test.pt"} error]} {
        return "success"
    } else {
        return "failed: should have thrown error"
    }
} -result "success"

test load_checkpoint-5.4 {Error handling - missing one parameter} -setup {
    set model [create_test_model "test_model_10"]
} -body {
    ;# Test positional syntax with missing parameter
    if {[catch {torch::load_checkpoint "test.pt" $model} error]} {
        return "success"
    } else {
        return "failed: should have thrown error"
    }
} -result "success"

test load_checkpoint-5.5 {Error handling - unknown parameter} -setup {
    set model [create_test_model "test_model_11"]
    set optimizer [create_test_optimizer "test_opt_11" $model]
} -body {
    ;# Test named syntax with unknown parameter
    if {[catch {torch::load_checkpoint -unknown_param "test.pt" -model $model -optimizer $optimizer} error]} {
        return "success"
    } else {
        return "failed: should have thrown error"
    }
} -result "success"

test load_checkpoint-5.6 {Error handling - nonexistent checkpoint file} -setup {
    set model [create_test_model "test_model_12"]
    set optimizer [create_test_optimizer "test_opt_12" $model]
} -body {
    ;# Test with nonexistent file
    if {[catch {torch::load_checkpoint "nonexistent.pt" $model $optimizer} error]} {
        return "success"
    } else {
        return "failed: should have thrown error for nonexistent file"
    }
} -result "success"

;# Test parameter order flexibility
test load_checkpoint-6.1 {Parameter order flexibility} -setup {
    set model [create_test_model "test_model_13"]
    set optimizer [create_test_optimizer "test_opt_13" $model]
    set checkpoint_file [file join $test_dir "test13.pt"]
    create_test_checkpoint $checkpoint_file $model $optimizer
    
    ;# Reset for loading
    set model_fresh1 [create_test_model "test_model_13"]
    set optimizer_fresh1 [create_test_optimizer "test_opt_13" $model_fresh1]
    set model_fresh2 [create_test_model "test_model_13"]
    set optimizer_fresh2 [create_test_optimizer "test_opt_13" $model_fresh2]
} -body {
    ;# Test different parameter orders
    set result1 [torch::load_checkpoint -filename $checkpoint_file -model $model_fresh1 -optimizer $optimizer_fresh1]
    set result2 [torch::load_checkpoint -optimizer $optimizer_fresh2 -model $model_fresh2 -filename $checkpoint_file]
    
    ;# Both should work and return success
    if {[string match "*Checkpoint loaded*" $result1] && [string match "*Checkpoint loaded*" $result2]} {
        return "success"
    } else {
        return "failed: parameter order should be flexible"
    }
} -result "success"

;# Test with different model and optimizer types
test load_checkpoint-7.1 {Different model types} -setup {
    ;# Create a different type of model (if available)
    set model [create_test_model "conv_model"]
    set optimizer [create_test_optimizer "conv_opt" $model]
    set checkpoint_file [file join $test_dir "conv_test.pt"]
    create_test_checkpoint $checkpoint_file $model $optimizer
} -body {
    ;# Test loading with different model type
    set result [torch::load_checkpoint -file $checkpoint_file -model $model -optimizer $optimizer]
    
    ;# Should work with any model type
    if {[string match "*Checkpoint loaded*" $result]} {
        return "success"
    } else {
        return "failed: should work with different model types"
    }
} -result "success"

;# Cleanup
test load_checkpoint-cleanup {Cleanup test files} -body {
    ;# Remove test directory and files
    if {[file exists $test_dir]} {
        file delete -force $test_dir
    }
    return "success"
} -result "success"

cleanupTests 