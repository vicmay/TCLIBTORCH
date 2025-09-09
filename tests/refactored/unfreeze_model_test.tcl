#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# ----------------------------------------------------------------------------
# Test 1: Positional syntax
# ----------------------------------------------------------------------------

test unfreeze_model-1.1 {basic positional syntax} -body {
    set model [torch::linear 4 2]
    # freeze first so we can unfreeze
    torch::freeze_model $model
    torch::unfreeze_model $model
} -result {Model parameters unfrozen}

test unfreeze_model-1.2 {wrong arg count} -body {
    catch {torch::unfreeze_model} res
    set res
} -match glob -result {*Wrong number of arguments*}

# ----------------------------------------------------------------------------
# Test 2: Named parameter syntax
# ----------------------------------------------------------------------------

test unfreeze_model-2.1 {named parameter syntax} -body {
    set model [torch::linear 4 2]
    torch::freeze_model -model $model
    torch::unfreeze_model -model $model
} -result {Model parameters unfrozen}

test unfreeze_model-2.2 {missing value} -body {
    catch {torch::unfreeze_model -model} res
    set res
} -match glob -result {*Missing value for parameter*}

# ----------------------------------------------------------------------------
# Test 3: camelCase alias
# ----------------------------------------------------------------------------

test unfreeze_model-3.1 {camelCase positional} -body {
    set model [torch::linear 4 2]
    torch::freezeModel $model
    torch::unfreezeModel $model
} -result {Model parameters unfrozen}

test unfreeze_model-3.2 {camelCase named} -body {
    set model [torch::linear 4 2]
    torch::freezeModel -model $model
    torch::unfreezeModel -model $model
} -result {Model parameters unfrozen}

# ----------------------------------------------------------------------------
# Test 4: Error handling
# ----------------------------------------------------------------------------

test unfreeze_model-4.1 {nonexistent model} -body {
    catch {torch::unfreeze_model "does_not_exist"} res
    set res
} -result {Model not found}

test unfreeze_model-4.2 {unknown parameter} -body {
    catch {torch::unfreeze_model -foo bar} res
    set res
} -match glob -result {*Unknown parameter*}

test unfreeze_model-4.3 {empty model name} -body {
    catch {torch::unfreeze_model -model ""} res
    set res
} -match glob -result {*Required parameter missing*}

# ----------------------------------------------------------------------------
# Test 5: Functionality check
# ----------------------------------------------------------------------------

test unfreeze_model-5.1 {freeze/unfreeze cycle} -body {
    set model [torch::linear 4 2]
    set res1 [torch::freeze_model $model]
    set res2 [torch::unfreeze_model $model]
    expr {$res1 eq "Model parameters frozen" && $res2 eq "Model parameters unfrozen"}
} -result {1}

cleanupTests 