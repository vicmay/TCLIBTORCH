# torch::optimizer_rmsprop

Creates an RMSprop optimizer with dual syntax support.

## Syntax

### Named Parameter Syntax
```tcl
torch::optimizer_rmsprop -parameters $paramList -lr 0.01 ?-alpha 0.99? ?-eps 1e-8?
# camelCase alias
torch::optimizerRmsprop -parameters $paramList -lr 0.01 -alpha 0.95 -eps 1e-8
```

### Legacy Positional Syntax
```tcl
torch::optimizer_rmsprop $paramList $learningRate ?$alpha? ?$eps?
```

## Parameters
| Option | Positional | Description | Default |
|--------|------------|-------------|---------|
| `-parameters` / `-params` | 1 | List of model parameter tensors | — |
| `-lr` / `-learningRate` | 2 | Learning rate (float > 0) | — |
| `-alpha` | 3 | Smoothing constant (float > 0) | 0.99 |
| `-eps` / `-epsilon` | 4 | Epsilon for numerical stability | 1e-8 |

## Returns
String handle identifying the optimizer.

## Examples
```tcl
# Named syntax
set opt [torch::optimizer_rmsprop -parameters $params -lr 0.005 -alpha 0.9]

# Positional syntax (legacy)
set opt [torch::optimizer_rmsprop $params 0.005 0.9 1e-8]

# camelCase alias
set opt [torch::optimizerRmsprop -parameters $params -lr 0.01]
```

## Error Handling
The command validates all numeric inputs and presence of required parameters, raising descriptive errors for invalid usage.

## Compatibility
• 100 % backward compatible with positional syntax.  
• Modern named-parameter syntax and camelCase alias enhance readability. 