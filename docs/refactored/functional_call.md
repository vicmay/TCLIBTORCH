# torch::functional_call

Performs a functional call of `func` with supplied `parameters` and optional extra arguments.
(Current implementation placeholder returns the parameters tensor.)

## Syntax
Positional:
```tcl
torch::functional_call func parameters ?args...?
```
Named:
```tcl
torch::functional_call -func func -parameters tensor ?-args list?
```
camelCase alias:
```tcl
torch::functionalCall ...
```

## Parameters
- `func` (string): function identifier
- `parameters` (tensor handle): tensor of parameters
- `args` (any): optional additional tensors/values

## Returns
Tensor containing the parameters (placeholder).

## Examples
```tcl
set p [torch::randn [list 2 2]]
set result [torch::functional_call myFunc $p]
```

## Migration guide
Legacy positional → named parameters as shown. 