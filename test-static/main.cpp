#include <tcl.h>
#include <iostream>

extern "C" int Torchtcl_Init(Tcl_Interp* interp);

int main(int argc, char* argv[]) {
    Tcl_Interp* interp = Tcl_CreateInterp();
    if (Tcl_Init(interp) == TCL_ERROR) {
        std::cerr << "Error initializing Tcl: " << Tcl_GetStringResult(interp) << std::endl;
        return 1;
    }

    if (Torchtcl_Init(interp) == TCL_ERROR) {
        std::cerr << "Error initializing LibTorch TCL: " << Tcl_GetStringResult(interp) << std::endl;
        return 1;
    }

    // Run the test script
    if (Tcl_EvalFile(interp, "test.tcl") == TCL_ERROR) {
        std::cerr << "Error running test script: " << Tcl_GetStringResult(interp) << std::endl;
        return 1;
    }

    Tcl_DeleteInterp(interp);
    return 0;
}
