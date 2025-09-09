#ifndef LIBTORCHTCL_H
#define LIBTORCHTCL_H

#if defined(_WIN32) || defined(__WIN32__)
#define LIBTORCHTCL_EXPORT __declspec(dllexport)
#else
#define LIBTORCHTCL_EXPORT
#endif

#include <tcl.h>
#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <sstream>

// Forward declarations of module classes
class ConcreteLinear;
class ConcreteConv2d;
class ConcreteMaxPool2d;
class ConcreteDropout;
class ConcreteBatchNorm2d;
class ConcreteAvgPool2d;
class ConcreteSequential;
class ConcreteBatchNorm1d;
class ConcreteLayerNorm;
class ConcreteGroupNorm;
class ConcreteConvTranspose2d;
class ConcreteLSTM;
class ConcreteGRU;
class ConcreteRNN;

// Global storage declarations
extern std::unordered_map<std::string, torch::Tensor> tensor_storage;
extern std::unordered_map<std::string, std::shared_ptr<torch::optim::Optimizer>> optimizer_storage;
extern std::unordered_map<std::string, std::shared_ptr<torch::nn::Module>> module_storage;

// Helper function declarations
c10::ScalarType GetScalarType(const char* type_str);
torch::Device GetDevice(const char* device_str);
torch::Tensor TclListToTensor(Tcl_Interp* interp, Tcl_Obj* list, 
                             const char* type_str,
                             const char* device_str,
                             bool requires_grad);
std::vector<int64_t> TclListToShape(Tcl_Interp* interp, Tcl_Obj* list);
std::string GetNextHandle(const std::string& prefix);

// Additional helper function declarations
torch::Tensor GetTensorFromObj(Tcl_Interp* interp, Tcl_Obj* obj);
int GetIntFromObj(Tcl_Interp* interp, Tcl_Obj* obj);
double GetDoubleFromObj(Tcl_Interp* interp, Tcl_Obj* obj);
bool GetBoolFromObj(Tcl_Interp* interp, Tcl_Obj* obj);
std::vector<int64_t> GetIntVectorFromObj(Tcl_Interp* interp, Tcl_Obj* obj);
int SetTensorResult(Tcl_Interp* interp, const torch::Tensor& tensor);

template<typename T>
std::shared_ptr<torch::nn::Module> convert_to_base_module(std::shared_ptr<T> derived) {
    return std::static_pointer_cast<torch::nn::Module>(derived);
}

template<typename T>
std::string StoreModule(const std::string& prefix, std::shared_ptr<T> module) {
    std::string handle = GetNextHandle(prefix);
    module_storage[handle] = convert_to_base_module(module);
    return handle;
}

// Command function declarations for neural network device management
int LayerTo_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LayerDevice_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LayerCuda_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LayerCpu_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for missing core tensor functions
int TensorRandn_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorRand_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorItem_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorNumel_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for tensor creation operations
int TensorZeros_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorOnes_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorEmpty_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorFull_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorEye_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorArange_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorLinspace_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorLogspace_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorZerosLike_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorOnesLike_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorEmptyLike_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorFullLike_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorRandLike_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorRandnLike_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorRandintLike_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for mathematical operations
// Trigonometric functions
int TensorSin_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorCos_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorTan_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAsin_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAcos_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAtan_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAtan2_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSinh_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorCosh_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAsinh_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAcosh_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAtanh_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDeg2rad_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorRad2deg_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Exponential and logarithmic functions
int TensorExp2_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorExp10_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorExpm1_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorLog2_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorLog10_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorLog1p_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorPow_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorRsqrt_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSquare_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Rounding and comparison functions
int TensorFloor_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorCeil_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorRound_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorTrunc_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorFrac_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorEq_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorNe_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorLt_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorLe_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorGt_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorGe_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorIsnan_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorIsinf_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorIsfinite_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorIsclose_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAllclose_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Logical operations
int TensorLogicalAnd_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorLogicalOr_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorLogicalNot_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorLogicalXor_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorBitwiseAnd_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorBitwiseOr_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorBitwiseNot_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorBitwiseXor_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorBitwiseLeftShift_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorBitwiseRightShift_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Reduction operations
int TensorMeanDim_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorStdDim_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorVarDim_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMedianDim_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorKthvalue_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorCumsum_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorCumprod_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorCummax_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorCummin_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDiff_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorGradient_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Activation functions (Phase 2 - Essential Deep Learning)
int TensorGelu_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSelu_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorElu_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorLeakyRelu_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorPrelu_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorRelu6_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorHardtanh_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorHardswish_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorHardsigmoid_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSilu_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMish_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSoftplus_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSoftsign_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorTanhshrink_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorThreshold_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorRrelu_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorCelu_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSoftmin_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSoftmax2d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorLogsoftmax_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorGlu_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Extended Convolution Operations (Phase 2 - Essential Deep Learning)
int TensorConv1d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorConv3d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorConvTranspose1d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorConvTranspose2d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorConvTranspose3d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorUnfold_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorFold_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Extended Pooling Operations (Phase 2 - Essential Deep Learning)
int MaxPool1d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int MaxPool2d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int MaxPool3d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMaxPool1d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMaxPool2d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMaxPool3d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAvgPool1d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAvgPool2d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAvgPool3d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAdaptiveAvgPool1d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAdaptiveAvgPool3d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAdaptiveMaxPool1d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAdaptiveMaxPool3d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorFractionalMaxPool2d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorFractionalMaxPool3d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorLpPool1d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorLpPool2d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorLpPool3d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Extended Loss Functions (Phase 2 - Essential Deep Learning)
int TensorL1Loss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSmoothL1Loss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorHuberLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorKLDivLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorCosineEmbeddingLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMarginRankingLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorTripletMarginLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorHingeEmbeddingLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorPoissonNLLLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorGaussianNLLLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorFocalLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDiceLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorTverskyLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorTripletMarginWithDistanceLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMultiMarginLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMultilabelMarginLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMultilabelSoftMarginLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSoftMarginLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for training workflow
int LayerParameters_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int ParametersTo_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int ModelTrain_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int ModelEval_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for additional optimizers
int OptimizerAdamW_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int OptimizerRMSprop_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int OptimizerMomentumSGD_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int OptimizerAdagrad_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for extended optimizers (Phase 2)
int OptimizerLBFGS_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int OptimizerRprop_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int OptimizerAdamax_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// ============================================================================
// NEW MISSING OPTIMIZERS - BATCH IMPLEMENTATION OF 6 OPTIMIZERS
// ============================================================================
int OptimizerSparseAdam_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int OptimizerNAdam_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int OptimizerRAdam_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int OptimizerAdafactor_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int OptimizerLAMB_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int OptimizerNovoGrad_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for extended learning rate schedulers (Phase 2)
int LRSchedulerLambda_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LRSchedulerExponentialDecay_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LRSchedulerCyclic_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LRSchedulerOneCycle_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LRSchedulerReduceOnPlateau_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LRSchedulerStepAdvanced_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int GetLRAdvanced_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// ============================================================================
// NEW MISSING LEARNING RATE SCHEDULERS - BATCH IMPLEMENTATION OF 12 SCHEDULERS
// ============================================================================
int LRSchedulerMultiplicative_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LRSchedulerPolynomial_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LRSchedulerCosineAnnealingWarmRestarts_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LRSchedulerLinearWithWarmup_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LRSchedulerConstantWithWarmup_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LRSchedulerMultiStep_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LRSchedulerCosineAnnealing_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LRSchedulerPlateau_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LRSchedulerInverseSqrt_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LRSchedulerNoam_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LRSchedulerOneCycleAdvanced_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// ============================================================================
// BATCH 5: EXTENDED NEURAL NETWORK LAYERS - 20 COMMANDS
// ============================================================================

// Extended Normalization Layers (10 commands)
int BatchNorm1D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int BatchNorm3D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int InstanceNorm1D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int InstanceNorm2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int InstanceNorm3D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LocalResponseNorm_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int CrossMapLRN2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int RMSNorm_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int SpectralNorm_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int WeightNorm_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Transformer Components (7 commands)
int MultiHeadAttention_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int ScaledDotProductAttention_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int PositionalEncoding_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TransformerEncoderLayer_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TransformerDecoderLayer_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TransformerEncoder_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TransformerDecoder_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Embedding Layers (3 commands)
int Embedding_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int EmbeddingBag_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int SparseEmbedding_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// ============================================================================
// TENSOR MANIPULATION EXTENSIONS - BATCH IMPLEMENTATION OF 15 OPERATIONS
// ============================================================================
int TensorFlip_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorRoll_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorRot90_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorNarrowCopy_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorTakeAlongDim_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorGatherNd_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorScatterNd_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMeshgrid_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorCombinations_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorCartesianProd_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorTensordot_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorEinsum_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorKron_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorBroadcastTensors_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAtleast1d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAtleast2d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAtleast3d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// ============================================================================
// VISION OPERATIONS - BATCH IMPLEMENTATION OF 15 OPERATIONS
// ============================================================================
int PixelShuffle_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int PixelUnshuffle_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int UpsampleNearest_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int UpsampleBilinear_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Interpolate_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int GridSample_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int AffineGrid_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int ChannelShuffle_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int NMS_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int BoxIoU_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int RoIAlign_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int RoIPool_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int NormalizeImage_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int DenormalizeImage_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int ResizeImage_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSelect_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// ============================================================================
// LINEAR ALGEBRA EXTENSIONS - BATCH IMPLEMENTATION OF 15 OPERATIONS
// ============================================================================
int TensorCross_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDot_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorOuter_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorTrace_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDiag_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDiagflat_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorTril_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorTriu_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMatrixPower_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMatrixRank_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorCond_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMatrixNorm_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorVectorNorm_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorLstsq_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSolveTriangular_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorCholeskySolve_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorLUSolve_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for loss functions
int MSELoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int CrossEntropyLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int NLLLoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int BCELoss_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for learning rate schedulers
int LRSchedulerStep_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LRSchedulerExponential_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LRSchedulerCosine_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LRSchedulerStepUpdate_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int GetLR_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for advanced layers
int BatchNorm1d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LayerNorm_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int GroupNorm_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int ConvTranspose2d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for advanced tensor operations
int TensorVar_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorStd_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorIsCuda_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorIsContiguous_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorContiguous_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorWhere_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorExpand_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorRepeat_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorIndexSelect_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMedian_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorQuantile_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMode_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for basic tensor operations
int TensorCreate_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorPrint_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorGetDtype_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorGetDevice_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorRequiresGrad_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorGetGrad_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorBackward_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAbs_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorExp_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorLog_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSqrt_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSum_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMean_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMax_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMin_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSigmoid_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorRelu_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorTanh_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorAdd_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSub_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMul_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDiv_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMatmul_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorBmm_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorTo_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorReshape_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorPermute_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorCat_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorStack_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorShape_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorToList_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for signal processing
int TensorFFT_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorIFFT_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorFFT2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorIFFT2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorRFFT_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorIRFFT_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSTFT_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorISTFT_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Padding Layer Operations
int ReflectionPad1D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int ReflectionPad2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int ReflectionPad3D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int ReplicationPad1D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int ReplicationPad2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int ReplicationPad3D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int ConstantPad1D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int ConstantPad2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int ConstantPad3D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int CircularPad1D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int CircularPad2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int CircularPad3D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int ZeroPad1D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int ZeroPad2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int ZeroPad3D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorConv1D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorConvTranspose1D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorConvTranspose2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// ============================================================================
// SIGNAL PROCESSING EXTENSIONS - BATCH IMPLEMENTATION OF 17 OPERATIONS
// ============================================================================
int TensorFFTShift_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorIFFTShift_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorRFFT2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorIRFFT2D_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorHilbert_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorBartlettWindow_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorBlackmanWindow_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorHammingWindow_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorHannWindow_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorKaiserWindow_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSpectrogram_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMelscaleFbanks_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMFCC_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorPitchShift_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorTimeStretch_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// ============================================================================
// SPARSE TENSOR OPERATIONS - BATCH IMPLEMENTATION OF 13 OPERATIONS
// ============================================================================
int TensorSparseCOO_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSparseCSR_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSparseCSC_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSparseToDense_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSparseAdd_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSparseMM_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSparseSum_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSparseSoftmax_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSparseLogSoftmax_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSparseMask_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSparseTranspose_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSparseCoalesce_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSparseReshape_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// ============================================================================
// QUANTIZATION OPERATIONS - BATCH IMPLEMENTATION OF 20 OPERATIONS
// ============================================================================
int TensorQuantizePerTensor_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorQuantizePerChannel_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDequantize_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorFakeQuantizePerTensor_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorFakeQuantizePerChannel_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorIntRepr_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorQScale_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorQZeroPoint_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorQPerChannelScales_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorQPerChannelZeroPoints_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorQPerChannelAxis_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorQuantizedAdd_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorQuantizedMul_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorQuantizedRelu_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for basic layers
int Linear_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Conv2d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int MaxPool2d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Dropout_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int BatchNorm2d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int AvgPool2d_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Sequential_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LayerForward_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Conv2dSetWeights_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for recurrent layers
int LSTM_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int GRU_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int RNNTanh_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int RNNRelu_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for basic optimizers
int OptimizerSGD_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int OptimizerAdam_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int OptimizerStep_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int OptimizerZeroGrad_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for model I/O
int SaveState_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int LoadState_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for CUDA utilities
int CudaIsAvailable_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int CudaDeviceCount_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int CudaDeviceInfo_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int CudaMemoryInfo_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for linear algebra
int TensorSVD_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorEigen_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorQR_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorCholesky_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMatrixExp_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorPinv_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

extern "C" {
// Command function declarations for AMP (Automatic Mixed Precision)
int Torch_AutocastEnable_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_AutocastDisable_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_AutocastIsEnabled_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_AutocastSetDtype_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_GradScalerNew_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_GradScalerScale_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_GradScalerStep_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_GradScalerUpdate_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_GradScalerGetScale_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_TensorMaskedFill_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_TensorClamp_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for advanced tensor operations
int Torch_TensorSlice_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_TensorAdvancedIndex_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_SparseTensorCreate_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_SparseTensorDense_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_ModelSummary_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_CountParameters_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_AllReduce_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_Broadcast_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_TensorNorm_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_TensorNormalize_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_TensorUnique_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for advanced model checkpointing
int Torch_SaveCheckpoint_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_LoadCheckpoint_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_GetCheckpointInfo_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_SaveStateDict_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_LoadStateDict_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_FreezeModel_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_UnfreezeModel_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Command function declarations for real distributed training
int Torch_DistributedInit_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_RealAllReduce_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_RealBroadcast_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_DistributedBarrier_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_GetRank_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_GetWorldSize_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int Torch_IsDistributed_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// ============================================================================
// RANDOM NUMBER GENERATION OPERATIONS - MISSING IMPLEMENTATION OF 12 OPERATIONS
// ============================================================================
int TensorManualSeed_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorInitialSeed_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSeed_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorGetRngState_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSetRngState_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorBernoulli_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMultinomial_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorNormal_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorUniform_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorExponential_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorGamma_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorPoisson_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// ============================================================================
// ADVANCED TENSOR OPERATIONS - MISSING IMPLEMENTATION OF 13 OPERATIONS
// ============================================================================
int TensorBlockDiag_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorBroadcastShapes_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSqueezeMultiple_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorUnsqueezeMultiple_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorTensorSplit_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorHSplit_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorVSplit_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDSplit_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorColumnStack_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorRowStack_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDStack_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorHStack_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorVStack_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// ============================================================================
// AUTOMATIC DIFFERENTIATION - MISSING IMPLEMENTATION OF 13 OPERATIONS
// ============================================================================
int TensorGrad_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorJacobian_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorHessian_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorVJP_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorJVP_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorFunctionalCall_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorVMap_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorGradCheck_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorGradCheckFiniteDiff_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorEnableGrad_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorNoGrad_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSetGradEnabled_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorIsGradEnabled_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// ============================================================================
// MEMORY AND PERFORMANCE - MISSING IMPLEMENTATION OF 11 OPERATIONS
// ============================================================================
int TensorMemoryStats_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMemorySummary_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorMemorySnapshot_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorEmptyCache_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSynchronize_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorProfilerStart_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorProfilerStop_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorBenchmark_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSetFlushDenormal_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorGetNumThreads_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorSetNumThreads_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// ============================================================================
// DISTRIBUTED OPERATIONS - FINAL IMPLEMENTATION OF 10 OPERATIONS
// ============================================================================
int TensorDistributedGather_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDistributedScatter_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDistributedReduceScatter_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDistributedAllToAll_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDistributedSend_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDistributedRecv_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDistributedISend_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDistributedIRecv_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDistributedWait_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
int TensorDistributedTest_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

// Tensor Info Operations
int TensorSize_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

}

#endif // LIBTORCHTCL_H 