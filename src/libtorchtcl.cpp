/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
 * Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
 * Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
 * Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
 * Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
 * Copyright (c) 2011-2013 NYU                      (Clement Farabet)
 * Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
 * Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
 * Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
 *    and IDIAP Research Institute nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "libtorchtcl.h"

// Initialize the extension
extern "C" {
#if defined(_WIN32) || defined(__WIN32__)
    LIBTORCHTCL_EXPORT
#endif
    int Torchtcl_Init(Tcl_Interp* interp) {
        if (Tcl_InitStubs(interp, "8.5", 0) == NULL) {
            return TCL_ERROR;
        }

        // Skip CUDA initialization to avoid warnings on older hardware
        // CUDA will be initialized on-demand when needed

        // Create namespace
        Tcl_CreateNamespace(interp, "torch", NULL, NULL);
        
        // Register basic tensor commands
        Tcl_CreateObjCommand(interp, "torch::tensor_create", TensorCreate_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorCreate", TensorCreate_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_print", TensorPrint_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorPrint", TensorPrint_Cmd, NULL, NULL);
        
        // Arithmetic operations
        Tcl_CreateObjCommand(interp, "torch::tensor_add", TensorAdd_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorAdd", TensorAdd_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_sub", TensorSub_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorSub", TensorSub_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_mul", TensorMul_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorMul", TensorMul_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_div", TensorDiv_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorDiv", TensorDiv_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_matmul", TensorMatmul_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorMatmul", TensorMatmul_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_bmm", TensorBmm_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorBmm", TensorBmm_Cmd, NULL, NULL);
        
        // Advanced operations
        Tcl_CreateObjCommand(interp, "torch::tensor_abs", TensorAbs_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorAbs", TensorAbs_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_exp", TensorExp_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorExp", TensorExp_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_log", TensorLog_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorLog", TensorLog_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_sqrt", TensorSqrt_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorSqrt", TensorSqrt_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_sum", TensorSum_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorSum", TensorSum_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_mean", TensorMean_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorMean", TensorMean_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_max", TensorMax_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorMax", TensorMax_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_min", TensorMin_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorMin", TensorMin_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_sigmoid", TensorSigmoid_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorSigmoid", TensorSigmoid_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_relu", TensorRelu_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorRelu", TensorRelu_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_tanh", TensorTanh_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorTanh", TensorTanh_Cmd, NULL, NULL);
        
        // Property getters
        Tcl_CreateObjCommand(interp, "torch::tensor_dtype", TensorGetDtype_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorDtype", TensorGetDtype_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_device", TensorGetDevice_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorDevice", TensorGetDevice_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_requires_grad", TensorRequiresGrad_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorRequiresGrad", TensorRequiresGrad_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_grad", TensorGetGrad_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorGrad", TensorGetGrad_Cmd, NULL, NULL);
        
        // Device operations
        Tcl_CreateObjCommand(interp, "torch::tensor_to", TensorTo_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorTo", TensorTo_Cmd, NULL, NULL);
        
        // Gradient operations
        Tcl_CreateObjCommand(interp, "torch::tensor_backward", TensorBackward_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorBackward", TensorBackward_Cmd, NULL, NULL);

        // Signal processing operations
        Tcl_CreateObjCommand(interp, "torch::tensor_fft", TensorFFT_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorFft", TensorFFT_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_ifft", TensorIFFT_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorIfft", TensorIFFT_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_fft2d", TensorFFT2D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorFft2d", TensorFFT2D_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_ifft2d", TensorIFFT2D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorIfft2d", TensorIFFT2D_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_rfft", TensorRFFT_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorRfft", TensorRFFT_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_irfft", TensorIRFFT_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorIrfft", TensorIRFFT_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_stft", TensorSTFT_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorStft", TensorSTFT_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_istft", TensorISTFT_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorIstft", TensorISTFT_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_conv1d", TensorConv1D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorConv1d", TensorConv1D_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_conv_transpose1d", TensorConvTranspose1D_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::tensorConvTranspose1d", TensorConvTranspose1D_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_conv_transpose2d", TensorConvTranspose2D_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::tensorConvTranspose2d", TensorConvTranspose2D_Cmd, NULL, NULL);  // camelCase alias

        // Padding layer operations
        Tcl_CreateObjCommand(interp, "torch::reflection_pad1d", ReflectionPad1D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::reflectionPad1d", ReflectionPad1D_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::reflection_pad2d", ReflectionPad2D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::reflectionPad2d", ReflectionPad2D_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::reflection_pad3d", ReflectionPad3D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::reflectionPad3d", ReflectionPad3D_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::replication_pad1d", ReplicationPad1D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::replicationPad1d", ReplicationPad1D_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::replication_pad2d", ReplicationPad2D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::replicationPad2d", ReplicationPad2D_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::replication_pad3d", ReplicationPad3D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::replicationPad3d", ReplicationPad3D_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::circular_pad1d", CircularPad1D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::circularPad1d", CircularPad1D_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::circular_pad2d", CircularPad2D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::circularPad2d", CircularPad2D_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::circular_pad3d", CircularPad3D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::circularPad3d", CircularPad3D_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::zero_pad1d", ZeroPad1D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::zeroPad1d", ZeroPad1D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::zero_pad2d", ZeroPad2D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::zeroPad2d", ZeroPad2D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::zero_pad3d", ZeroPad3D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::zeroPad3d", ZeroPad3D_Cmd, NULL, NULL);

        // Neural Network Layer Commands
        Tcl_CreateObjCommand(interp, "torch::linear", Linear_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::linearLayer", Linear_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::conv2d", Conv2d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::conv2dLayer", Conv2d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::batchnorm2d", BatchNorm2d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::batchNorm2d", BatchNorm2d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::maxpool2d", MaxPool2d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::maxPool2d", MaxPool2d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::maxpool1d", MaxPool1d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::maxPool1d", MaxPool1d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::maxpool3d", MaxPool3d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::maxPool3d", MaxPool3d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::dropout", Dropout_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::avgpool2d", AvgPool2d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::avgPool2d", AvgPool2d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::sequential", Sequential_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::layer_forward", LayerForward_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::layerForward", LayerForward_Cmd, NULL, NULL);  // camelCase alias

        // Recurrent Neural Network Layer Commands
        Tcl_CreateObjCommand(interp, "torch::lstm", LSTM_Cmd, NULL, NULL);  // Already in camelCase format
        Tcl_CreateObjCommand(interp, "torch::gru", GRU_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Gru", GRU_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::rnn_tanh", RNNTanh_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::rnnTanh", RNNTanh_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::rnn_relu", RNNRelu_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::rnnRelu", RNNRelu_Cmd, NULL, NULL);  // camelCase alias

        // Optimizer Commands
        Tcl_CreateObjCommand(interp, "torch::optimizer_sgd", OptimizerSGD_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::optimizerSgd", OptimizerSGD_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::optimizer_adam", OptimizerAdam_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::optimizerAdam", OptimizerAdam_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::optimizer_step", OptimizerStep_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::optimizerStep", OptimizerStep_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::optimizer_zero_grad", OptimizerZeroGrad_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::optimizerZeroGrad", OptimizerZeroGrad_Cmd, NULL, NULL);  // camelCase alias

        // Serialization Commands
        Tcl_CreateObjCommand(interp, "torch::save_checkpoint", Torch_SaveCheckpoint_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::saveCheckpoint", Torch_SaveCheckpoint_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::load_state", LoadState_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::loadState", LoadState_Cmd, NULL, NULL);  // camelCase alias

        // Register tensor manipulation operations
        Tcl_CreateObjCommand(interp, "torch::tensor_reshape", TensorReshape_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorReshape", TensorReshape_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_permute", TensorPermute_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorPermute", TensorPermute_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_cat", TensorCat_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorCat", TensorCat_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_stack", TensorStack_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorStack", TensorStack_Cmd, NULL, NULL);

        // Add tensor shape check command
        Tcl_CreateObjCommand(interp, "torch::tensor_shape", TensorShape_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorShape", TensorShape_Cmd, NULL, NULL);

        // Add tensor_to_list command
        Tcl_CreateObjCommand(interp, "torch::tensor_to_list", TensorToList_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorToList", TensorToList_Cmd, NULL, NULL);  // camelCase alias

        // Register Conv2dSetWeights_Cmd
        Tcl_CreateObjCommand(interp, "torch::conv2d_set_weights", Conv2dSetWeights_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::conv2dSetWeights", Conv2dSetWeights_Cmd, NULL, NULL);  // camelCase alias

        // Register CUDA commands
        Tcl_CreateObjCommand(interp, "torch::cuda_is_available", CudaIsAvailable_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::cudaIsAvailable", CudaIsAvailable_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::cuda_device_count", CudaDeviceCount_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::cudaDeviceCount", CudaDeviceCount_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::cuda_device_info", CudaDeviceInfo_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::cudaDeviceInfo", CudaDeviceInfo_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::cuda_memory_info", CudaMemoryInfo_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::cudaMemoryInfo", CudaMemoryInfo_Cmd, NULL, NULL);  // camelCase alias

        // Register advanced math commands
        Tcl_CreateObjCommand(interp, "torch::tensor_svd", TensorSVD_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorSvd", TensorSVD_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_eigen", TensorEigen_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorEigen", TensorEigen_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_qr", TensorQR_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorQr", TensorQR_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_cholesky", TensorCholesky_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::tensorCholesky", TensorCholesky_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_matrix_exp", TensorMatrixExp_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorMatrixExp", TensorMatrixExp_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_pinv", TensorPinv_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorPinv", TensorPinv_Cmd, NULL, NULL);  // camelCase alias

        // Register new neural network device management commands
        Tcl_CreateObjCommand(interp, "torch::layer_to", LayerTo_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::layerTo", LayerTo_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::layer_device", LayerDevice_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::layerDevice", LayerDevice_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::layer_cuda", LayerCuda_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::layerCuda", LayerCuda_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::layer_cpu", LayerCpu_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::layerCpu", LayerCpu_Cmd, NULL, NULL);  // camelCase alias

        // Register new core tensor functions
        Tcl_CreateObjCommand(interp, "torch::tensor_randn", TensorRandn_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorRandn", TensorRandn_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_rand", TensorRand_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_item", TensorItem_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorItem", TensorItem_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorRand", TensorRand_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_numel", TensorNumel_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorNumel", TensorNumel_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::zeros", TensorZeros_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::ones", TensorOnes_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::empty", TensorEmpty_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Empty", TensorEmpty_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::full", TensorFull_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::eye", TensorEye_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::arange", TensorArange_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::linspace", TensorLinspace_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::logspace", TensorLogspace_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::zeros_like", TensorZerosLike_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::zerosLike", TensorZerosLike_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::ones_like", TensorOnesLike_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::onesLike", TensorOnesLike_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::empty_like", TensorEmptyLike_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::emptyLike", TensorEmptyLike_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::full_like", TensorFullLike_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::fullLike", TensorFullLike_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::rand_like", TensorRandLike_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::randLike", TensorRandLike_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::randn_like", TensorRandnLike_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::randnLike", TensorRandnLike_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::randint_like", TensorRandintLike_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::randintLike", TensorRandintLike_Cmd, NULL, NULL);

        // Register mathematical operations
        // Trigonometric functions
        Tcl_CreateObjCommand(interp, "torch::sin", TensorSin_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::cos", TensorCos_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::cos", TensorCos_Cmd, NULL, NULL);  // camelCase alias (same as snake_case)
        Tcl_CreateObjCommand(interp, "torch::tan", TensorTan_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::asin", TensorAsin_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::acos", TensorAcos_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::acos", TensorAcos_Cmd, NULL, NULL);  // camelCase alias (same as snake_case)
        Tcl_CreateObjCommand(interp, "torch::atan", TensorAtan_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::atan", TensorAtan_Cmd, NULL, NULL);  // camelCase alias (same as snake_case)
        Tcl_CreateObjCommand(interp, "torch::atan2", TensorAtan2_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::sinh", TensorSinh_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::siNh", TensorSinh_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::cosh", TensorCosh_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::cosh", TensorCosh_Cmd, NULL, NULL);  // camelCase alias (same as snake_case)
        Tcl_CreateObjCommand(interp, "torch::asinh", TensorAsinh_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::acosh", TensorAcosh_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::atanh", TensorAtanh_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::deg2rad", TensorDeg2rad_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::deg2Rad", TensorDeg2rad_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::rad2deg", TensorRad2deg_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::radToDeg", TensorRad2deg_Cmd, NULL, NULL);  // camelCase alias

        // Exponential and logarithmic functions
        Tcl_CreateObjCommand(interp, "torch::exp2", TensorExp2_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::exp10", TensorExp10_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::expm1", TensorExpm1_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::expm1", TensorExpm1_Cmd, NULL, NULL);  // camelCase alias (same as snake_case)
        Tcl_CreateObjCommand(interp, "torch::log2", TensorLog2_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::log2", TensorLog2_Cmd, NULL, NULL);  // camelCase alias (same as snake_case)
        Tcl_CreateObjCommand(interp, "torch::log10", TensorLog10_Cmd, NULL, NULL);  // camelCase alias (same as snake_case)
        Tcl_CreateObjCommand(interp, "torch::log1p", TensorLog1p_Cmd, NULL, NULL);  // camelCase alias (same as snake_case)
        Tcl_CreateObjCommand(interp, "torch::pow", TensorPow_Cmd, NULL, NULL);  // Already camelCase compatible
        Tcl_CreateObjCommand(interp, "torch::rsqrt", TensorRsqrt_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::rSqrt", TensorRsqrt_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::square", TensorSquare_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Square", TensorSquare_Cmd, NULL, NULL);  // camelCase alias

        // Rounding and comparison functions
        Tcl_CreateObjCommand(interp, "torch::floor", TensorFloor_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Floor", TensorFloor_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::ceil", TensorCeil_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::round", TensorRound_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Round", TensorRound_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::trunc", TensorTrunc_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::frac", TensorFrac_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Frac", TensorFrac_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::eq", TensorEq_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Eq", TensorEq_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::ne", TensorNe_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Ne", TensorNe_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lt", TensorLt_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Lt", TensorLt_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::le", TensorLe_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::Le", TensorLe_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::gt", TensorGt_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Gt", TensorGt_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::ge", TensorGe_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::isnan", TensorIsnan_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::isNan", TensorIsnan_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::isinf", TensorIsinf_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::isInf", TensorIsinf_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::isfinite", TensorIsfinite_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::isFinite", TensorIsfinite_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::isclose", TensorIsclose_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::isClose", TensorIsclose_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::allclose", TensorAllclose_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::allClose", TensorAllclose_Cmd, NULL, NULL);  // camelCase alias

        // Logical operations
        Tcl_CreateObjCommand(interp, "torch::logical_and", TensorLogicalAnd_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::logicalAnd", TensorLogicalAnd_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::logical_or", TensorLogicalOr_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::logicalOr", TensorLogicalOr_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::logical_not", TensorLogicalNot_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::logicalNot", TensorLogicalNot_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::logical_xor", TensorLogicalXor_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::logicalXor", TensorLogicalXor_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::bitwise_and", TensorBitwiseAnd_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::bitwiseAnd", TensorBitwiseAnd_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::bitwise_or", TensorBitwiseOr_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::bitwiseOr", TensorBitwiseOr_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::bitwise_not", TensorBitwiseNot_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::bitwiseNot", TensorBitwiseNot_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::bitwise_xor", TensorBitwiseXor_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::bitwiseXor", TensorBitwiseXor_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::bitwise_left_shift", TensorBitwiseLeftShift_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::bitwiseLeftShift", TensorBitwiseLeftShift_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::bitwise_right_shift", TensorBitwiseRightShift_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::bitwiseRightShift", TensorBitwiseRightShift_Cmd, NULL, NULL);  // camelCase alias

        // Reduction operations
        Tcl_CreateObjCommand(interp, "torch::mean_dim", TensorMeanDim_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::meanDim", TensorMeanDim_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::std_dim", TensorStdDim_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::stdDim", TensorStdDim_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::var_dim", TensorVarDim_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::varDim", TensorVarDim_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::median_dim", TensorMedianDim_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::medianDim", TensorMedianDim_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::kthvalue", TensorKthvalue_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::kthValue", TensorKthvalue_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::cumsum", TensorCumsum_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::cumSum", TensorCumsum_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::cumprod", TensorCumprod_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::cumProd", TensorCumprod_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::cummax", TensorCummax_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::cumMax", TensorCummax_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::cummin", TensorCummin_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::cumMin", TensorCummin_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::diff", TensorDiff_Cmd, NULL, NULL);  // Already in camelCase format
        Tcl_CreateObjCommand(interp, "torch::gradient", TensorGradient_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::gradientCmd", TensorGradient_Cmd, NULL, NULL);  // camelCase alias

        // Activation functions (Phase 2 - Essential Deep Learning)
        Tcl_CreateObjCommand(interp, "torch::gelu", TensorGelu_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::selu", TensorSelu_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::elu", TensorElu_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::Elu", TensorElu_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::leaky_relu", TensorLeakyRelu_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::leakyRelu", TensorLeakyRelu_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::prelu", TensorPrelu_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::relu6", TensorRelu6_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::relu6", TensorRelu6_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::hardtanh", TensorHardtanh_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::hardTanh", TensorHardtanh_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::hardswish", TensorHardswish_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::hardSwish", TensorHardswish_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::hardsigmoid", TensorHardsigmoid_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::hardSigmoid", TensorHardsigmoid_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::silu", TensorSilu_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::siLU", TensorSilu_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::mish", TensorMish_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::softplus", TensorSoftplus_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::softPlus", TensorSoftplus_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::softsign", TensorSoftsign_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::softSign", TensorSoftsign_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tanhshrink", TensorTanhshrink_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tanhShrink", TensorTanhshrink_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::threshold", TensorThreshold_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Threshold", TensorThreshold_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::rrelu", TensorRrelu_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::rRelu", TensorRrelu_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::celu", TensorCelu_Cmd, NULL, NULL);
        // Note: celu doesn't need camelCase alias (already camelCase)
        Tcl_CreateObjCommand(interp, "torch::softmin", TensorSoftmin_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::softMin", TensorSoftmin_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::softmax2d", TensorSoftmax2d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::softmax2D", TensorSoftmax2d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::logsoftmax", TensorLogsoftmax_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::logSoftmax", TensorLogsoftmax_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::glu", TensorGlu_Cmd, NULL, NULL);

        // Extended convolution operations (Phase 2 - Essential Deep Learning)
        Tcl_CreateObjCommand(interp, "torch::conv1d", TensorConv1d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::conv3d", TensorConv3d_Cmd, NULL, NULL);  // already camelCase
        Tcl_CreateObjCommand(interp, "torch::conv_transpose1d", TensorConvTranspose1d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::convTranspose1d", TensorConvTranspose1d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::conv_transpose3d", TensorConvTranspose3d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::convTranspose3d", TensorConvTranspose3d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::unfold", TensorUnfold_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::fold", TensorFold_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Fold", TensorFold_Cmd, NULL, NULL);  // camelCase alias

        // Extended pooling operations (Phase 2 - Essential Deep Learning)
        Tcl_CreateObjCommand(interp, "torch::maxpool1d", MaxPool1d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::maxPool1d", MaxPool1d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::maxpool2d", TensorMaxPool2d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::maxPool2d", TensorMaxPool2d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::maxpool3d", MaxPool3d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::maxPool3d", MaxPool3d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::avgpool1d", TensorAvgPool1d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::avgPool1d", TensorAvgPool1d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::avgpool2d", TensorAvgPool2d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::avgPool2d", TensorAvgPool2d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::avgpool3d", TensorAvgPool3d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::avgPool3d", TensorAvgPool3d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::adaptive_avgpool1d", TensorAdaptiveAvgPool1d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::adaptiveAvgpool1d", TensorAdaptiveAvgPool1d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::adaptive_avgpool3d", TensorAdaptiveAvgPool3d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::adaptiveAvgpool3d", TensorAdaptiveAvgPool3d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::adaptive_maxpool1d", TensorAdaptiveMaxPool1d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::adaptiveMaxpool1d", TensorAdaptiveMaxPool1d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::adaptive_maxpool3d", TensorAdaptiveMaxPool3d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::adaptiveMaxpool3d", TensorAdaptiveMaxPool3d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::fractional_maxpool2d", TensorFractionalMaxPool2d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::fractionalMaxpool2d", TensorFractionalMaxPool2d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::fractional_maxpool3d", TensorFractionalMaxPool3d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::fractionalMaxpool3d", TensorFractionalMaxPool3d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lppool1d", TensorLpPool1d_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::lpPool1d", TensorLpPool1d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::lppool2d", TensorLpPool2d_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::lpPool2d", TensorLpPool2d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::lppool3d", TensorLpPool3d_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::lpPool3d", TensorLpPool3d_Cmd, NULL, NULL);

        // Extended loss functions (Phase 2 - Essential Deep Learning)
        Tcl_CreateObjCommand(interp, "torch::l1_loss", TensorL1Loss_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::l1Loss", TensorL1Loss_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::smooth_l1_loss", TensorSmoothL1Loss_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::smoothL1Loss", TensorSmoothL1Loss_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::huber_loss", TensorHuberLoss_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::huberLoss", TensorHuberLoss_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::kl_div_loss", TensorKLDivLoss_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::klDivLoss", TensorKLDivLoss_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::cosine_embedding_loss", TensorCosineEmbeddingLoss_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::cosineEmbeddingLoss", TensorCosineEmbeddingLoss_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::margin_ranking_loss", TensorMarginRankingLoss_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::marginRankingLoss", TensorMarginRankingLoss_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::triplet_margin_loss", TensorTripletMarginLoss_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::tripletMarginLoss", TensorTripletMarginLoss_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::hinge_embedding_loss", TensorHingeEmbeddingLoss_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::hingeEmbeddingLoss", TensorHingeEmbeddingLoss_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::poisson_nll_loss", TensorPoissonNLLLoss_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::poissonNllLoss", TensorPoissonNLLLoss_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::gaussian_nll_loss", TensorGaussianNLLLoss_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::gaussianNllLoss", TensorGaussianNLLLoss_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::focal_loss", TensorFocalLoss_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::focalLoss", TensorFocalLoss_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::dice_loss", TensorDiceLoss_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::diceLoss", TensorDiceLoss_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tversky_loss", TensorTverskyLoss_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::tverskyLoss", TensorTverskyLoss_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::triplet_margin_with_distance_loss", TensorTripletMarginWithDistanceLoss_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::tripletMarginWithDistanceLoss", TensorTripletMarginWithDistanceLoss_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::multi_margin_loss", TensorMultiMarginLoss_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::multiMarginLoss", TensorMultiMarginLoss_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::multilabel_margin_loss", TensorMultilabelMarginLoss_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::multilabelMarginLoss", TensorMultilabelMarginLoss_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::multilabel_soft_margin_loss", TensorMultilabelSoftMarginLoss_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::multilabelSoftMarginLoss", TensorMultilabelSoftMarginLoss_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::soft_margin_loss", TensorSoftMarginLoss_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::softMarginLoss", TensorSoftMarginLoss_Cmd, NULL, NULL);  // camelCase alias

        // Register new training workflow commands
        Tcl_CreateObjCommand(interp, "torch::layer_parameters", LayerParameters_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::layerParameters", LayerParameters_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::parameters_to", ParametersTo_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::parametersTo", ParametersTo_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::model_train", ModelTrain_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::modelTrain", ModelTrain_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::model_eval", ModelEval_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::modelEval", ModelEval_Cmd, NULL, NULL);  // camelCase alias

        // Register additional optimizers
        Tcl_CreateObjCommand(interp, "torch::optimizer_adamw", OptimizerAdamW_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::optimizerAdamW", OptimizerAdamW_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::optimizer_rmsprop", OptimizerRMSprop_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::optimizerRmsprop", OptimizerRMSprop_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::optimizer_momentum_sgd", OptimizerMomentumSGD_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::optimizer_adagrad", OptimizerAdagrad_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::optimizerAdagrad", OptimizerAdagrad_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::optimizer_momentum_sgd", OptimizerMomentumSGD_Cmd, NULL, NULL);  // duplicate removal
        Tcl_CreateObjCommand(interp, "torch::optimizerMomentumSgd", OptimizerMomentumSGD_Cmd, NULL, NULL);  // camelCase alias

        // Register extended optimizers (Phase 2)
        Tcl_CreateObjCommand(interp, "torch::optimizer_lbfgs", OptimizerLBFGS_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::optimizerLbfgs", OptimizerLBFGS_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::optimizer_rprop", OptimizerRprop_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::optimizerRprop", OptimizerRprop_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::optimizer_adamax", OptimizerAdamax_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::optimizerAdamax", OptimizerAdamax_Cmd, NULL, NULL);  // camelCase alias

        // ============================================================================
        // REGISTER NEW MISSING OPTIMIZERS - BATCH IMPLEMENTATION OF 6 OPTIMIZERS
        // ============================================================================
        Tcl_CreateObjCommand(interp, "torch::optimizer_sparse_adam", OptimizerSparseAdam_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::optimizerSparseAdam", OptimizerSparseAdam_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::optimizer_nadam", OptimizerNAdam_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::optimizerNadam", OptimizerNAdam_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::optimizer_radam", OptimizerRAdam_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::optimizerRAdam", OptimizerRAdam_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::optimizer_adafactor", OptimizerAdafactor_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::optimizerAdafactor", OptimizerAdafactor_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::optimizer_lamb", OptimizerLAMB_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::optimizerLamb", OptimizerLAMB_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::optimizer_novograd", OptimizerNovoGrad_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::optimizerNovograd", OptimizerNovoGrad_Cmd, NULL, NULL);  // camelCase alias

        // Register extended learning rate schedulers (Phase 2)
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_lambda", LRSchedulerLambda_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::lrSchedulerLambda", LRSchedulerLambda_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_exponential_decay", LRSchedulerExponentialDecay_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::lrSchedulerExponentialDecay", LRSchedulerExponentialDecay_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_cyclic", LRSchedulerCyclic_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::lrSchedulerCyclic", LRSchedulerCyclic_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_one_cycle", LRSchedulerOneCycle_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::lrSchedulerOneCycle", LRSchedulerOneCycle_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_reduce_on_plateau", LRSchedulerReduceOnPlateau_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::lrSchedulerReduceOnPlateau", LRSchedulerReduceOnPlateau_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_step_advanced", LRSchedulerStepAdvanced_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::lrSchedulerStepAdvanced", LRSchedulerStepAdvanced_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::get_lr_advanced", GetLRAdvanced_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::getLrAdvanced", GetLRAdvanced_Cmd, NULL, NULL);  // camelCase alias

        // ============================================================================
        // REGISTER NEW MISSING LEARNING RATE SCHEDULERS - BATCH IMPLEMENTATION OF 12 SCHEDULERS
        // ============================================================================
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_multiplicative", LRSchedulerMultiplicative_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::lrSchedulerMultiplicative", LRSchedulerMultiplicative_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_polynomial", LRSchedulerPolynomial_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::lrSchedulerPolynomial", LRSchedulerPolynomial_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_cosine_annealing_warm_restarts", LRSchedulerCosineAnnealingWarmRestarts_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::lrSchedulerCosineAnnealingWarmRestarts", LRSchedulerCosineAnnealingWarmRestarts_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_linear_with_warmup", LRSchedulerLinearWithWarmup_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::lrSchedulerLinearWithWarmup", LRSchedulerLinearWithWarmup_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_constant_with_warmup", LRSchedulerConstantWithWarmup_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::lrSchedulerConstantWithWarmup", LRSchedulerConstantWithWarmup_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_multi_step", LRSchedulerMultiStep_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::lrSchedulerMultiStep", LRSchedulerMultiStep_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_cosine_annealing", LRSchedulerCosineAnnealing_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::lrSchedulerCosineAnnealing", LRSchedulerCosineAnnealing_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_plateau", LRSchedulerPlateau_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::lrSchedulerPlateau", LRSchedulerPlateau_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_inverse_sqrt", LRSchedulerInverseSqrt_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::lrSchedulerInverseSqrt", LRSchedulerInverseSqrt_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_noam", LRSchedulerNoam_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::lrSchedulerNoam", LRSchedulerNoam_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_onecycle_advanced", LRSchedulerOneCycleAdvanced_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::lrSchedulerOnecycleAdvanced", LRSchedulerOneCycleAdvanced_Cmd, NULL, NULL);  // camelCase alias

        // ============================================================================
        // REGISTER BATCH 5: EXTENDED NEURAL NETWORK LAYERS - 20 COMMANDS
        // ============================================================================
        
        // Extended Normalization Layers (10 commands)
        Tcl_CreateObjCommand(interp, "torch::batch_norm1d", BatchNorm1d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::batch_norm_1d", BatchNorm1d_Cmd, NULL, NULL);  // legacy alias
        Tcl_CreateObjCommand(interp, "torch::batchNorm1d", BatchNorm1d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::batch_norm3d", BatchNorm3D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::batchNorm3d", BatchNorm3D_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::instance_norm1d", InstanceNorm1D_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::instanceNorm1d", InstanceNorm1D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::instance_norm2d", InstanceNorm2D_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::instanceNorm2d", InstanceNorm2D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::instance_norm3d", InstanceNorm3D_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::instanceNorm3d", InstanceNorm3D_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::local_response_norm", LocalResponseNorm_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::localResponseNorm", LocalResponseNorm_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::cross_map_lrn2d", CrossMapLRN2D_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::crossMapLrn2d", CrossMapLRN2D_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::rms_norm", RMSNorm_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::rmsNorm", RMSNorm_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::spectral_norm", SpectralNorm_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::spectralNorm", SpectralNorm_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::weight_norm", WeightNorm_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::weightNorm", WeightNorm_Cmd, NULL, NULL);
        
        // Transformer Components (7 commands)
        Tcl_CreateObjCommand(interp, "torch::multihead_attention", MultiHeadAttention_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::multiheadAttention", MultiHeadAttention_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::scaled_dot_product_attention", ScaledDotProductAttention_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::positional_encoding", PositionalEncoding_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::positionalEncoding", PositionalEncoding_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::transformer_encoder_layer", TransformerEncoderLayer_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::transformerEncoderLayer", TransformerEncoderLayer_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::transformer_decoder_layer", TransformerDecoderLayer_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::transformerDecoderLayer", TransformerDecoderLayer_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::transformer_encoder", TransformerEncoder_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::transformerEncoder", TransformerEncoder_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::transformer_decoder", TransformerDecoder_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::transformerDecoder", TransformerDecoder_Cmd, NULL, NULL);  // camelCase alias
        
        // Embedding Layers (3 commands)
        Tcl_CreateObjCommand(interp, "torch::embedding", Embedding_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Embedding", Embedding_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::embedding_bag", EmbeddingBag_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::embeddingBag", EmbeddingBag_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::sparse_embedding", SparseEmbedding_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::sparseEmbedding", SparseEmbedding_Cmd, NULL, NULL);  // camelCase alias

        // ============================================================================
        // REGISTER TENSOR MANIPULATION EXTENSIONS - BATCH IMPLEMENTATION OF 15 OPERATIONS  
        // ============================================================================
        Tcl_CreateObjCommand(interp, "torch::flip", TensorFlip_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Flip", TensorFlip_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::roll", TensorRoll_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Roll", TensorRoll_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::rot90", TensorRot90_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Rot90", TensorRot90_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::narrow_copy", TensorNarrowCopy_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::narrowCopy", TensorNarrowCopy_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::take_along_dim", TensorTakeAlongDim_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::takeAlongDim", TensorTakeAlongDim_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::gather_nd", TensorGatherNd_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::gatherNd", TensorGatherNd_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::scatter_nd", TensorScatterNd_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::scatterNd", TensorScatterNd_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::meshgrid", TensorMeshgrid_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::meshGrid", TensorMeshgrid_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::combinations", TensorCombinations_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::cartesian_prod", TensorCartesianProd_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::cartesianProd", TensorCartesianProd_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensordot", TensorTensordot_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorDot", TensorTensordot_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::einsum", TensorEinsum_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Einsum", TensorEinsum_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::kron", TensorKron_Cmd, NULL, NULL);
        // Note: torch::kron is already camelCase, no alias needed
        Tcl_CreateObjCommand(interp, "torch::broadcast_tensors", TensorBroadcastTensors_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::broadcastTensors", TensorBroadcastTensors_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::atleast_1d", TensorAtleast1d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::atleast1d", TensorAtleast1d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::atleast_2d", TensorAtleast2d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::atleast2d", TensorAtleast2d_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::atleast_3d", TensorAtleast3d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::atleast3d", TensorAtleast3d_Cmd, NULL, NULL);  // camelCase alias

        // ============================================================================
        // REGISTER VISION OPERATIONS - BATCH IMPLEMENTATION OF 15 OPERATIONS
        // ============================================================================
        Tcl_CreateObjCommand(interp, "torch::pixel_shuffle", PixelShuffle_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::pixelShuffle", PixelShuffle_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::pixel_unshuffle", PixelUnshuffle_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::pixelUnshuffle", PixelUnshuffle_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::upsample_nearest", UpsampleNearest_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::upsampleNearest", UpsampleNearest_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::upsample_bilinear", UpsampleBilinear_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::upsampleBilinear", UpsampleBilinear_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::interpolate", Interpolate_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::grid_sample", GridSample_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::gridSample", GridSample_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::affine_grid", AffineGrid_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::affineGrid", AffineGrid_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::channel_shuffle", ChannelShuffle_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::channelShuffle", ChannelShuffle_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::nms", NMS_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Nms", NMS_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::box_iou", BoxIoU_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::boxIou", BoxIoU_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::roi_align", RoIAlign_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::roiAlign", RoIAlign_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::roi_pool", RoIPool_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::roiPool", RoIPool_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::normalize_image", NormalizeImage_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::normalizeImage", NormalizeImage_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::denormalize_image", DenormalizeImage_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::denormalizeImage", DenormalizeImage_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::resize_image", ResizeImage_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::resizeImage", ResizeImage_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::rms_norm", RMSNorm_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::rmsNorm", RMSNorm_Cmd, NULL, NULL);  // camelCase alias

        // ============================================================================
        // REGISTER LINEAR ALGEBRA EXTENSIONS - BATCH IMPLEMENTATION OF 15 OPERATIONS
        // ============================================================================
        Tcl_CreateObjCommand(interp, "torch::cross", TensorCross_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::cross", TensorCross_Cmd, NULL, NULL);  // camelCase alias (same as snake_case)
        Tcl_CreateObjCommand(interp, "torch::dot", TensorDot_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::outer", TensorOuter_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Outer", TensorOuter_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::trace", TensorTrace_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Trace", TensorTrace_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::diag", TensorDiag_Cmd, NULL, NULL);  // Note: diag is already camelCase (no underscores)
        Tcl_CreateObjCommand(interp, "torch::diagflat", TensorDiagflat_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::diagFlat", TensorDiagflat_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tril", TensorTril_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::triu", TensorTriu_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::matrix_power", TensorMatrixPower_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::matrixPower", TensorMatrixPower_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::matrix_rank", TensorMatrixRank_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::matrixRank", TensorMatrixRank_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::cond", TensorCond_Cmd, NULL, NULL);  // Note: cond is already camelCase
        Tcl_CreateObjCommand(interp, "torch::matrix_norm", TensorMatrixNorm_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::matrixNorm", TensorMatrixNorm_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::vector_norm", TensorVectorNorm_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::vectorNorm", TensorVectorNorm_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lstsq", TensorLstsq_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::leastSquares", TensorLstsq_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::solve_triangular", TensorSolveTriangular_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::solveTriangular", TensorSolveTriangular_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::cholesky_solve", TensorCholeskySolve_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::choleskySolve", TensorCholeskySolve_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lu_solve", TensorLUSolve_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::luSolve", TensorLUSolve_Cmd, NULL, NULL);  // camelCase alias

        // Register loss functions
        Tcl_CreateObjCommand(interp, "torch::mse_loss", MSELoss_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::mseLoss", MSELoss_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::cross_entropy_loss", CrossEntropyLoss_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::crossEntropyLoss", CrossEntropyLoss_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::nll_loss", NLLLoss_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::nllLoss", NLLLoss_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::bce_loss", BCELoss_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::bceLoss", BCELoss_Cmd, NULL, NULL);

        // Register learning rate schedulers
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_step", LRSchedulerStep_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::lrSchedulerStep", LRSchedulerStep_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_exponential", LRSchedulerExponential_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::lrSchedulerExponential", LRSchedulerExponential_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_cosine", LRSchedulerCosine_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::lrSchedulerCosine", LRSchedulerCosine_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::lr_scheduler_step_update", LRSchedulerStepUpdate_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::lrSchedulerStepUpdate", LRSchedulerStepUpdate_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::get_lr", GetLR_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::getLr", GetLR_Cmd, NULL, NULL);  // camelCase alias

        // Register advanced layer commands
        Tcl_CreateObjCommand(interp, "torch::layer_norm", LayerNorm_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::layerNorm", LayerNorm_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::group_norm", GroupNorm_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::groupNorm", GroupNorm_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::conv_transpose_2d", TensorConvTranspose2d_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::convTranspose2d", TensorConvTranspose2d_Cmd, NULL, NULL);  // camelCase alias

        // Register advanced tensor operation commands
        Tcl_CreateObjCommand(interp, "torch::tensor_var", TensorVar_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorVar", TensorVar_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_std", TensorStd_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorStd", TensorStd_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_is_cuda", TensorIsCuda_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorIsCuda", TensorIsCuda_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_is_contiguous", TensorIsContiguous_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorIsContiguous", TensorIsContiguous_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_contiguous", TensorContiguous_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorContiguous", TensorContiguous_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_where", TensorWhere_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorWhere", TensorWhere_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_expand", TensorExpand_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorExpand", TensorExpand_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_repeat", TensorRepeat_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorRepeat", TensorRepeat_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_index_select", TensorIndexSelect_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorIndexSelect", TensorIndexSelect_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_median", TensorMedian_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorMedian", TensorMedian_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_quantile", TensorQuantile_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorQuantile", TensorQuantile_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_mode", TensorMode_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorMode", TensorMode_Cmd, NULL, NULL);  // camelCase alias

        // Register AMP (Automatic Mixed Precision) commands
        Tcl_CreateObjCommand(interp, "torch::autocast_enable", Torch_AutocastEnable_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::autocastEnable", Torch_AutocastEnable_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::autocast_disable", Torch_AutocastDisable_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::autocastDisable", Torch_AutocastDisable_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::autocast_is_enabled", Torch_AutocastIsEnabled_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::autocastIsEnabled", Torch_AutocastIsEnabled_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::autocast_set_dtype", Torch_AutocastSetDtype_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::autocastSetDtype", Torch_AutocastSetDtype_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::grad_scaler_new", Torch_GradScalerNew_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::gradScalerNew", Torch_GradScalerNew_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::grad_scaler_scale", Torch_GradScalerScale_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::gradScalerScale", Torch_GradScalerScale_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::grad_scaler_step", Torch_GradScalerStep_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::gradScalerStep", Torch_GradScalerStep_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::grad_scaler_update", Torch_GradScalerUpdate_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::gradScalerUpdate", Torch_GradScalerUpdate_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::grad_scaler_get_scale", Torch_GradScalerGetScale_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::gradScalerGetScale", Torch_GradScalerGetScale_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_masked_fill", Torch_TensorMaskedFill_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorMaskedFill", Torch_TensorMaskedFill_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_clamp", Torch_TensorClamp_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::tensorClamp", Torch_TensorClamp_Cmd, NULL, NULL);  // camelCase alias

        // Register advanced tensor operation commands
        Tcl_CreateObjCommand(interp, "torch::tensor_slice", Torch_TensorSlice_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorSlice", Torch_TensorSlice_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_advanced_index", Torch_TensorAdvancedIndex_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::tensorAdvancedIndex", Torch_TensorAdvancedIndex_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::sparse_tensor_create", Torch_SparseTensorCreate_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::sparseTensorCreate", Torch_SparseTensorCreate_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::sparse_to_dense", TensorSparseToDense_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::sparseToDense", TensorSparseToDense_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::model_summary", Torch_ModelSummary_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::modelSummary", Torch_ModelSummary_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::count_parameters", Torch_CountParameters_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::countParameters", Torch_CountParameters_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::all_reduce", Torch_AllReduce_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::allReduce", Torch_AllReduce_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::broadcast", Torch_Broadcast_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensor_norm", Torch_TensorNorm_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorNorm", Torch_TensorNorm_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_normalize", Torch_TensorNormalize_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorNormalize", Torch_TensorNormalize_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_unique", Torch_TensorUnique_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorUnique", Torch_TensorUnique_Cmd, NULL, NULL);  // camelCase alias

        // Register advanced model checkpointing commands
        Tcl_CreateObjCommand(interp, "torch::save_checkpoint", Torch_SaveCheckpoint_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::saveCheckpoint", Torch_SaveCheckpoint_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::load_checkpoint", Torch_LoadCheckpoint_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::loadCheckpoint", Torch_LoadCheckpoint_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::get_checkpoint_info", Torch_GetCheckpointInfo_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::getCheckpointInfo", Torch_GetCheckpointInfo_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::save_state_dict", Torch_SaveStateDict_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::saveStateDict", Torch_SaveStateDict_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::scaled_dot_product_attention", ScaledDotProductAttention_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::scaledDotProductAttention", ScaledDotProductAttention_Cmd, NULL, NULL);  // camelCase alias

        // Register real distributed training commands
        Tcl_CreateObjCommand(interp, "torch::distributed_init", Torch_DistributedInit_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::distributedInit", Torch_DistributedInit_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::distributed_all_reduce", Torch_RealAllReduce_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::distributedAllReduce", Torch_RealAllReduce_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::distributed_broadcast", Torch_RealBroadcast_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::distributedBroadcast", Torch_RealBroadcast_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::distributed_barrier", Torch_DistributedBarrier_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::distributedBarrier", Torch_DistributedBarrier_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::get_rank", Torch_GetRank_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::getRank", Torch_GetRank_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::get_world_size", Torch_GetWorldSize_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::getWorldSize", Torch_GetWorldSize_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::is_distributed", Torch_IsDistributed_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::isDistributed", Torch_IsDistributed_Cmd, NULL, NULL);  // camelCase alias

        // ============================================================================
        // REGISTER SPARSE TENSOR OPERATIONS - BATCH IMPLEMENTATION OF 13 OPERATIONS
        // ============================================================================
        Tcl_CreateObjCommand(interp, "torch::sparse_coo_tensor", TensorSparseCOO_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::sparseCooTensor", TensorSparseCOO_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::sparse_csr_tensor", TensorSparseCSR_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::sparseCsrTensor", TensorSparseCSR_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::sparse_csc_tensor", TensorSparseCSC_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::sparseCscTensor", TensorSparseCSC_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::sparse_to_dense", TensorSparseToDense_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::sparseToDense", TensorSparseToDense_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::sparse_add", TensorSparseAdd_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::sparseAdd", TensorSparseAdd_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::sparse_mm", TensorSparseMM_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::sparseMm", TensorSparseMM_Cmd, NULL, NULL);  // camelCase alias
            Tcl_CreateObjCommand(interp, "torch::sparse_sum", TensorSparseSum_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::sparseSum", TensorSparseSum_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::sparse_softmax", TensorSparseSoftmax_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::sparseSoftmax", TensorSparseSoftmax_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::sparse_log_softmax", TensorSparseLogSoftmax_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::sparseLogSoftmax", TensorSparseLogSoftmax_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::sparse_mask", TensorSparseMask_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::sparseMask", TensorSparseMask_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::sparse_transpose", TensorSparseTranspose_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::sparseTranspose", TensorSparseTranspose_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::sparse_coalesce", TensorSparseCoalesce_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::sparseCoalesce", TensorSparseCoalesce_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::sparse_reshape", TensorSparseReshape_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::sparseReshape", TensorSparseReshape_Cmd, NULL, NULL);  // camelCase alias

        // ============================================================================
        // REGISTER QUANTIZATION OPERATIONS - BATCH IMPLEMENTATION OF 20 OPERATIONS
        // ============================================================================
        Tcl_CreateObjCommand(interp, "torch::quantize_per_tensor", TensorQuantizePerTensor_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::quantizePerTensor", TensorQuantizePerTensor_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::quantize_per_channel", TensorQuantizePerChannel_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::quantizePerChannel", TensorQuantizePerChannel_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::dequantize", TensorDequantize_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::deQuantize", TensorDequantize_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::fake_quantize_per_tensor", TensorFakeQuantizePerTensor_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::fakeQuantizePerTensor", TensorFakeQuantizePerTensor_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::fake_quantize_per_channel", TensorFakeQuantizePerChannel_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::fakeQuantizePerChannel", TensorFakeQuantizePerChannel_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::int_repr", TensorIntRepr_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::intRepr", TensorIntRepr_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::q_scale", TensorQScale_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::qScale", TensorQScale_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::q_zero_point", TensorQZeroPoint_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::qZeroPoint", TensorQZeroPoint_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::q_per_channel_scales", TensorQPerChannelScales_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::qPerChannelScales", TensorQPerChannelScales_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::q_per_channel_zero_points", TensorQPerChannelZeroPoints_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::qPerChannelZeroPoints", TensorQPerChannelZeroPoints_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::q_per_channel_axis", TensorQPerChannelAxis_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::qPerChannelAxis", TensorQPerChannelAxis_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::quantized_add", TensorQuantizedAdd_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::quantizedAdd", TensorQuantizedAdd_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::quantized_mul", TensorQuantizedMul_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::quantizedMul", TensorQuantizedMul_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::quantized_relu", TensorQuantizedRelu_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::quantizedRelu", TensorQuantizedRelu_Cmd, NULL, NULL);  // camelCase alias

        // ============================================================================
        // REGISTER RANDOM NUMBER GENERATION OPERATIONS - BATCH IMPLEMENTATION OF 12 OPERATIONS
        // ============================================================================
        Tcl_CreateObjCommand(interp, "torch::manual_seed", TensorManualSeed_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::manualSeed", TensorManualSeed_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::initial_seed", TensorInitialSeed_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::initialSeed", TensorInitialSeed_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::seed", TensorSeed_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::get_rng_state", TensorGetRngState_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::getRngState", TensorGetRngState_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::set_rng_state", TensorSetRngState_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::setRngState", TensorSetRngState_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::bernoulli", TensorBernoulli_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::multinomial", TensorMultinomial_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::normal", TensorNormal_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Normal", TensorNormal_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::uniform", TensorUniform_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::exponential", TensorExponential_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::gamma", TensorGamma_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::poisson", TensorPoisson_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Poisson", TensorPoisson_Cmd, NULL, NULL);  // camelCase alias

        // ============================================================================
        // REGISTER ADVANCED TENSOR OPERATIONS - BATCH IMPLEMENTATION OF 13 OPERATIONS
        // ============================================================================
        Tcl_CreateObjCommand(interp, "torch::block_diag", TensorBlockDiag_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::blockDiag", TensorBlockDiag_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::broadcast_shapes", TensorBroadcastShapes_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::broadcastShapes", TensorBroadcastShapes_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::squeeze_multiple", TensorSqueezeMultiple_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::squeezeMultiple", TensorSqueezeMultiple_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::unsqueeze_multiple", TensorUnsqueezeMultiple_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::unsqueezeMultiple", TensorUnsqueezeMultiple_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::tensor_split", TensorTensorSplit_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorSplit", TensorTensorSplit_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::hsplit", TensorHSplit_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::hSplit", TensorHSplit_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::vsplit", TensorVSplit_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::vSplit", TensorVSplit_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::dsplit", TensorDSplit_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::column_stack", TensorColumnStack_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::columnStack", TensorColumnStack_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::row_stack", TensorRowStack_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::rowStack", TensorRowStack_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::dstack", TensorDStack_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::dStack", TensorDStack_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::hstack", TensorHStack_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::hStack", TensorHStack_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::vstack", TensorVStack_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::vStack", TensorVStack_Cmd, NULL, NULL);  // camelCase alias

        // ============================================================================
        // REGISTER AUTOMATIC DIFFERENTIATION - BATCH IMPLEMENTATION OF 13 OPERATIONS
        // ============================================================================
        Tcl_CreateObjCommand(interp, "torch::grad", TensorGrad_Cmd, NULL, NULL);
        // Note: torch::grad is already camelCase, no alias needed
        Tcl_CreateObjCommand(interp, "torch::jacobian", TensorJacobian_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::Jacobian", TensorJacobian_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::hessian", TensorHessian_Cmd, NULL, NULL); // supports both snake_case and camelCase (same name)
        Tcl_CreateObjCommand(interp, "torch::vjp", TensorVJP_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::vectorJacobianProduct", TensorVJP_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::jvp", TensorJVP_Cmd, NULL, NULL);
        // Note: torch::jvp is already camelCase, no alias needed
        Tcl_CreateObjCommand(interp, "torch::functional_call", TensorFunctionalCall_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::functionalCall", TensorFunctionalCall_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::vmap", TensorVMap_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::vectorMap", TensorVMap_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::grad_check", TensorGradCheck_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::gradCheck", TensorGradCheck_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::grad_check_finite_diff", TensorGradCheckFiniteDiff_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::gradCheckFiniteDiff", TensorGradCheckFiniteDiff_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::enable_grad", TensorEnableGrad_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::enableGrad", TensorEnableGrad_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::no_grad", TensorNoGrad_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::noGrad", TensorNoGrad_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::set_grad_enabled", TensorSetGradEnabled_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::setGradEnabled", TensorSetGradEnabled_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::is_grad_enabled", TensorIsGradEnabled_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::isGradEnabled", TensorIsGradEnabled_Cmd, NULL, NULL);  // camelCase alias

        // ============================================================================
        // REGISTER MEMORY AND PERFORMANCE - BATCH IMPLEMENTATION OF 11 OPERATIONS
        // ============================================================================
        Tcl_CreateObjCommand(interp, "torch::memory_stats", TensorMemoryStats_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::memoryStats", TensorMemoryStats_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::memory_summary", TensorMemorySummary_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::memorySummary", TensorMemorySummary_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::memory_snapshot", TensorMemorySnapshot_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::memorySnapshot", TensorMemorySnapshot_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::empty_cache", TensorEmptyCache_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::emptyCache", TensorEmptyCache_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::synchronize", TensorSynchronize_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::synchronize", TensorSynchronize_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::profiler_start", TensorProfilerStart_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::profilerStart", TensorProfilerStart_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::profiler_stop", TensorProfilerStop_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::profilerStop", TensorProfilerStop_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::benchmark", TensorBenchmark_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::set_flush_denormal", TensorSetFlushDenormal_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::setFlushDenormal", TensorSetFlushDenormal_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::get_num_threads", TensorGetNumThreads_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::getNumThreads", TensorGetNumThreads_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::set_num_threads", TensorSetNumThreads_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::setNumThreads", TensorSetNumThreads_Cmd, NULL, NULL);  // camelCase alias

        // ============================================================================
        // REGISTER ADVANCED SIGNAL PROCESSING - BATCH IMPLEMENTATION OF 13 OPERATIONS
        // ============================================================================
        Tcl_CreateObjCommand(interp, "torch::fftshift", TensorFFTShift_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::fftShift", TensorFFTShift_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::ifftshift", TensorIFFTShift_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::ifftShift", TensorIFFTShift_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::hilbert", TensorHilbert_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::bartlett_window", TensorBartlettWindow_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::bartlettWindow", TensorBartlettWindow_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::blackman_window", TensorBlackmanWindow_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::blackmanWindow", TensorBlackmanWindow_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::hamming_window", TensorHammingWindow_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::hammingWindow", TensorHammingWindow_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::hann_window", TensorHannWindow_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::hannWindow", TensorHannWindow_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::kaiser_window", TensorKaiserWindow_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::kaiserWindow", TensorKaiserWindow_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::spectrogram", TensorSpectrogram_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::melscale_fbanks", TensorMelscaleFbanks_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::melscaleFbanks", TensorMelscaleFbanks_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::mfcc", TensorMFCC_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::mfcc", TensorMFCC_Cmd, NULL, NULL);  // camelCase alias (same name, no change needed)
        Tcl_CreateObjCommand(interp, "torch::pitch_shift", TensorPitchShift_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::pitchShift", TensorPitchShift_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::time_stretch", TensorTimeStretch_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::timeStretch", TensorTimeStretch_Cmd, NULL, NULL);  // camelCase alias

        // ============================================================================
        // REGISTER DISTRIBUTED OPERATIONS - FINAL IMPLEMENTATION OF 10 OPERATIONS
        // ============================================================================
        Tcl_CreateObjCommand(interp, "torch::distributed_gather", TensorDistributedGather_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::distributedGather", TensorDistributedGather_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::distributed_scatter", TensorDistributedScatter_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::distributedScatter", TensorDistributedScatter_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::distributed_reduce_scatter", TensorDistributedReduceScatter_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::distributedReduceScatter", TensorDistributedReduceScatter_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::distributed_all_to_all", TensorDistributedAllToAll_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::distributedAllToAll", TensorDistributedAllToAll_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::distributed_send", TensorDistributedSend_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::distributedSend", TensorDistributedSend_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::distributed_recv", TensorDistributedRecv_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::distributedRecv", TensorDistributedRecv_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::distributed_isend", TensorDistributedISend_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::distributedIsend", TensorDistributedISend_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::distributed_irecv", TensorDistributedIRecv_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::distributedIrecv", TensorDistributedIRecv_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::distributed_wait", TensorDistributedWait_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::distributedWait", TensorDistributedWait_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::distributed_test", TensorDistributedTest_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "torch::distributedTest", TensorDistributedTest_Cmd, NULL, NULL);

        // Short convenience aliases matching common PyTorch naming conventions and used by test suites
        Tcl_CreateObjCommand(interp, "torch::randn", TensorRandn_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::rand", TensorRand_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::meanDim", TensorMeanDim_Cmd, NULL, NULL);  // camelCase alias

        Tcl_CreateObjCommand(interp, "torch::save_state", SaveState_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::saveState", SaveState_Cmd, NULL, NULL);  // camelCase alias

        Tcl_CreateObjCommand(interp, "torch::save_state_dict", Torch_SaveStateDict_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::saveStateDict", Torch_SaveStateDict_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::load_state_dict", Torch_LoadStateDict_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::loadStateDict", Torch_LoadStateDict_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::freeze_model", Torch_FreezeModel_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::freezeModel", Torch_FreezeModel_Cmd, NULL, NULL);  // camelCase alias
        Tcl_CreateObjCommand(interp, "torch::unfreeze_model", Torch_UnfreezeModel_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::unfreezeModel", Torch_UnfreezeModel_Cmd, NULL, NULL);  // camelCase alias

        Tcl_CreateObjCommand(interp, "torch::tensor_size", TensorSize_Cmd, NULL, NULL);
        Tcl_CreateObjCommand(interp, "torch::tensorSize", TensorSize_Cmd, NULL, NULL);  // camelCase alias

        return TCL_OK;
    }
}
