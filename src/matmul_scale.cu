#include "megbrain/custom/custom.h"
#include "megbrain/custom/platform/custom_cuda.h"
#include <cuda.h>
#include <cuda_runtime.h>

/****************** My Kernels ******************/

// matmul_forward for Mat_mxk * Mat_k*n
template <typename T>
__global__ void matmul_forward_naive(const T *lhs, const T *rhs, T *res, size_t M,
                                     size_t K, size_t N, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    T acc = 0;
    for (int i = 0; i < K; ++i)
        acc += lhs[row * K + i] * rhs[i * N + col];
    res[row * N + col] = acc * scale;
}

// matmul_backward_lhs for Mat_mxk * Mat_k*n = Mat_mxn
// that is Mat_mxn * Mat_nxk
template <typename T>
__global__ void matmul_backward_lhs_naive(const T *rhs, const T *ograd, T *lhs_grad,
                                          size_t M, size_t K, size_t N, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    T acc = 0;
    for (int i = 0; i < N; ++i)
        acc += ograd[row * N + i] * rhs[col * N + i];
    lhs_grad[row * K + col] = acc / scale;
}

// matmul_backward_rhs for Mat_mxk * Mat_k*n = Mat_mxn
// that is Mat_kxm * Mat_mxn
template <typename T>
__global__ void matmul_backward_rhs_naive(const T *lhs, const T *ograd, T *rhs_grad,
                                          size_t M, size_t K, size_t N, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    T acc = 0;
    for (int i = 0; i < M; ++i)
        acc += lhs[i * K + row] * ograd[i * N + col];
    rhs_grad[row * N + col] = acc / scale;
}

/****************** wrap kernel by custom op ******************/

// using custom::Shape;
// using custom::Param;
// using custom::Tensor;
using namespace custom;

void forward_shape_infer(const std::vector<Shape> &inputs, const Param &params,
                         std::vector<Shape> &outputs) {
    outputs[0] = {inputs[0][0], inputs[1][1]};
}

void matmul_forward(const Tensor &lhs, const Tensor &rhs, Tensor &res, size_t M,
                    size_t K, size_t N, float scale) {}

void forward_compute(const std::vector<Tensor> &inputs, const Param &params,
                     std::vector<Tensor> &outputs) {
    const Tensor &lhs = inputs[0];
    const Tensor &rhs = inputs[1];
    const Tensor &res = outputs[0];
    size_t M = lhs.shape()[0], K = lhs.shape()[1], N = rhs.shape()[1];
    float scale = params["scale"].as<float>();

    // get input cuda stream, and launch kernel on this stream
    auto stream = get_cuda_stream(lhs.device());
    dim3 block(1, 1);
    dim3 grid(N / block.x, M / block.y);
    matmul_forward_naive<float><<<grid, block, 0, stream>>>(
        lhs.data<float>(), rhs.data<float>(), res.data<float>(), M, K, N, scale);
}

CUSTOM_OP_REG(MatMulScaleForward)
    .add_inputs(2)  // lhs, rhs
    .add_outputs(1) // output
    .add_param("scale", 1.0f)
    .set_shape_infer(forward_shape_infer)
    .set_compute("cuda", forward_compute);

void backward_shape_infer(const std::vector<Shape> &ograd_and_inputs,
                          const Param &params, std::vector<Shape> &outputs) {
    outputs[0] = ograd_and_inputs[1];
    outputs[1] = ograd_and_inputs[2];
}

void backward_compute(const std::vector<Tensor> &ograd_and_inputs, const Param &params,
                      std::vector<Tensor> &igrads) {
    const Tensor &ograd = ograd_and_inputs[0];
    const Tensor &lhs = ograd_and_inputs[1];
    const Tensor &rhs = ograd_and_inputs[2];
    Tensor &lhs_grad = igrads[0], &rhs_grad = igrads[1];

    size_t M = lhs.shape()[0], K = lhs.shape()[1], N = rhs.shape()[1];
    float scale = params["scale"].as<float>();

    // get input cuda stream, and launch kernel on this stream
    auto stream = get_cuda_stream(lhs.device());
    dim3 block(1, 1);
    dim3 grid_lhs(K / block.x, M / block.y);
    dim3 grid_rhs(N / block.x, K / block.y);
    matmul_backward_lhs_naive<float><<<grid_lhs, block, 0, stream>>>(
        rhs.data<float>(), ograd.data<float>(), lhs_grad.data<float>(), M, K, N, scale);
    matmul_backward_rhs_naive<float><<<grid_rhs, block, 0, stream>>>(
        lhs.data<float>(), ograd.data<float>(), rhs_grad.data<float>(), M, K, N, scale);
}

CUSTOM_OP_REG(MatMulScaleBackward)
    .add_inputs(3)  // ograd, lhs, rhs
    .add_outputs(2) // lhs_grad, rhs_grad
    .add_param("scale", 1.0f)
    .set_shape_infer(backward_shape_infer)
    .set_compute("cuda", backward_compute);
