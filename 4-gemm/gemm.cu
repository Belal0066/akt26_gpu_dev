
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Custom batched GEMM
//
// C[b] = A[b] * B[b]
//
// Matrices are N x N, column-major.
////////////////////////////////////////////////////////////////////////////////
//
// Parallelization (n <= 32):
//  - threadIdx.y = column of C; threadIdx.x = a group of rows: each thread computes GEMM_R elements
//    of its column (rows x, x + blockDim.x, x + 2*blockDim.x, ...), so one value of B loaded from
//    shared memory is reused GEMM_R times from a register (register blocking);
//  - several matrices per block when n is small (threadIdx.z = matrix inside the block), so that
//    a block keeps about GEMM_THREADS threads busy even for tiny matrices;
//  - each block first copies its A and B matrices into shared memory with coalesced loads
//    (consecutive threads read consecutive addresses), then every thread reads the values it
//    needs from shared memory instead of global memory (each element is reused n times).
// Shared memory per block: 2 * n * n * (matrices per block) floats.
#ifndef GEMM_THREADS
#define GEMM_THREADS 128   // target number of threads per block
#endif

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                         cudaGetErrorString(err_), __FILE__, __LINE__);           \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t st_ = (call);                                              \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                       \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n",                     \
                         (int)st_, __FILE__, __LINE__);                           \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

#ifndef GEMM_R
#define GEMM_R 4           // rows of C computed by one thread (register blocking)
#endif

__global__ void batch_gemm_kernel(int n, int batch_size,
                                    const float* A,
                                    const float* B,
                                    float* C)
{
    extern __shared__ float smem[];

    const int nn  = n * n;
    const int tpm = blockDim.x * blockDim.y;                     // threads per matrix
    const int lin = threadIdx.y * blockDim.x + threadIdx.x;      // thread index inside its matrix
    const int col = threadIdx.y;
    const int b   = blockIdx.x * blockDim.z + threadIdx.z;       // matrix index in the batch

    float* sA = smem + (size_t)threadIdx.z * 2 * nn;
    float* sB = sA + nn;

    if (b < batch_size) {
        const size_t base = (size_t)b * nn;
        for (int e = lin; e < nn; e += tpm) {                    // coalesced copy into shared memory
            sA[e] = A[base + e];
            sB[e] = B[base + e];
        }
    }
    __syncthreads();

    if (b < batch_size) {
        float acc[GEMM_R];
        int   rows[GEMM_R];
        #pragma unroll
        for (int q = 0; q < GEMM_R; ++q) {
            acc[q]  = 0.0f;
            rows[q] = min(threadIdx.x + q * blockDim.x, n - 1);  // clamp: rows beyond n are computed but never stored
        }
        for (int k = 0; k < n; ++k) {
            const float bk = sB[col * n + k];                    // B(k,col): one load reused GEMM_R times
            #pragma unroll
            for (int q = 0; q < GEMM_R; ++q) {
                acc[q] += sA[k * n + rows[q]] * bk;              // A(row,k)
            }
        }
        #pragma unroll
        for (int q = 0; q < GEMM_R; ++q) {
            const int row = threadIdx.x + q * blockDim.x;
            if (row < n) C[(size_t)b * nn + col * n + row] = acc[q];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// CPU reference implementation
void batched_gemm_cpu(int n, int batch_size,
                      const float* A,
                      const float* B,
                      float* C)
{
    const size_t matrix_size = (size_t)n * n;

    for (int b = 0; b < batch_size; ++b) {

        const size_t base = (size_t)b * matrix_size;

        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < n; ++row) {

                float sum = 0.0f;

                for (int k = 0; k < n; ++k) {
                    sum += A[base + (size_t)k * n + row] *
                           B[base + (size_t)col * n + k];
                }

                C[base + (size_t)col * n + row] = sum;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Forward error for one matrix
//
// ||Cgpu-Cref||_F / (N ||Cref||_F)
// ------------------------------------------------------------
double forward_error(int n,
                     const float* result,
                     const float* reference)
{
    double diff2 = 0.0;
    double ref2  = 0.0;

    const size_t size = (size_t)n * n;

    for (size_t i = 0; i < size; ++i) {
        const double diff =
            (double)result[i] - (double)reference[i];

        diff2 += diff * diff;
        ref2  += (double)reference[i] * reference[i];
    }

    const double diff_norm = std::sqrt(diff2);
    const double ref_norm  = std::sqrt(ref2);

    if (ref_norm > 0.0)
        return diff_norm / (n * ref_norm);

    return (diff_norm == 0.0) ? 0.0 : INFINITY;
}

////////////////////////////////////////////////////////////////////////////////
// Maximum forward error over the batch
double max_batch_forward_error(int n,
                               int batch_size,
                               const float* result,
                               const float* reference)
{
    const size_t matrix_size = (size_t)n * n;

    double max_error = 0.0;

    for (int b = 0; b < batch_size; ++b) {

        const size_t base = (size_t)b * matrix_size;

        const double e =
            forward_error(n,
                          result + base,
                          reference + base);

        max_error = std::max(max_error, e);
    }

    return max_error;
}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    const int n          = (argc > 1) ? std::atoi(argv[1]) : 16;
    const int batch_size = (argc > 2) ? std::atoi(argv[2]) : 1000;
    const int repeats = 100;

    if (n <= 0 || n > 32 || batch_size <= 0) {

        std::fprintf(stderr, "Usage: %s matrix_size(1..32) batch_size\n", argv[0]);

        return EXIT_FAILURE;
    }

    const size_t matrix_size = (size_t)n * n;
    const size_t total = matrix_size * batch_size;
    const size_t bytes = total * sizeof(float);

    // --------------------------------------------------------
    // Initialize input matrices
    // --------------------------------------------------------
    std::vector<float> h_A(total);
    std::vector<float> h_B(total);
    std::vector<float> h_ref(total);
    std::vector<float> h_custom(total);
    std::vector<float> h_cublas(total);

    for (size_t i = 0; i < total; ++i) {
        h_A[i] = std::sin(0.013f * (float)(i + 1));
        h_B[i] = std::cos(0.017f * (float)(i + 1));
    }

    // ref. CPU GEMM
    batched_gemm_cpu(n, batch_size, h_A.data(), h_B.data(), h_ref.data());

    // --------------------------------------------------------
    // Allocate GPU memory
    // --------------------------------------------------------
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_C_cublas = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));            // output of the custom kernel
    CUDA_CHECK(cudaMalloc(&d_C_cublas, bytes));     // output of cuBLAS (separate, so one cannot hide the other)

    // --------------------------------------------------------
    // Copy A and B to GPU
    // --------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // --------------------------------------------------------
    // GPU information
    // --------------------------------------------------------
    int device = 0;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::printf("GPU: %s (compute capability %d.%d)\n",
                prop.name, prop.major, prop.minor);

    std::printf("Matrix size: %d x %d, batch size: %d\n",
                n, n, batch_size);

    std::printf("FP32, column-major matrices, repeats: %d\n\n",
                repeats);

    // --------------------------------------------------------
    // CUDA events
    // --------------------------------------------------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --------------------------------------------------------
    // Custom kernel configuration
    // --------------------------------------------------------
    // GEMM_R rows of one column per thread, and as many matrices per block as fit in GEMM_THREADS threads
    // (blockDim.z is limited to 64 by the hardware, which matters for tiny n)
    const int rows_threads   = (n + GEMM_R - 1) / GEMM_R;          // threads along the rows
    const int mats_per_block = std::max(1, std::min({GEMM_THREADS / (rows_threads * n), 64, batch_size}));
    dim3 block(rows_threads, n, mats_per_block);
    dim3 grid((batch_size + mats_per_block - 1) / mats_per_block, 1, 1);
    const size_t shmem_bytes = (size_t)2 * n * n * mats_per_block * sizeof(float);

    std::printf("Launch configuration: grid=(%d,1,1), block=(%d,%d,%d), shared memory=%zu bytes\n\n",
                grid.x, block.x, block.y, block.z, shmem_bytes);

    // --------------------------------------------------------
    // Warmup run
    // --------------------------------------------------------
    batch_gemm_kernel<<<grid, block, shmem_bytes>>>(n, batch_size, d_A, d_B, d_C);
    CUDA_CHECK(cudaGetLastError());

    cudaDeviceSynchronize();

    CUDA_CHECK(cudaEventRecord(start));

    for (int r = 0; r < repeats; ++r) {
        batch_gemm_kernel<<<grid, block, shmem_bytes>>>(n, batch_size, d_A, d_B, d_C);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float custom_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&custom_ms, start, stop));

    const double custom_us = custom_ms * 1000.0 / repeats;


    CUDA_CHECK(cudaMemcpy(h_custom.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // --------------------------------------------------------
    // cuBLAS strided batched GEMM
    // --------------------------------------------------------

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f, beta = 0.0f;
    const long long stride = (long long)n * n;     // A[b+1] starts n*n floats after A[b]

    // C[b] = 1 * A[b] * B[b] + 0 * C[b], all matrices n x n with leading dimension n
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                           &alpha, d_A, n, stride, d_B, n, stride,
                                           &beta, d_C_cublas, n, stride, batch_size));

    cudaDeviceSynchronize();

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r) {

        cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                  &alpha, d_A, n, stride, d_B, n, stride,
                                  &beta, d_C_cublas, n, stride, batch_size);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cublas_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));

    const double cublas_us = cublas_ms * 1000.0 / repeats;

    CUDA_CHECK(cudaMemcpy(h_cublas.data(), d_C_cublas, bytes, cudaMemcpyDeviceToHost));

    // --------------------------------------------------------
    // Performance and forward error
    // --------------------------------------------------------
    const double flops = 2.0 * n * n * n * batch_size;

    const double custom_gflops = flops / (custom_us * 1.0e3);
    const double cublas_gflops = flops / (cublas_us * 1.0e3);

    const double custom_error = max_batch_forward_error(
                                    n, batch_size,
                                    h_custom.data(), h_ref.data());

    const double cublas_error = max_batch_forward_error(
                                    n, batch_size,
                                    h_cublas.data(), h_ref.data());

    std::printf("%-24s %12s %12s %20s\n",
                "Implementation",
                "Time (us)",
                "GFLOP/s",
                "Max batch fwd error");

    std::printf("-----------------------------------------------------------------------\n");

    std::printf("%-24s %12.3f %12.3f %20.6e\n",
                "Custom kernel",
                custom_us,
                custom_gflops,
                custom_error);

    std::printf("%-24s %12.3f %12.3f %20.6e\n",
                "cuBLAS Strided Batched",
                cublas_us,
                cublas_gflops,
                cublas_error);

    const double tol = 1e-6;
    if( double(custom_error) <= tol && double(cublas_error) <= tol ) {
        std::printf("\nTEST PASS\n");
        std::printf("Speedup (cuBLAS time / custom time): %.3fx\n", cublas_us / custom_us);
    }
    else {
        std::printf("\nTEST FAIL\n");
        std::printf("Speedup is not calculated\n");
    }

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_cublas));

    return 0;
}
