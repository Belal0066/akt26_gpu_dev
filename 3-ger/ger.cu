
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Custom rank-1 update kernel
// A <- A + alpha * x * y^T
// Column-major storage: A(i,j) = A[j*m + i]
//
// Parallelization: one thread per row i, and each thread updates GER_NJ consecutive columns j.
//  - threadIdx.x runs along the rows, so the threads of a warp touch consecutive addresses of
//    the same column (column-major): every load and store of A is coalesced.
//  - alpha * x[i] is computed once per thread and reused for its GER_NJ columns.
//  - y[j] is the same for all threads of a warp (a broadcast read).
//  - gridDim.y is limited to 65535, so when n needs more column blocks than that, each thread
//    also strides over the columns (j0 += gridDim.y * GER_NJ).
// The kernel is memory bound: each element of A is read once and written once.
#ifndef GER_TX
#define GER_TX 128   // threads per block (rows handled by one block)
#endif
#ifndef GER_NJ
#define GER_NJ 2     // columns handled by one thread
#endif

__global__ void ger_kernel(int m, int n, float alpha, const float* x, const float* y, float* A)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;    // row of A

    if (i >= m) return;                                       // last block may be partial

    const float ax = alpha * x[i];

    for (int j0 = blockIdx.y * GER_NJ; j0 < n; j0 += gridDim.y * GER_NJ) {
        const int jend = min(j0 + GER_NJ, n);                 // last column block may be partial
        for (int j = j0; j < jend; ++j) {
            A[(size_t)j * m + i] += ax * y[j];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// CPU reference implementation, FP32
void ger_cpu(int m, int n, float alpha, const float* x, const float* y, float* A)
{
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            A[(size_t)j * m + i] += alpha * x[i] * y[j];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// compute error norm
double compute_error_norm(int size,
                          const float* result,
                          const float* reference)
{
    double error = 0.0;
    double norm  = 0.0;

    for (int i = 0; i < size; ++i) {
        const double diff  = std::fabs((double)result[i] - (double)reference[i]);
        const double denom = std::fabs((double)reference[i]);

        error += diff * diff;
        norm  += denom * denom;
    }

    return sqrt(error) / sqrt(norm);
}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    const int m = (argc > 1) ? std::atoi(argv[1]) : 4000;
    const int n = (argc > 2) ? std::atoi(argv[2]) : 2000;
    const int repeats = (argc > 3) ? std::atoi(argv[3]) : 100;
    const float alpha = 1.5f;

    if (m <= 0 || n <= 0 || repeats <= 0) {
        std::fprintf(stderr, "Usage: %s [M > 0] [N > 0] [repeats > 0]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const size_t matrix_size = (size_t)m * n;
    const size_t matrix_bytes = matrix_size * sizeof(float);

    // --------------------------------------------------------
    // Initialize vectors and matrix on the CPU
    // --------------------------------------------------------
    std::vector<float> h_x(m), h_y(n);
    std::vector<float> h_A0(matrix_size);
    std::vector<float> h_ref(matrix_size);
    std::vector<float> h_custom(matrix_size);
    std::vector<float> h_cublas(matrix_size);

    for (int i = 0; i < m; ++i) {
        h_x[i] = std::sin(0.1f * (i + 1));
    }

    for (int j = 0; j < n; ++j) {
        h_y[j] = std::cos(0.2f * (j + 1));
    }

    for (size_t i = 0; i < matrix_size; ++i) {
        h_A0[i] = 0.1f * std::sin(0.1f * (float)(i + 1));
    }

    h_ref = h_A0;
    ger_cpu(m, n, alpha, h_x.data(), h_y.data(), h_ref.data());

    // --------------------------------------------------------
    // Allocate GPU memory
    // --------------------------------------------------------
    float *d_x = nullptr, *d_y = nullptr, *d_A = nullptr;

    cudaMalloc(&d_x, (size_t)m * sizeof(float));
    cudaMalloc(&d_y, (size_t)n * sizeof(float));
    cudaMalloc(&d_A, matrix_bytes);

    // copy input vectors from CPU -> GPU
    cudaMemcpy(d_x, h_x.data(), (size_t)m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice);

    std::printf("Matrix: %d x %d, alpha = %.2f, repeats = %d\n",
                m, n, alpha, repeats);
    std::printf("Matrix storage: column-major\n\n");

    // n1000: configure your kernel here
    // TODO: the configuration below is a default that you must change
    dim3 block(GER_TX, 1, 1);
    dim3 grid((m + GER_TX - 1) / GER_TX,      // blocks along the rows
              std::min((n + GER_NJ - 1) / GER_NJ, 65535),   // blocks along the columns (GER_NJ columns each, at most 65535)
              1);
    // end of TODO: kernel configuration

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --------------------------------------------------------
    // Custom kernel: warm-up and timing
    // --------------------------------------------------------
    cudaMemcpy(d_A, h_A0.data(), matrix_bytes, cudaMemcpyHostToDevice);

    ger_kernel<<<grid, block>>>(m, n, alpha, d_x, d_y, d_A);
    cudaDeviceSynchronize();

    // Restore the initial matrix before the timed repetitions.
    cudaMemcpy(d_A, h_A0.data(), matrix_bytes,
               cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    for (int r = 0; r < repeats; ++r) {
        ger_kernel<<<grid, block>>>(m, n, alpha, d_x, d_y, d_A);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float custom_ms = 0.0f;
    cudaEventElapsedTime(&custom_ms, start, stop);
    const double custom_us = custom_ms * 1000.0 / repeats;

    // Correctness run: restore initial A and update once.
    cudaMemcpy(d_A, h_A0.data(), matrix_bytes, cudaMemcpyHostToDevice);
    ger_kernel<<<grid, block>>>(m, n, alpha, d_x, d_y, d_A);
    cudaMemcpy(h_custom.data(), d_A, matrix_bytes, cudaMemcpyDeviceToHost);

    // --------------------------------------------------------
    // cuBLAS SGER: warm-up and timing
    // --------------------------------------------------------
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMemcpy(d_A, h_A0.data(), matrix_bytes, cudaMemcpyHostToDevice);

    cublasSger(handle, m, n, &alpha, d_x, 1, d_y, 1, d_A, m);
    cudaDeviceSynchronize();

    // Restore the initial matrix before the timed repetitions.
    cudaMemcpy(d_A, h_A0.data(), matrix_bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    for (int r = 0; r < repeats; ++r) {
        cublasSger(handle, m, n, &alpha, d_x, 1, d_y, 1, d_A, m);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cublas_ms = 0.0f;
    cudaEventElapsedTime(&cublas_ms, start, stop);
    const double cublas_us = cublas_ms * 1000.0 / repeats;

    // Correctness run: restore initial A and update once.
    cudaMemcpy(d_A, h_A0.data(), matrix_bytes, cudaMemcpyHostToDevice);
    cublasSger(handle, m, n, &alpha, d_x, 1, d_y, 1, d_A, m);
    cudaMemcpy(h_cublas.data(), d_A, matrix_bytes, cudaMemcpyDeviceToHost);

    // --------------------------------------------------------
    // Performance and error norm
    // Approximately 2 FLOPs per matrix element.
    // --------------------------------------------------------
    const double flops = 2.0 * m * n;

    const double custom_gflops = flops / (custom_us * 1.0e3);
    const double cublas_gflops = flops / (cublas_us * 1.0e3);

    const double custom_error = compute_error_norm( (int)matrix_size, h_custom.data(), h_ref.data());
    const double cublas_error = compute_error_norm( (int)matrix_size, h_cublas.data(), h_ref.data());

    std::printf("%-20s %12s %12s %20s\n",
                "Implementation", "Time (us)", "GFLOP/s",
                "Error Norm");
    std::printf("--------------------------------------------------------------------\n");

    std::printf("%-20s %12.3f %12.3f %20.6e\n",
                "Custom kernel", custom_us, custom_gflops,
                custom_error);
    std::printf("%-20s %12.3f %12.3f %20.6e\n",
                "cuBLAS SGER", cublas_us, cublas_gflops,
                cublas_error);

    const double tol = 1e-6;
    if( double(custom_error) <= tol ) {
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
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_A);

    return 0;
}
