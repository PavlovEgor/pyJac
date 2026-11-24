// CUDA driver used by benchmark_methane_gpu.py.
#include "jacob.cuh"
#include "jacob_block.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

constexpr double pressure = 101325.0;
constexpr int entries = NSP * NSP;

// A fixed direction makes the consumer observable while avoiding another input array.
__device__ double apply_row(const double* jac, int row, int stride) {
    double sum = 0.0;
    for (int col = 0; col < NSP; ++col)
        sum += jac[(row + col * NSP) * stride] / (1.0 + col);
    return sum;
}

template<bool apply>
__global__ void scalar_kernel(int count, mechanism_memory* mem) {
    const int id = T_ID;
    if (id >= count) return;
    eval_jacob(0., pressure, mem->y, mem->jac, mem);
    if (apply)
        for (int row = 0; row < NSP; ++row)
            mem->dy[INDEX(row)] = apply_row(mem->jac + id, row, GRID_DIM);
}

template<bool apply>
__global__ void block_kernel(const double* states, double* output) {
    __shared__ jacob_block_workspace work;
    __shared__ double jac[entries];
    const int id = blockIdx.x;
    eval_jacob_block(pressure, states + id * NSP, jac, work);
    if (apply) {
        for (int row = threadIdx.x; row < NSP; row += blockDim.x)
            output[id * NSP + row] = apply_row(jac, row, 1);
    } else {
        for (int i = threadIdx.x; i < entries; i += blockDim.x)
            output[id * entries + i] = jac[i];
    }
}

template<class Launch>
float sample(Launch launch, int iterations, cudaEvent_t start, cudaEvent_t end) {
    cudaErrorCheck(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) launch();
    cudaErrorCheck(cudaGetLastError());
    cudaErrorCheck(cudaEventRecord(end));
    cudaErrorCheck(cudaEventSynchronize(end));
    float ms;
    cudaErrorCheck(cudaEventElapsedTime(&ms, start, end));
    return ms / iterations;
}

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return 0.5f * (v[(v.size()-1)/2] + v[v.size()/2]);
}

template<bool apply>
void compare(int n, int padded, int threads, int repeats, int iterations,
             const mechanism_memory& host, mechanism_memory* mem,
             const double* states, double* output) {
    auto scalar = [&]() {
        scalar_kernel<apply><<<padded / TARGET_BLOCK_SIZE, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(n, mem);
    };
    auto block = [&]() { block_kernel<apply><<<n, threads>>>(states, output); };

    scalar();
    cudaErrorCheck(cudaGetLastError());
    block();
    cudaErrorCheck(cudaGetLastError());
    const int width = apply ? NSP : entries;
    std::vector<double> reference(size_t(padded) * width), actual(size_t(n) * width);
    cudaErrorCheck(cudaMemcpy(reference.data(), apply ? host.dy : host.jac,
                              reference.size()*sizeof(double), cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaMemcpy(actual.data(), output, actual.size()*sizeof(double), cudaMemcpyDeviceToHost));
    for (int id = 0; id < n; ++id) {
        for (int i = 0; i < width; ++i) {
            double a = actual[id * width + i], b = reference[i * padded + id];
            if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a-b) > 1e-8 + 1e-8*std::abs(b)) {
                std::fprintf(stderr, "%s mismatch: system=%d entry=%d scalar=%.17g block=%.17g\n",
                             apply ? "Jv" : "J", id, i, b, a);
                std::exit(2);
            }
        }
    }

    cudaEvent_t start, end;
    cudaErrorCheck(cudaEventCreate(&start));
    cudaErrorCheck(cudaEventCreate(&end));
    std::vector<float> scalar_ms, block_ms;
    for (int r = 0; r < repeats; ++r) {
        if (r % 2 == 0) scalar_ms.push_back(sample(scalar, iterations, start, end));
        block_ms.push_back(sample(block, iterations, start, end));
        if (r % 2 == 1) scalar_ms.push_back(sample(scalar, iterations, start, end));
    }
    cudaErrorCheck(cudaEventDestroy(start));
    cudaErrorCheck(cudaEventDestroy(end));
    const float s = median(scalar_ms), b = median(block_ms);
    std::printf("%-7s %8d %8d %12.4f %12.4f %13.3f\n",
                apply ? "Jv" : "J", n, threads, s, b, b/s);
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    if (argc < 5) return 1;
    const int n = std::atoi(argv[1]), repeats = std::atoi(argv[2]), iterations = std::atoi(argv[3]);
    const int padded = ((n + TARGET_BLOCK_SIZE - 1) / TARGET_BLOCK_SIZE) * TARGET_BLOCK_SIZE;
    cudaDeviceProp prop;
    cudaErrorCheck(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU %s (sm_%d%d), scalar threads/block %d, cooperative shared bytes %zu\n",
                prop.name, prop.major, prop.minor, TARGET_BLOCK_SIZE,
                sizeof(jacob_block_workspace) + entries * sizeof(double));

    std::vector<double> snapshots(32 * NSP);
    std::ifstream input("states.bin", std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(snapshots.data()), snapshots.size()*sizeof(double))) {
        std::fprintf(stderr, "Unable to read ignition snapshots\n");
        return 1;
    }
    std::vector<double> aos(n * NSP), soa(padded * NSP);
    for (int id = 0; id < n; ++id)
        for (int j = 0; j < NSP; ++j)
            soa[j * padded + id] = aos[id * NSP + j] = snapshots[(id % 32) * NSP + j];

    mechanism_memory storage{}, *host = &storage, *mem;
    initialize_gpu_memory(padded, &host, &mem);
    double *states, *output;
    cudaErrorCheck(cudaMalloc(&states, aos.size()*sizeof(double)));
    cudaErrorCheck(cudaMalloc(&output, size_t(n)*entries*sizeof(double)));
    cudaErrorCheck(cudaMemcpy(states, aos.data(), aos.size()*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(host->y, soa.data(), soa.size()*sizeof(double), cudaMemcpyHostToDevice));
    std::puts("output   systems  threads    scalar_ms     block_ms  block/scalar");
    for (int i = 4; i < argc; ++i) {
        int threads = std::atoi(argv[i]);
        compare<false>(n, padded, threads, repeats, iterations, storage, mem, states, output);
        compare<true>(n, padded, threads, repeats, iterations, storage, mem, states, output);
    }
    cudaErrorCheck(cudaFree(states));
    cudaErrorCheck(cudaFree(output));
    free_gpu_memory(&host, &mem);
}
