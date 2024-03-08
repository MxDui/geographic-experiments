#include <cuda_runtime.h>
#include <json.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>

using json = nlohmann::json;

// Kernel function to process GeoJSON data
__global__ void processGeoJSON(float* coordinates, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        printf("coordinate[%d] = %f\n", index, coordinates[index]);
    }
}

int main() {
    cudaError_t cudaStatus;

    // Read GeoJSON file
    std::ifstream file("./micro.geojson");
    json j;
    file >> j;

    // Extract coordinates from GeoJSON
    std::vector<float> coordinates;
    for (const auto& feature : j["features"]) {
        for (const auto& polygon : feature["geometry"]["coordinates"]) {
            for (const auto& point : polygon) {
                coordinates.push_back(point[0]);
                coordinates.push_back(point[1]);
            }
        }
    }
 
    // Allocate GPU memory
    float* d_coordinates;
    cudaStatus = cudaMalloc((void**)&d_coordinates, coordinates.size() * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed!" << std::endl;
        return 1;
    }

    // Copy data to GPU
    cudaStatus = cudaMemcpy(d_coordinates, coordinates.data(), coordinates.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed!" << std::endl;
        return 1;
    }

    // Launch kernel
    int blockSize = 128;
    int numBlocks = (coordinates.size() + blockSize - 1) / blockSize;

    auto start = std::chrono::high_resolution_clock::now();
    processGeoJSON<<<numBlocks, blockSize>>>(d_coordinates, coordinates.size());
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed with error " << cudaStatus << std::endl;
        return 1;
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Kernel execution time: " << elapsed.count() << " ms" << std::endl;

    // Free GPU memory
    cudaFree(d_coordinates);

    return 0;
}