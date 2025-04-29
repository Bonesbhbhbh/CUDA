#include <windows.h>
#include <iostream>
#include <string>
#include <cuda_runtime.h>

__device__ int num_dirs = 0;
__device__ int num_regular = 0;

__global__ void increment_dirs() {
    atomicAdd(&num_dirs, 1);
}

__global__ void increment_regular() {
    atomicAdd(&num_regular, 1);
}

void process_file(const std::string& file_path) {
    // File processing logic (same as before)
    std::cout << "Processing file: " << file_path << std::endl;
}

void process_directory(const std::string& dir_path) {
    increment_dirs<<<1, 1>>>();
    cudaDeviceSynchronize();

    WIN32_FIND_DATA find_data;
    HANDLE hFind = FindFirstFile((dir_path + "\\*").c_str(), &find_data);

    if (hFind == INVALID_HANDLE_VALUE) {
        std::cerr << "Error opening directory: " << dir_path << std::endl;
        return;
    }

    do {
        const std::string name = find_data.cFileName;
        if (name == "." || name == "..") {
            continue;
        }

        const std::string full_path = dir_path + "\\" + name;
        if (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            process_directory(full_path);
        } else {
            increment_regular<<<1, 1>>>();
            cudaDeviceSynchronize();
            process_file(full_path);
        }
    } while (FindNextFile(hFind, &find_data) != 0);

    FindClose(hFind);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path>" << std::endl;
        return 1;
    }

    const std::string root_path = argv[1];

    // Initialize device variables
    int zero = 0;
    cudaMemcpyToSymbol(num_dirs, &zero, sizeof(int));
    cudaMemcpyToSymbol(num_regular, &zero, sizeof(int));

    // Process the directory
    process_directory(root_path);

    // Retrieve results from device
    int host_num_dirs = 0;
    int host_num_regular = 0;
    cudaMemcpyFromSymbol(&host_num_dirs, num_dirs, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_num_regular, num_regular, sizeof(int), 0, cudaMemcpyDeviceToHost);

    std::cout << "Total directories: " << host_num_dirs << std::endl;
    std::cout << "Total regular files: " << host_num_regular << std::endl;

    return 0;
}