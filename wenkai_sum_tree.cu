#include <stdio.h>
#include <sys/stat.h>
#include <stdbool.h>
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <cuda_runtime.h>
static int num_dirs, num_regular;
bool is_dir(const char* path) {
/*
   * Use the stat() function (try "man 2 stat") to determine if the file
   * referenced by path is a directory or not.  Call stat, and then use
   * S_ISDIR to see if the file is a directory. Make sure you check the
   * return value from stat() in case there is a problem, e.g., maybe the
   * the file doesn't actually exist.
   */
    struct stat buf;
    if (stat(path, &buf) == 0) {
        return S_ISDIR(buf.st_mode);
    }
    perror("Error: the path is not a directory.\n");
    exit(1);
}
 /* 
 * I needed this because the multiple recursion means there's no way to
 * order them so that the definitions all precede the cause.
 */
 
void process_path(const char*);
void process_directory(const char* path) {
   /*
   * Update the number of directories seen, use opendir() to open the
   * directory, and then use readdir() to loop through the entries
   * and process them. You have to be careful not to process the
   * "." and ".." directory entries, or you'll end up spinning in
   * (infinite) loops. Also make sure you closedir() when you're done.
   *
   * You'll also want to use chdir() to move into this new directory,
   * with a matching call to chdir() to move back out of it when you're
   * done.
   */
    num_dirs++;
    if (chdir(path) != 0) {
        perror("Error: change directory failed\n");
        exit(1);
    }
    DIR *dir = opendir(".");
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (strcmp(ent->d_name, "..") == 0 || strcmp(ent->d_name, ".") == 0) {
            continue;
        }
        process_path(ent->d_name);
    }
    closedir(dir);
    if (chdir("..") != 0) {
        perror("chdir");
        exit(1);
    }
}
__global__ void calculate_sum_kernel(char* data, int size, int* result) { // Key differences
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
        sum += (int)data[i];
    }
    atomicAdd(result, sum); //using shared memory, adding the partial sums to the results
}
void process_path(const char* path) {
    if (is_dir(path)) {
        process_directory(path);
    } else { //replacing process_file
        num_regular++; // increment file count ( 🤨😬 )
        FILE* file = fopen(path, "r"); // reading file
        if (file == NULL) {
            perror("fopen error \n");
            exit(1);
        }
        fseek(file, 0, SEEK_END); // reading files in CUDA
        int size = ftell(file); // ftell() returns location of pointer to file
        fseek(file, 0, SEEK_SET);
        char* data = (char*)malloc(size);
        fread(data, 1, size, file);
        fclose(file);
        char* d_data;
        int* d_result;
        int result = 0;
        cudaMalloc((void**)&d_data, size); // Allocate memory on the device
        cudaMalloc((void**)&d_result, sizeof(int));
        cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice); // transfer memory between CPU and GPU
        cudaMemset(d_result, 0, sizeof(int)); // Initializes or sets device memory to a value
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        calculate_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size, d_result); // launch kernel
        cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_data); // Frees memory on the device.
        cudaFree(d_result);
        free(data);
        printf("%s: %d\n", path, result); 
    }
}
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <path>\n", argv[0]);
        printf("       where <path> is the file or root of the tree you want to summarize.\n");
        exit(1);
    }
    num_dirs = 0;
    num_regular = 0;
    process_path(argv[1]);
    printf("There were %d directories.\n", num_dirs);
    printf("There were %d regular files.\n", num_regular);
    return 0;
}
