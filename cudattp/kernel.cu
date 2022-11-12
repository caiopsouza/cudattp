
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "problem.h"

__global__
void matrixMulKernel(const int* const m, const int* const n, int* const p, const unsigned int size) {
	const auto row = blockIdx.y * blockDim.y + threadIdx.y;
	const auto col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= size || col >= size) {
		return;
	}

	int res = 0;

	for (unsigned int k = 0; k < size; ++k) {
		res += m[row * size + k] * n[k * size + col];
	};

	p[row * size + col] = res;
}

void matrixMul(const int* const m, const int* const n, int* p, const unsigned int size) {
	const auto size_in_bytes = size * size * sizeof(int);

	int* m_dev = nullptr;
	int* n_dev = nullptr;
	int* p_dev = nullptr;

	cudaMalloc(&m_dev, size_in_bytes);
	cudaMalloc(&n_dev, size_in_bytes);
	cudaMalloc(&p_dev, size_in_bytes);

	cudaMemcpy(m_dev, m, size_in_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(n_dev, n, size_in_bytes, cudaMemcpyHostToDevice);

	dim3 blockDim(size, size, 1);
	matrixMulKernel << <1, blockDim >> > (m_dev, n_dev, p_dev, size);

	cudaMemcpy(p, p_dev, size_in_bytes, cudaMemcpyDeviceToHost);

	cudaFree(m_dev);
	cudaFree(n_dev);
	cudaFree(p_dev);
}

int main()
{
	const auto problem = loadProblemFromFile("instances/a280_n279_bounded-strongly-corr_01.ttp");

	std::cout
		<< "dimension: " << problem.node_count << std::endl
		<< "number of items: " << problem.item_count << std::endl
		<< "capacity of knapsack: " << problem.knapsack_capacity << std::endl
		<< "min speed: " << problem.min_speed << std::endl
		<< "max speed: " << problem.max_speed << std::endl
		<< "renting speed: " << problem.renting_ratio << std::endl;

	freeProblem(problem);
	return 0;
}
