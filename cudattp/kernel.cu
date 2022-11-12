
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
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

int* computeDistanceMatrix(const Problem problem) {
	int* res = (int*)malloc(problem.node_length * problem.node_length * sizeof(int));

	for (auto j = 1; j < problem.node_length; j++) {
		for (auto i = 1; i <= j; i++) {
			const auto n_j = problem.nodes[j];
			const auto n_i = problem.nodes[i];

			const auto dx = n_j.x - n_i.x;
			const auto dy = n_j.y - n_i.y;

			const auto dist = static_cast<int>(ceil(sqrt(dx * dx + dy * dy)));
			res[j * problem.node_length + i] = res[i * problem.node_length + j] = dist;
		}
	}

	return res;
}

int main()
{
	std::cout.imbue(std::locale(""));

	std::cout << "time in seconds" << std::endl;

	typedef std::chrono::high_resolution_clock Time;
	typedef std::chrono::duration<float> fsec;

	auto t0 = Time::now();
	const auto problem = loadProblemFromFile("instances/pla33810_n338090_uncorr_10.ttp");
	auto t1 = Time::now();
	fsec elapsed = t1 - t0;
	std::cout << "load problem time: " << elapsed.count() << std::endl;

	t0 = Time::now();
	const auto dist_matrix = computeDistanceMatrix(problem);
	t1 = Time::now();
	elapsed = t1 - t0;
	std::cout << "compute distance matrix time (cpu): " << elapsed.count() << std::endl;

	std::cout << dist_matrix[1 * problem.node_length + 1] << " " << dist_matrix[1 * problem.node_length + 1] << std::endl;
	std::cout << dist_matrix[1 * problem.node_length + 2] << " " << dist_matrix[2 * problem.node_length + 1] << std::endl;
	std::cout << dist_matrix[1 * problem.node_length + 3] << " " << dist_matrix[3 * problem.node_length + 1] << std::endl;

	freeProblem(problem);
	return 0;
}
