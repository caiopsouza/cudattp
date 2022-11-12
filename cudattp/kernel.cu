
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

inline int distance(const Problem problem, const size_t j, const size_t i) {
	const auto n_j = problem.nodes[j];
	const auto n_i = problem.nodes[i];

	const auto dx = n_j.x - n_i.x;
	const auto dy = n_j.y - n_i.y;

	return static_cast<int>(ceil(sqrt(dx * dx + dy * dy)));
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

	int* solution = (int*)malloc(problem.item_length * sizeof(int));
	if (!solution) {
		fprintf(stderr, "Cannot allocate memory for solution at line %d in %s", __LINE__, __FILE__);
		exit(1);
	}

	for (auto i = 0; i < problem.item_length;i++) {
		solution[i] = i;
	}

	free(solution);
	freeProblem(problem);
	return 0;
}
