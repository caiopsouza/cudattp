#include <assert.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <chrono>
#include <stdio.h>
#include <iostream>

#include "problem.h"
#include "solution.h"

double objectiveValue(const Problem& problem, const Solution& solution) {
	auto weight_total = 0;
	auto utility = 0;

	double res = 0;
	auto speed_coef = (problem.max_speed - problem.min_speed) / problem.knapsack_capacity;

	auto previous_node_index = solution.nodes[0];
	for (auto i = 1; i < solution.nodes.size(); ++i)
	{
		auto& node = solution.nodes[i];

		auto weight_node = 0;
		/*for (auto& item : problem.items)
		{
			if (item.node == node_index && solution.items[item.index])
			{
				weight_node += item.weight;
				utility += item.profit;
				res += item.profit;
			}
		}

		const auto dist = distance(problem, previous_node_index, node_index);
		const auto speed = problem.max_speed - speed_coef * weight_total;
		const auto rent = dist * problem.renting_ratio / speed;

		res -= rent;
		previous_node_index = node_index;*/

		weight_total += weight_node;
	}

	return res;
}

void localSearch(Solution& solution) {
	auto iteration = 1;
	auto has_improved = true;

	while (has_improved) {
		has_improved = false;
		auto best_swap_i = 0;
		auto best_swap_j = 0;
		auto best_change = 0;

		for (auto i = 1; i < solution.nodes.size() - 1; ++i) {
			for (auto j = i + 1; j < solution.nodes.size(); ++j) {
				auto change = tspCostChangeSquaredSwap(solution.nodes, i, j);
				if (change < best_change) {
					best_change = change;
					best_swap_i = i;
					best_swap_j = j;
					has_improved = true;
				}
			}
		}

		//std::cout << iteration << " " << best_change << " " << best_swap_i << " " << best_swap_j << std::endl;
		++iteration;

		while (best_swap_i < best_swap_j) {
			std::swap(solution.nodes[best_swap_i], solution.nodes[best_swap_j]);
			++best_swap_i;
			--best_swap_j;
		}
	}
}

__device__
inline int distanceSquared2(const Node n_j, const Node n_i) {
	const auto dx = n_j.x - n_i.x;
	const auto dy = n_j.y - n_i.y;

	return dx * dx + dy * dy;
}

__device__
inline int positiveModulo2(int n, int m) {
	return (n + m) % m;
}

__device__
int tspCostChangeSquaredSwapDev2(Node* solution_nodes, size_t node_size, unsigned int sol_node_a, unsigned int sol_node_b) {
	const auto& node_a = solution_nodes[sol_node_a];
	const auto& node_b = solution_nodes[sol_node_b];

	const auto& node_before_a = solution_nodes[positiveModulo2(sol_node_a - 1, node_size)];
	const auto& node_after_a = solution_nodes[(sol_node_a + 1) % node_size];

	const auto& node_before_b = solution_nodes[positiveModulo2(sol_node_b - 1, node_size)];
	const auto& node_after_b = solution_nodes[(sol_node_b + 1) % node_size];

	auto distance_removed = distanceSquared2(node_a, node_before_a) + distanceSquared2(node_b, node_after_b);

	auto distance_added = distanceSquared2(node_a, node_after_b) + distanceSquared2(node_b, node_before_a);

	return distance_added - distance_removed;
}

__global__
void searchBestKernel(Node* solution, size_t solution_size, int* best_cost, int* best_j, int* best_i) {
	auto j = blockIdx.x * blockDim.x + threadIdx.x;
	auto i = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= 1 && i < solution_size - 1 && j >= i + 1 && j < solution_size) {
		auto change = tspCostChangeSquaredSwapDev2(solution, solution_size, i, j);
		if (change < *best_cost) {
			*best_cost = change;
			*best_j = j;
			*best_i = i;
		}
	}
}

void localSearchWithCuda(Solution& solution_host) {
	Node* solution = nullptr;

	auto solution_size = solution_host.nodes.size();
	auto solution_size_in_bytes = solution_size * sizeof(Node);

	cudaMalloc((void**)&solution, solution_size_in_bytes);
	cudaMemcpy(solution, solution_host.nodes.data(), solution_size_in_bytes, cudaMemcpyHostToDevice);

	int* aux_params = nullptr;
	cudaMalloc((void**)&aux_params, 3 * sizeof(int));

	int size_of_block = 32;
	int size_of_grid = ceil(solution_size / (float)size_of_block);
	dim3 dim_grid(size_of_grid, size_of_grid, 1);
	dim3 dim_block(size_of_block, size_of_block, 1);

	int return_kernel[3] = { 0, 0, 0 };

	auto has_improved = true;

	while (has_improved) {
		cudaMemset(aux_params, 0, 3 * sizeof(int));
		searchBestKernel << <dim_grid, dim_block >> > (solution, solution_size, &aux_params[0], &aux_params[1], &aux_params[2]);
		cudaMemcpy(return_kernel, aux_params, 3 * sizeof(int), cudaMemcpyDeviceToHost);
		
		has_improved = return_kernel[0] < 0;

		//std::cout << return_kernel[0] << " " << return_kernel[1] << " " << return_kernel[2] << std::endl;

		if (has_improved) {
			auto best_swap_i = return_kernel[2];
			auto best_swap_j = return_kernel[1];

			while (best_swap_i < best_swap_j) {
				std::swap(solution_host.nodes[best_swap_i], solution_host.nodes[best_swap_j]);
				++best_swap_i;
				--best_swap_j;
			}

			cudaMemcpy(solution, solution_host.nodes.data(), solution_size_in_bytes, cudaMemcpyHostToDevice);
		}
	}

	cudaFree(solution);
}

int main()
{
	std::cout.imbue(std::locale(""));

	std::cout << "time in microseconds" << std::endl;

	typedef std::chrono::high_resolution_clock Time;
	typedef std::chrono::microseconds ms;

	auto t0 = Time::now();
	const auto problem = loadProblemFromFile("instances/tsp/pcb3038.tsp");
	auto t1 = Time::now();
	auto elapsed = t1 - t0;
	std::cout << "load problem time: " << std::chrono::duration_cast<ms>(elapsed).count() << std::endl;

	Solution solution = solutionEmpty(problem);
	for (auto i = 0; i < problem.nodes.size(); i++) {
		solution.nodes[i] = problem.nodes[i];
	}
	solution.nodes[solution.nodes.size() - 1] = problem.nodes[0];

	// 22.473.780 us, 151.250 cost, 137.694 bkr
	t0 = Time::now();
	localSearchWithCuda(solution);
	t1 = Time::now();
	elapsed = t1 - t0;
	std::cout << "local search time: " << std::chrono::duration_cast<ms>(elapsed).count() << std::endl;

	std::cout << tspCost(problem, solution) << std::endl;

	for (Node node : solution.nodes) {
		std::cout << problem.nodes_index.at(node) << " ";
	}
	std::cout << std::endl;

	freeProblem(problem);
	return 0;
}
