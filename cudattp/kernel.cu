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
		auto node_index = solution.nodes[i];
		assert(node_index > 0);

		auto& node = problem.nodes[node_index];

		auto weight_node = 0;
		for (auto& item : problem.items)
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
		previous_node_index = node_index;

		weight_total += weight_node;
	}

	return res;
}

void localSearch(const Problem& problem, Solution& solution) {
	auto iteration = 1;
	auto has_improved = true;

	while (has_improved) {
		has_improved = false;
		auto best_swap_i = 0;
		auto best_swap_j = 0;
		double best_change = 0;

		for (auto i = 1; i < solution.nodes.size() - 1; ++i) {
			for (auto j = i + 1; j < solution.nodes.size(); ++j) {
				auto change = tspCostChangeSwap(problem, solution, i, j);
				if (change < best_change) {
					best_change = change;
					best_swap_i = i;
					best_swap_j = j;
					has_improved = true;
				}
			}
		}

		//std::cout << iteration << " " << best_change << " " << best_swap_i << " " << best_swap_j << " " << solution.nodes[best_swap_i] << " " << solution.nodes[best_swap_j] << " " << tspCost(problem, solution) << " ";
		++iteration;

		while (best_swap_i < best_swap_j) {
			std::swap(solution.nodes[best_swap_i], solution.nodes[best_swap_j]);
			++best_swap_i;
			--best_swap_j;
		}

		//std::cout << tspCost(problem, solution) << " " << std::endl;
	}
}

int main()
{
	std::cout.imbue(std::locale(""));

	std::cout << "time in milliseconds" << std::endl;

	typedef std::chrono::high_resolution_clock Time;
	typedef std::chrono::milliseconds ms;
	//typedef std::chrono::duration<std::chrono::milliseconds> fsec;
	//typedef std::chrono::duration_cast<std::chrono::milliseconds> miliseconds;

	auto t0 = Time::now();
	const auto problem = loadProblemFromFile("instances/tsp/pr1002.tsp");
	auto t1 = Time::now();
	auto elapsed = t1 - t0;
	std::cout << "load problem time: " << std::chrono::duration_cast<ms>(elapsed).count() << std::endl;

	Solution solution = solutionEmpty(problem);
	for (auto i = 0; i < solution.nodes.size(); i++) {
		solution.nodes[i] = i;
	}

	t0 = Time::now();
	localSearch(problem, solution);
	t1 = Time::now();
	elapsed = t1 - t0;
	std::cout << "local search time: " << std::chrono::duration_cast<ms>(elapsed).count() << std::endl;

	std::cout << tspCost(problem, solution) << std::endl;

	/*std::cout << tspCostChangeSwap(problem, solution, 0, 1) << std::endl;

	std::swap(solution.nodes[0], solution.nodes[1]);

	std::cout << tspCost(problem, solution) << std::endl;

	std::cout << tspCostChangeSwap(problem, solution, 0, 1) << std::endl;*/

	/*auto node_a = 12;
	auto node_b = 21;

	auto cost = tspCost(problem, solution);
	std::cout << "cost initial solution: " << cost << std::endl;

	auto cost_swap = tspCostChangeSwap(problem, solution, node_a, node_b);
	std::cout << "cost difference of the swap: " << cost_swap << std::endl;

	std::cout << "cost predict after swap: " << cost + cost_swap << std::endl;

	std::swap(solution.nodes[node_a], solution.nodes[node_b]);
	std::cout << "actual cost after swap: " << tspCost(problem, solution) << std::endl;*/

	freeProblem(problem);
	return 0;
}
