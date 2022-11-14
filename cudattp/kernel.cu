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

void localSearch(const Problem& problem, Solution& solution) {
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

	std::cout << "time in microseconds" << std::endl;

	typedef std::chrono::high_resolution_clock Time;
	typedef std::chrono::microseconds ms;

	auto t0 = Time::now();
	const auto problem = loadProblemFromFile("instances/tsp/pr1002.tsp");
	auto t1 = Time::now();
	auto elapsed = t1 - t0;
	std::cout << "load problem time: " << std::chrono::duration_cast<ms>(elapsed).count() << std::endl;

	Solution solution = solutionEmpty(problem);
	for (auto i = 0; i < problem.nodes.size(); i++) {
		solution.nodes[i] = problem.nodes[i];
	}
	solution.nodes[solution.nodes.size() - 1] = problem.nodes[0];

	// 424 us
	t0 = Time::now();
	localSearch(problem, solution);
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
