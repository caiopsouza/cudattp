#include <assert.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <chrono>
#include <stdio.h>
#include <iostream>

#include "problem.h"
#include "solution.h"

inline int distance(const Problem& problem, const size_t j, const size_t i) {
	const auto& n_j = problem.nodes[j];
	const auto& n_i = problem.nodes[i];

	const auto dx = n_j.x - n_i.x;
	const auto dy = n_j.y - n_i.y;

	return static_cast<int>(ceil(sqrt(dx * dx + dy * dy)));
}

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

int main()
{
	std::cout.imbue(std::locale(""));

	std::cout << "time in seconds" << std::endl;

	typedef std::chrono::high_resolution_clock Time;
	typedef std::chrono::duration<float> fsec;

	auto t0 = Time::now();
	const auto problem = loadProblemFromFile("instances/eil101_n100_uncorr-similar-weights_01.ttp");
	auto t1 = Time::now();
	fsec elapsed = t1 - t0;
	std::cout << "load problem time: " << elapsed.count() << std::endl;

	Solution solution = solutionEmpty(problem);
	for (auto i = 0; i < solution.nodes.size(); i++) {
		solution.nodes[i] = i + 1;
	}
	solution.nodes[solution.nodes.size() - 1] = 1;

	std::vector<Item> items_copy(problem.items);
	std::sort(items_copy.begin(), items_copy.end(), [](Item a, Item b) { return a.weight < b.weight; });

	auto utility = 0, weight = 0;
	for (auto& item : items_copy) {
		if (weight + item.weight > problem.knapsack_capacity) {
			break;
		}
		weight += item.weight;
		utility += item.profit;
		solution.items[item.node] = true;
	}

	const auto benefit = objectiveValue(problem, solution);
	assert(benefit == -53287.851827228456);

	std::cout << std::endl;
	std::cout << "knapsack_capacity: " << problem.knapsack_capacity << std::endl;
	std::cout << "weight: " << weight << std::endl;
	std::cout << "utility: " << utility << std::endl;
	std::cout << "benefit: " << benefit << std::endl;


	freeProblem(problem);
	return 0;
}
