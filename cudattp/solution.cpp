#include "solution.h"

#include <assert.h>

Solution solutionEmpty(Problem problem) {
	Solution res;

	res.nodes.resize(problem.nodes.size());
	res.items.resize(problem.items.size(), false);

	return res;
}

int distance(const Problem& problem, const Node n_j, const Node n_i) {
	const auto dx = n_j.x - n_i.x;
	const auto dy = n_j.y - n_i.y;

	return static_cast<int>(ceil(sqrt(dx * dx + dy * dy)));
}

int distance(const Problem& problem, const int j, const int i) {
	return distance(problem, problem.nodes[j], problem.nodes[i]);
}

double actualDistance(const Problem& problem, const Node n_j, const Node n_i) {
	const auto dx = n_j.x - n_i.x;
	const auto dy = n_j.y - n_i.y;

	return sqrt(dx * dx + dy * dy);
}

double actualDistance(const Problem& problem, const int j, const int i) {
	return actualDistance(problem, problem.nodes[j], problem.nodes[i]);
}

double actualDistance(const Problem& problem, const Solution& solution, const int j, const int i) {
	return actualDistance(problem, problem.nodes[solution.nodes[j]], problem.nodes[solution.nodes[i]]);
}

double tspCost(const Problem& problem, const Solution& solution) {
	double res = 0;
	Node previous_node = problem.nodes[solution.nodes.back()];

	for (auto& node_index : solution.nodes)
	{
		auto& node = problem.nodes[node_index];

		const auto dist = actualDistance(problem, previous_node, node);
		res += dist;

		previous_node = node;
	}

	return res;
}

inline int positiveModulo(int n, int m) {
	return (n + m) % m;
}

double tspCostChangeSwap(const Problem& problem, Solution& solution, int node_a, int node_b) {
	auto node_size = problem.nodes.size();

	const int node_before_a = positiveModulo(node_a - 1, node_size);
	const int node_after_a = (node_a + 1) % node_size;

	const int node_before_b = positiveModulo(node_b - 1, node_size);
	const int node_after_b = (node_b + 1) % node_size;

	auto distance_removed = actualDistance(problem, solution, node_a, node_before_a)
		+ actualDistance(problem, solution, node_b, node_after_b);

	auto distance_added = actualDistance(problem, solution, node_a, node_after_b)
		+ actualDistance(problem, solution, node_b, node_before_a);

	return distance_added - distance_removed;
}
