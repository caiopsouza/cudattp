#include "solution.h"

#include <assert.h>

Solution solutionEmpty(Problem problem) {
	Solution res;

	res.nodes.resize(problem.nodes.size() + 1);
	res.items.resize(problem.items.size(), false);

	return res;
}

inline int distanceSquared(const Node n_j, const Node n_i) {
	const auto dx = n_j.x - n_i.x;
	const auto dy = n_j.y - n_i.y;

	return dx * dx + dy * dy;
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
	return actualDistance(problem,solution.nodes[j], solution.nodes[i]);
}

double tspCost(const Problem& problem, const Solution& solution) {
	double res = 0;
	Node previous_node = solution.nodes.back();

	for (auto& node: solution.nodes)
	{
		const auto dist = actualDistance(problem, previous_node, node);
		res += dist;

		previous_node = node;
	}

	return res;
}

__device__
int tspCostChangeSquaredSwapDev(Node* solution_nodes, size_t node_size, unsigned int sol_node_a, unsigned int sol_node_b) {
	const auto& node_a = solution_nodes[sol_node_a];
	const auto& node_b = solution_nodes[sol_node_b];

	const auto& node_before_a = solution_nodes[sol_node_a - 1];
	const auto& node_after_a = solution_nodes[sol_node_a + 1];

	const auto& node_before_b = solution_nodes[sol_node_b - 1];
	const auto& node_after_b = solution_nodes[sol_node_b + 1];

	auto distance_removed = distanceSquared(node_a, node_before_a) + distanceSquared(node_b, node_after_b);

	auto distance_added = distanceSquared(node_a, node_after_b) + distanceSquared(node_b, node_before_a);

	return distance_added - distance_removed;
}

int tspCostChangeSquaredSwap(const std::vector<Node>& solution_nodes, int sol_node_a, int sol_node_b) {
	auto node_size = solution_nodes.size();

	const auto& node_a = solution_nodes[sol_node_a];
	const auto& node_b = solution_nodes[sol_node_b];

	const auto& node_before_a = solution_nodes[sol_node_a - 1];
	const auto& node_after_a = solution_nodes[sol_node_a + 1];

	const auto& node_before_b = solution_nodes[sol_node_b - 1];
	const auto& node_after_b = solution_nodes[sol_node_b + 1];

	auto distance_removed = distanceSquared(node_a, node_before_a) + distanceSquared(node_b, node_after_b);

	auto distance_added = distanceSquared(node_a, node_after_b) + distanceSquared(node_b, node_before_a);

	return distance_added - distance_removed;
}

int tspCostChangeSquaredSwap(const Problem& problem, Solution& solution, int sol_node_a, int sol_node_b) {
	auto node_size = problem.nodes.size();

	auto& node_a = solution.nodes[sol_node_a];
	auto& node_b = solution.nodes[sol_node_b];

	const auto& node_before_a = solution.nodes[sol_node_a - 1];
	const auto& node_after_a = solution.nodes[sol_node_a + 1];

	const auto& node_before_b = solution.nodes[sol_node_b - 1];
	const auto& node_after_b = solution.nodes[sol_node_b + 1];

	auto distance_removed = distanceSquared(node_a, node_before_a) + distanceSquared(node_b, node_after_b);

	auto distance_added = distanceSquared(node_a, node_after_b) + distanceSquared(node_b, node_before_a);

	return distance_added - distance_removed;
}

/*double tspCostChangeSwap(const Problem& problem, Solution& solution, int node_a, int node_b) {
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
}*/
