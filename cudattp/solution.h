#pragma once

#include <vector>
#include "problem.h"

struct solution
{
	std::vector<Node> nodes;
	std::vector<int> items;
};

typedef struct solution Solution;

Solution solutionEmpty(Problem problem);

// Euclidian distance between two nodes
int distance(const Problem& problem, const Node n_j, const Node n_i);
int distance(const Problem& problem, const int j, const int i);

// Cost of the solution for the TSP
double tspCost(const Problem& problem, const Solution& solution);

double tspCostChangeSwap(const Problem& problem, Solution& solution, int node_a, int node_b);

int tspCostChangeSquaredSwap(const std::vector<Node>& solution_nodes, int sol_node_a, int sol_node_b);
int tspCostChangeSquaredSwap(const Problem& problem, Solution& solution, int sol_node_a, int sol_node_b);
