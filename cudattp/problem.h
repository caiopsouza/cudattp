#pragma once

#include <string>
#include <vector>

struct node {
	unsigned int index = 0;
	int x = 0, y = 0;
};
typedef struct node Node;

struct item {
	size_t index = 0, node = 0;
	int profit = 0, weight = 0;
};
typedef struct item Item;

struct problem {
	std::vector<Node> nodes;
	std::vector<Item> items;
	int knapsack_capacity = 0;
	double min_speed = 0, max_speed = 0, renting_ratio = 0;
};

typedef struct problem Problem;

Problem loadProblemFromFile(const std::string filename);

void freeProblem(const Problem problem);