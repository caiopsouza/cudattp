#pragma once

#include <string>
#include <vector>
#include <map>

struct node {
	int x = 0, y = 0;

	bool operator==(const node& o) const {
		return x == o.x && y == o.y;
	}

	bool operator<(const node& o) const {
		return x < o.x || (x == o.x && y < o.y);
	}
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
	std::map<Node, unsigned int> nodes_index;
	int knapsack_capacity = 0;
	double min_speed = 0, max_speed = 0, renting_ratio = 0;
};

typedef struct problem Problem;

Problem loadProblemFromFile(const std::string filename);

void freeProblem(const Problem problem);