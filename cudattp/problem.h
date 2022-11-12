#pragma once

#include <string>

struct node {
	int x = 0, y = 0;
	size_t next_node_index_item;
};
typedef struct node Node;

struct item {
	size_t index = 0, node = 0;
	int profit = 0, weight = 0;
};
typedef struct item Item;

struct problem {
	size_t node_length = 0;
	Node* nodes = nullptr;

	size_t item_length = 0;
	Item* items = nullptr;

	int knapsack_capacity = 0;
	float min_speed = 0, max_speed = 0, renting_ratio = 0;
};

typedef struct problem Problem;

Problem loadProblemFromFile(const std::string filename);

void freeProblem(const Problem problem);