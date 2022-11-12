#include "problem.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

template<typename T>
void getValueFromLine(std::string line, std::string key, T& value) {
	if (line.find(key) == std::string::npos) {
		return;
	}

	const auto index = key.length() + 1;
	if (index > line.length()) {
		return;
	}

	std::istringstream ss(line.substr(index));
	ss >> value;
}

bool compareItemsByLocation(const Item a, const Item b)
{
	if (a.node != b.node) {
		return a.node < b.node;
	}
	return a.index < b.index;
}

void sortItems(Problem problem) {
	std::sort(problem.items.begin(), problem.items.end(), compareItemsByLocation);

	size_t item_index = 0;
	for (auto i = 0; i < problem.nodes.size(); i++) {
		while (item_index < problem.items.size() && problem.items[item_index].node == i) {
			item_index++;
		}
		problem.nodes[i].next_node_index_item = item_index;
	}
}

Problem loadProblemFromFile(const std::string filename)
{
	Problem problem;

	std::ifstream infile(filename);

	if (!infile) {
		fprintf(stderr, "Cannot read file %s at line %d in %s", filename.c_str(), __LINE__, __FILE__);
		exit(1);
	}

	std::string line;
	std::string edge_weight_type;

	size_t node_length = 0;
	size_t item_length = 0;

	while (std::getline(infile, line) && line.find("NODE_COORD_SECTION") == std::string::npos) {
		getValueFromLine(line, "DIMENSION", node_length);
		getValueFromLine(line, "NUMBER OF ITEMS", item_length);
		getValueFromLine(line, "CAPACITY OF KNAPSACK", problem.knapsack_capacity);
		getValueFromLine(line, "MIN SPEED", problem.min_speed);
		getValueFromLine(line, "MAX SPEED", problem.max_speed);
		getValueFromLine(line, "RENTING RATIO", problem.renting_ratio);
		getValueFromLine(line, "EDGE_WEIGHT_TYPE", edge_weight_type);
	}

	if (edge_weight_type != "CEIL_2D") {
		fprintf(stderr, "This program only support edge_weight_type=CEIL_2D at line %d in %s", __LINE__, __FILE__);
		exit(1);
	}

	if (node_length == 0) {
		fprintf(stderr, "Cannot read DIMENSION from the file at line %d in %s", __LINE__, __FILE__);
		exit(1);
	}

	if (item_length == 0) {
		fprintf(stderr, "Cannot read NUMBER OF ITEMS from the file at line %d in %s", __LINE__, __FILE__);
		exit(1);
	}

	problem.nodes.resize(node_length + 1);
	if (problem.nodes.empty()) {
		fprintf(stderr, "Cannot allocate memory for nodes at line %d in %s", __LINE__, __FILE__);
		exit(1);
	}

	problem.items.resize(item_length + 1);
	if (problem.items.empty()) {
		fprintf(stderr, "Cannot allocate memory for items at line %d in %s", __LINE__, __FILE__);
		exit(1);
	}

	problem.nodes[0] = Node{};
	for (auto i = 1; i < problem.nodes.size(); i++) {
		unsigned int index;
		int x, y;

		infile >> index >> x >> y;

		if (index != i) {
			fprintf(stderr, "Index %d different of position %d when reading nodes at line %d in %s", index, i, __LINE__, __FILE__);
			exit(1);
		}

		problem.nodes[i] = { .x = x, .y = y };
	}

	infile.ignore(1, '\n');
	std::getline(infile, line);

	problem.items[0] = Item{};
	for (auto i = 1; i < problem.items.size(); i++) {
		unsigned int index, node;
		int profit, weight;

		infile >> index >> profit >> weight >> node;

		if (index != i) {
			fprintf(stderr, "Index %d different of position %d when reading items at line %d in %s", index, i, __LINE__, __FILE__);
			exit(1);
		}

		problem.items[i] = {
			.index = index,
			.node = node,
			.profit = profit,
			.weight = weight
		};
	}

	sortItems(problem);

	return problem;
}

void freeProblem(const Problem problem) {
}
