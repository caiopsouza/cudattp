#include "problem.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

template<typename T>
void getValueFromLine(std::string line, std::string key, T& value) {
	if (!line.starts_with(key)) {
		return;
	}

	std::istringstream ss(line);
	ss.ignore(std::numeric_limits<std::streamsize>::max(), ':');
	ss >> value;
}

bool compareItemsByLocation(const Item a, const Item b)
{
	if (a.node != b.node) {
		return a.node < b.node;
	}
	return a.index < b.index;
}

Problem loadProblemFromFile(const std::string filename)
{
	Problem problem{};

	std::ifstream infile(filename);

	if (!infile) {
		fprintf(stderr, "Cannot read file %s at line %d in %s", filename.c_str(), __LINE__, __FILE__);
		exit(1);
	}

	std::string type;
	std::string line;
	std::string edge_weight_type;

	size_t node_length = 0;
	size_t item_length = 0;

	while (std::getline(infile, line) && line.find("NODE_COORD_SECTION") == std::string::npos) {
		getValueFromLine(line, "TYPE", type);
		getValueFromLine(line, "DIMENSION", node_length);
		getValueFromLine(line, "NUMBER OF ITEMS", item_length);
		getValueFromLine(line, "CAPACITY OF KNAPSACK", problem.knapsack_capacity);
		getValueFromLine(line, "MIN SPEED", problem.min_speed);
		getValueFromLine(line, "MAX SPEED", problem.max_speed);
		getValueFromLine(line, "RENTING RATIO", problem.renting_ratio);
		getValueFromLine(line, "EDGE_WEIGHT_TYPE", edge_weight_type);
	}

	if (edge_weight_type != "EUC_2D" && edge_weight_type != "CEIL_2D") {
		fprintf(stderr, "This program only support EDGE_WEIGHT_TYPE=EUC_2D|CEIL_2D at line %d in %s", __LINE__, __FILE__);
		exit(1);
	}

	if (node_length == 0) {
		fprintf(stderr, "Cannot read DIMENSION from the file at line %d in %s", __LINE__, __FILE__);
		exit(1);
	}

	problem.nodes.resize(node_length);
	if (problem.nodes.empty()) {
		fprintf(stderr, "Cannot allocate memory for nodes at line %d in %s", __LINE__, __FILE__);
		exit(1);
	}

	if (type != "TSP") { // Assume TTP
		if (item_length == 0) {
			fprintf(stderr, "Cannot read NUMBER OF ITEMS from the file at line %d in %s", __LINE__, __FILE__);
			exit(1);
		}

		problem.items.resize(item_length);
		if (problem.items.empty()) {
			fprintf(stderr, "Cannot allocate memory for items at line %d in %s", __LINE__, __FILE__);
			exit(1);
		}
	}

	for (auto i = 0; i < problem.nodes.size(); ++i) {
		unsigned int index;
		double x, y;

		infile >> index >> x >> y;

		if (index != i + 1) {
			fprintf(stderr, "Index %d different of position %d when reading nodes at line %d in %s", index, i + 1, __LINE__, __FILE__);
			exit(1);
		}

		problem.nodes[i] = { .index = index, .x = (int)x, .y = (int)y };
	}

	if (type != "TSP") { // Assume TTP
		infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		std::getline(infile, line);

		problem.items[0] = Item{};
		for (auto i = 0; i < problem.items.size(); i++) {
			unsigned int index, node;
			int profit, weight;

			infile >> index >> profit >> weight >> node;

			if (index != i + 1) {
				fprintf(stderr, "Index %d different of position %d when reading items at line %d in %s", index, i + 1, __LINE__, __FILE__);
				exit(1);
			}

			problem.items[i] = {
				.index = index,
				.node = node,
				.profit = profit,
				.weight = weight
			};
		}
	}

	return problem;
}

void freeProblem(const Problem problem) {
}
