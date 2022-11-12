#include "problem.h"

#include <fstream>
#include <iostream>
#include <sstream>

template<typename T>
void getValueFromLine(std::string line, std::string key, T* value) {
	if (line.find(key) == std::string::npos) {
		return;
	}

	const auto index = key.length() + 1;
	if (index > line.length()) {
		return;
	}

	std::istringstream ss(line.substr(index));
	ss >> *value;
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

	while (std::getline(infile, line) && line.find("NODE_COORD_SECTION") == std::string::npos) {
		getValueFromLine(line, "DIMENSION", &problem.node_count);
		getValueFromLine(line, "NUMBER OF ITEMS", &problem.item_count);
		getValueFromLine(line, "CAPACITY OF KNAPSACK", &problem.knapsack_capacity);
		getValueFromLine(line, "MIN SPEED", &problem.min_speed);
		getValueFromLine(line, "MAX SPEED", &problem.max_speed);
		getValueFromLine(line, "RENTING RATIO", &problem.renting_ratio);
		getValueFromLine(line, "EDGE_WEIGHT_TYPE", &edge_weight_type);
	}

	if (edge_weight_type != "CEIL_2D") {
		fprintf(stderr, "This program only support edge_weight_type=CEIL_2D at line %d in %s", __LINE__, __FILE__);
		exit(1);
	}

	if (problem.node_count == 0) {
		fprintf(stderr, "Cannot read DIMENSION from the file at line %d in %s", __LINE__, __FILE__);
		exit(1);
	}

	if (problem.item_count == 0) {
		fprintf(stderr, "Cannot read NUMBER OF ITEMS from the file at line %d in %s", __LINE__, __FILE__);
		exit(1);
	}

	problem.nodes = (Point*)malloc((problem.node_count + 1) * sizeof(Point));
	if (!problem.nodes) {
		fprintf(stderr, "Cannot allocate memory for nodes at line %d in %s", __LINE__, __FILE__);
		exit(1);
	}

	problem.items = (Item*)malloc((problem.item_count + 1) * sizeof(Item));
	if (!problem.items) {
		fprintf(stderr, "Cannot allocate memory for items at line %d in %s", __LINE__, __FILE__);
		exit(1);
	}

	for (auto i = 1; i <= problem.node_count; i++) {
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

	for (auto i = 1; i <= problem.item_count; i++) {
		unsigned int index, node;
		int profit, weight;

		infile >> index >> profit >> weight >> node;

		if (index != i) {
			fprintf(stderr, "Index %d different of position %d when reading items at line %d in %s", index, i, __LINE__, __FILE__);
			exit(1);
		}

		problem.items[i] = { .profit = profit, .weight = weight, .node = node };
	}

	return problem;
}


void freeProblem(const Problem problem) {
	free(problem.nodes);;
}