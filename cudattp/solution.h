#pragma once

#include <vector>
#include "problem.h"

struct solution
{
	std::vector<int> nodes;
	std::vector<int> items;
};

typedef struct solution Solution;

Solution solutionEmpty(Problem problem);
