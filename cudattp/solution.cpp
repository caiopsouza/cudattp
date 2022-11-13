#include "solution.h"

Solution solutionEmpty(Problem problem) {
	Solution res;

	res.nodes.resize(problem.nodes.size());
	res.items.resize(problem.items.size(), false);

	return res;
}
