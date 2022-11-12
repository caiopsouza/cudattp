
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <stdio.h>
#include <iostream>

#include "problem.h"

inline int distance(const Problem problem, const size_t j, const size_t i) {
	const auto& n_j = problem.nodes[j];
	const auto& n_i = problem.nodes[i];

	const auto dx = n_j.x - n_i.x;
	const auto dy = n_j.y - n_i.y;

	return static_cast<int>(ceil(sqrt(dx * dx + dy * dy)));
}

int main()
{
	std::cout.imbue(std::locale(""));

	std::cout << "time in seconds" << std::endl;

	typedef std::chrono::high_resolution_clock Time;
	typedef std::chrono::duration<float> fsec;

	auto t0 = Time::now();
	const auto problem = loadProblemFromFile("instances/pla33810_n338090_uncorr_10.ttp");
	auto t1 = Time::now();
	fsec elapsed = t1 - t0;
	std::cout << "load problem time: " << elapsed.count() << std::endl;

	int* solution = (int*)malloc(problem.items.size() * sizeof(int));
	if (!solution) {
		fprintf(stderr, "Cannot allocate memory for solution at line %d in %s", __LINE__, __FILE__);
		exit(1);
	}

	for (auto i = 0; i < problem.items.size(); i++) {
		solution[i] = i;
	}

	free(solution);
	freeProblem(problem);
	return 0;
}
