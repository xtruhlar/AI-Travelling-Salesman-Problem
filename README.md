# AI-Travelling-Salesman-Problem

task involved solving the traveling salesman problem. I had to visit multiple cities while minimizing travel costs, with the route forming a closed loop, meaning I had to return to the city I started from.
I had at least 20 cities at my disposal, each with randomly generated coordinates. The cost of travel between two cities was determined by the Euclidean distance. The total length of the route was determined by a permutation of cities, and my task was to find a permutation with the smallest total distance.
In the case of the genetic algorithm, I represented individuals using a vector with the order of cities. The fitness value of an individual was the inverse of the length of its route. I initialized the first generation and implemented at least two methods for parent selection. I addressed crossover and mutations of individuals. I evaluated the results and compared different methods for generating the next generation or selection.
In the case of tabu search, I also used a representation with a vector of the order of cities. My algorithm generated successors and moved to better states while maintaining a list of tabu states to avoid cycling in local extremes. The length of this list was an important parameter.
Example:
![image](https://github.com/user-attachments/assets/2e69f0c3-2acb-43e5-afae-3d6f8e277f5c)
