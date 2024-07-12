import math
import random
import matplotlib.pyplot as plt
import config


# class reprezentujuci jedno mesto
class Mesto:
    def __init__(self, num, x, y):
        self.num = num
        self.x = x
        self.y = y


# class reprezentujuci jeden chromozom
class Chromosome:
    def __init__(self, zoznam):
        self.chromosome = zoznam

        chromosome_representation = []
        for i in range(0, len(zoznam)):
            chromosome_representation.append(self.chromosome[i].num)
        self.chromosome_representation = chromosome_representation

        # vypocet fitness a cost, distance je vzdialenost medzi miestami
        cost = 0
        # vzdialenost ziskam z matice vzdialenosti
        for j in range(1,
                       len(self.chromosome_representation) - 1):
            cost += matrix[self.chromosome_representation[j] - 1][self.chromosome_representation[j + 1] - 1]
        self.cost = cost
        self.fitness_value = 1 / self.cost


# distance je využitá pri generovaní miest
def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def generate_points(N, min_distance):
    cities = set()
    while len(cities) < N:
        x = random.randint(1, 200)
        y = random.randint(1, 200)
        point = (x, y)

        # zabezpecenie minimalnej vzdialenosti medzi mestami
        valid_point = all(distance(point, existing_point) >= min_distance for existing_point in cities)
        if valid_point:
            cities.add(point)

    # zapísanie miest do suboru
    with open('mesta.txt', 'w') as file:
        for i, (x, y) in enumerate(cities, start=1):
            file.write(f"{i}\t{x}\t{y}\n")


# nacitanie miest zo suboru
def load_points():
    mesta = []
    with open('mesta.txt', 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                num, x, y = line.split('\t')
                mesta.append(Mesto(int(num), int(x), int(y)))
    return mesta


# vytvorenie matice vzdialenosti
def create_distance_matrix(cities):
    num_cities = config.NUMBER_OF_CITIES
    mat = [[0] * num_cities for _ in range(num_cities)]

    for i in range(0, len(mat)):
        for j in range(0, len(mat)):
            # mat[x][y] = √(x1-x2)^2 + (y1-y2)^2
            mat[i][j] = math.sqrt((cities[i].x - cities[j].x) ** 2 + (cities[i].y - cities[j].y) ** 2)
    return mat


# vygeneruje nahodne poradie miest - chromozom
def create_random_list(zoznam_miest):
    shuffled_miest = random.sample(zoznam_miest[1:], len(zoznam_miest) - 1)
    return [zoznam_miest[0]] + shuffled_miest + [zoznam_miest[0]]


# vytvorenie novej generacie - Chromozom
def initialization(data, pop_size):
    return [Chromosome(create_random_list(data)) for _ in range(pop_size)]


# vyber jedinca pomocou rulety
def roulette_selection(population):
    total_fitness = sum(chromosome.cost for chromosome in population)
    rand_value = random.uniform(0, total_fitness)

    bullet = 0
    for chromosome in population:
        bullet += chromosome.cost
        if bullet >= rand_value:
            return chromosome


# vyber jedinca pomocou turnaja
def tournament_selection(population, tournament_size=config.TOURNAMENT_SIZE):
    tickets = random.sample(range(len(population)), tournament_size)
    best_chromosome = min((population[ticket] for ticket in tickets), key=lambda chromosome: chromosome.cost)
    return best_chromosome


# krizenie - 2 body + mixing
def crossover_mixing(parent1, parent2):
    # vyberie 2 nahodne indexy
    index1, index2 = random.sample(range(1, len(parent1.chromosome) - 1), 2)
    start, end = min(index1, index2), max(index1, index2)

    # child1 je zaciatok parent1 + nepouzite z parent2 + koniec parent1
    child_1_s = parent1.chromosome[:start]
    child_1_e = parent1.chromosome[end:]
    child1 = child_1_s + child_1_e
    unused2 = [city for city in parent2.chromosome[1:-1] if city not in child1]
    child1 = child_1_s + unused2 + child_1_e

    # child2 je index1 v parent2 až index2 v parent2 + nepouzite z parent1
    child2 = parent2.chromosome[start:end + 1]
    unused1 = [city for city in parent1.chromosome[1:-1] if city not in child2]
    child2 = child2 + unused1
    child2 = [parent2.chromosome[0]] + child2 + [parent2.chromosome[0]]

    return Chromosome(child1), Chromosome(child2)


# krizenie - 2 body
def crossover_2_points(parent1, parent2):
    # vyberie 2 nahodne indexy
    index1, index2 = random.sample(range(1, len(parent1.chromosome) - 1), 2)
    start, end = min(index1, index2), max(index1, index2)

    # child 1 je od index1 do index2 z parent1 + nepouzite z parent2
    base1 = parent1.chromosome[start:end + 1]
    unused2 = [city for city in parent2.chromosome[1:-1] if city not in base1]
    base1.extend(unused2)
    child1 = [parent1.chromosome[0]] + base1 + [parent1.chromosome[0]]

    # child 2 je od index1 do index2 z parent2 + nepouzite z parent1
    base2 = parent2.chromosome[start:end + 1]
    unused1 = [city for city in parent1.chromosome[1:-1] if city not in base2]
    base2.extend(unused1)
    child2 = [parent2.chromosome[0]] + base2 + [parent2.chromosome[0]]

    return Chromosome(child1), Chromosome(child2)


# mutacia - vymeni 2 nahodne miesta, ktore nie su od seba vzdialenejšie ako MUTATION_DIST
def mutation(chromosome):
    index1, index2 = random.sample(range(1, len(chromosome.chromosome) - 1), 2)
    # kontrola vzdialenosti
    while abs(index1 - index2) > config.MUTATION_DIST:
        index1, index2 = random.sample(range(1, len(chromosome.chromosome) - 1), 2)

    chromosome.chromosome[index1], chromosome.chromosome[index2] = chromosome.chromosome[index2], chromosome.chromosome[index1]
    return chromosome


# vytvorenie novej generacie
def create_new_generation(population, mutation_rate):
    # elitizmus
    new_generation = [min(population, key=lambda chromosome: chromosome.cost)]
    # vytvorenie novej generacie
    for x in range(0, int(len(population) / 2)):
        # vyber rodicov
        parent1_tournament, parent2_tournament = tournament_selection(population), tournament_selection(population)
        parent1_roulette, parent2_roulette = roulette_selection(population), roulette_selection(population)
        parent1 = max([parent1_tournament, parent1_roulette], key=lambda chromosome: chromosome.fitness_value)
        parent2 = max([parent2_tournament, parent2_roulette], key=lambda chromosome: chromosome.fitness_value)

        # krizenie
        child1_2_points, child2_2_points = crossover_2_points(parent1, parent2)
        child1_mixing, child2_mixing = crossover_mixing(parent1, parent2)
        child1 = max([child1_2_points, child1_mixing], key=lambda chromosome: chromosome.fitness_value)
        child2 = max([child2_2_points, child2_mixing], key=lambda chromosome: chromosome.fitness_value)

        # mutacia
        rand_num = random.randrange(0, 100)
        if rand_num <= mutation_rate:
            child1 = mutation(child1)
        rand_num = random.randrange(0, 100)
        if rand_num <= mutation_rate:
            child2 = mutation(child2)

        # pridanie do novej generacie
        new_generation.append(Chromosome(child1.chromosome))
        new_generation.append(Chromosome(child2.chromosome))

    return new_generation


# geneticky algoritmus
def genetic_algorithm(mesta, pop_size, mutation_rate, num_of_generations):
    # inicializacia
    generation = initialization(mesta, pop_size)

    # vytvorenie novej generacie
    for i in range(0, num_of_generations):
        generation = create_new_generation(generation, mutation_rate)

        # vypis najlepsieho jedinca
        print(
            f"\r" + str(i + 1) + ". generation: " + "Chromosome: " + str(
                generation[0].chromosome_representation) + " Distance: " + str(round(generation[0].cost, 2)),
            end="")

        # vykreslenie grafu
        if i % config.ITERATION == 0:
            draw_graph(generation[0].chromosome, generation[0].chromosome_representation, i, generation[0].cost)

    return generation


# tabu search - vytvorenie susedov
def get_Neighbours(sBest):
    neighbours_chromosome = []
    reverse_neighbors = set()
    swap_neighbors = set()

    # Damocles - 50% pravdepodobnost
    Damocles = random.uniform(0, 1)
    if Damocles < 0.5:
        # ziskanie susedov pomocou swapu
        for i in range(1, len(sBest.chromosome) - 2):
            for j in range(i + 1, len(sBest.chromosome) - 1):
                neighbour = sBest.chromosome[:]
                neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
                swap_neighbors.add(tuple(neighbour))

    elif Damocles >= 0.5:
        # ziskanie susedov pomocou reverzu
        for i in range(1, len(sBest.chromosome) - 2):
            for j in range(i + 1, len(sBest.chromosome) - 1):
                neighbour = sBest.chromosome[:]
                neighbour[i:j + 1] = neighbour[i:j + 1][::-1]
                reverse_neighbors.add(tuple(neighbour))

    # ziskanie unikatnych susedov
    unique_neighbors = [list(neighbor) for neighbor in swap_neighbors.union(reverse_neighbors)]

    for neighbour in unique_neighbors:
        neighbours_chromosome.append(Chromosome(neighbour))

    return neighbours_chromosome


# tabu search - pseudocode z prednasky
def tabu_search(mesta, pop_size, TABU_SIZE):
    iteration = 1
    # inicializacia
    mesta = initialization(mesta, pop_size)[0]
    sBest = mesta
    tabu_list = [sBest]

    # vytvorenie novej generacie pomocou susedov
    while iteration <= config.NUMBER_OF_GENERATIONS:
        sNeighbour = get_Neighbours(sBest)
        best_candidate = sNeighbour[0]

        # porovnanie susedov s najlepsim kandidatom
        for neighbour in sNeighbour:
            if neighbour.cost < best_candidate.cost and neighbour not in tabu_list:
                best_candidate = neighbour

        # porovnanie najlepsieho kandidata s najlepsim doteraz
        if best_candidate.cost < sBest.cost:
            sBest = best_candidate
            tabu_list.append(sBest)

        # ak je tabu list plny, vymaze sa najstarsi prvok
        if len(tabu_list) > TABU_SIZE:
            tabu_list.pop(0)

        #
        print(
            f"\r" + str(iteration) + ". generation: " + "Chromosome " + str(
                sBest.chromosome_representation) + " Distance: " + str(round(sBest.cost, 2)),
            end="")

        # vykreslenie grafu
        if iteration % config.ITERATION == 0:
            draw_graph(sBest.chromosome, sBest.chromosome_representation, iteration, sBest.cost)

        iteration += 1
    return sBest


# vykreslenie grafu (https://matplotlib.org/stable/users/index.html)
def draw_graph(mesta, permutation, iteration, cost):
    x = []
    y = []

    # vytvorenie zoznamu suradnic pre vykreslenie
    for city_num in permutation:
        city = next(city for city in mesta if city.num == city_num)
        x.append(city.x)
        y.append(city.y)

    # parametre grafu
    fig, ax = plt.subplots()
    ax.grid(0)
    ax.plot(x, y, '--', lw=2, color='gray', ms=10)
    ax.set_xlim(-5, 205)
    ax.set_ylim(-5, 205)

    # anotacia miest
    for city in mesta:
        ax.annotate(city.num, (city.x + 3, city.y + 3), fontsize=15, color="black")
    ax.scatter(x, y, s=100, facecolor='C0', edgecolor='k')
    ax.set_title("Iteration: " + str(iteration) + " Cost: " + str(round(cost, 2)))
    plt.show()


# In your main section
if __name__ == '__main__':
    # inicializacia programu - vygenerovanie miest, nacitanie miest, vytvorenie matice vzdialenosti
    generate_points(config.NUMBER_OF_CITIES, config.MIN_DISTANCE)
    cities = load_points()
    matrix = create_distance_matrix(cities)

    # vyber algoritmu
    print("\nArtificial Intelligence - Project 2")
    print("David Truhlar - 12087 - Travelling Salesman Problem solver\n")
    print("Please choose the algorithm you want to use:")
    print("for 'Genetic algorithm' enter '1'\nfor 'Tabu search' enter '2'\nfor 'Exit' enter '3'\n")
    choice = input("Enter your choice: ")

    if choice == '1':
        solution = genetic_algorithm(cities, config.POPULATIOM_SIZE, config.MUTATION_RATE, config.NUMBER_OF_GENERATIONS)
        draw_graph(cities, solution[0].chromosome_representation, config.NUMBER_OF_GENERATIONS, solution[0].cost)
    elif choice == '2':
        solution = tabu_search(cities, config.POPULATIOM_SIZE, config.TABU_SIZE)
        draw_graph(cities, solution.chromosome_representation, config.NUMBER_OF_GENERATIONS, solution.cost)
    elif choice == '3':
        exit(0)
    else:
        print("Invalid input")
        exit(1)
