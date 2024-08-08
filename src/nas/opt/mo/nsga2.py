"""
File: searcher.py
Author: Jan Kastner
Date: February 29, 2024

Description:
    NSGAE:  wraps the individual and provides method important for NSGA-II algorithm.
    NSGA_II: represents NSGA_II algorithm.
"""

import os
import sys
import nas
import torchvision.transforms as transforms
import torchvision
import torch

# Get the current directory of the script
current_directory = os.path.dirname(os.path.realpath(__file__))

# Append the current directory to sys.path
sys.path.append(current_directory)

import random
import copy


class NSGAE:
    """
    Represents an element in a sort algorithm.

    Attributes:
        __individual (Individual): The individual associated with this element.

        fitness_value (list): List of fitness_value of the individual determined by the optimization strategy.

        __dominates (list): List of elements dominated by this element.

        __dominated (int): Number of elements dominating this element.

        __crowding_distance (float): The crowding distance of the element.

        __rank (int): The rank of the element.
    """

    def __init__(self, individual):
        """
        Initializes the  object.

        Parameters:
            individual (Individual): The individual associated with the element.
        """
        self.individual = individual
        self._fitness_value = individual.fitness_value
        self.__dominates = []
        self._dominated = 0
        self._crowding_distance = None
        self._rank = None

    def clear(self):
        self.__dominates = []
        self._dominated = 0
        self._crowding_distance = None
        self._rank = None

    def set_rank(self, rank):
        """
        Sets the rank of the element.

        Parameters:
            rank (int): The rank to be set.
        """
        self._rank = rank

    def set_crowding_distance(self, crowding_distance):
        """
        Sets the crowding distance of the element.

        Parameters:
            crowding_distance (float): The crowding distance to be set.
        """
        self._crowding_distance = crowding_distance

    def append_dominates(self, nsgae):
        """
        Appends an element to the list which contains elements dominated by this element.

        Parameters:
            nsgae (NSGAE): The dominated element to be appended.
        """
        self.__dominates.append(nsgae)

    def decrease_dominated(self):
        """Decreases the number of elements dominating this element."""
        self._dominated -= 1

    def increase_dominated(self):
        """Increases the number of elements dominating this element."""
        self._dominated += 1

    def decrease_dominates(self):
        """Erase domination by this element."""
        for el in self.__dominates:
            el.decrease_dominated()

    def __lt__(self, other):
        """
        Compares this element with another element based on rank and crowding distance.

        Parameters:
            other (NSGAE): The other element to be compared with.

        Returns:
            bool: True if this element is less than the other, False otherwise.
        """
        if self._rank != other.rank:
            return self._rank < other.rank
        else:
            return self._crowding_distance > other.crowding_distance

    def dominates(self, other):
        """
        Checks if this element dominates another element.

        Parameters:
            other (NSGAE): The other element to be checked.

        Returns:
            bool: True if this element dominates the other, False otherwise.
        """
        strict_less = False
        for idx in range(len(self.individual.fitness_value)):
            if self.individual.fitness_value[idx] < other.fitness_value[idx]:
                strict_less = True
            if self.individual.fitness_value[idx] > other.fitness_value[idx]:
                return False
        return strict_less

    @property
    def fitness_value(self):
        """
        Property to access the fitness value of the individual.

        Returns:
            list: List containing fitness value components.
        """
        return self.individual.fitness_value

    @property
    def rank(self):
        """
        Property to access rank.

        Returns:
            int: The rank.
        """
        return self._rank

    @property
    def crowding_distance(self):
        """
        Property to access the crowding distance.

        Returns:
            int: The crowding distance.
        """
        return self._crowding_distance

    @property
    def dominated(self):
        """
        Property to access the the number of elements dominating this element.

        Returns:
            int: The number of elements dominating this element.
        """
        return self._dominated


class NSGA_II:
    """
    Represents NSGA_II algorithm.

    Attributes:
        __population_size (int): The number of individuals in the population.

        __mating_pool_size (int): The number of sorted individuals to be used as parents for new
        population.

        __mutation_rate (float): The ration between the mutated alleles and number of alleles in the
        individual genotype.

        __parents (list): The list of individuals which will be used for creating new population.

        __population (list): The list of individuals which contains the individuals of the original population.

        __fronts (list): The list which contains individuals sorted by '__non_dominated_sorting' to pareto
        fronts.

        __offsprings (list): The list of the individuals which represents the new population.

    """

    def __init__(
        self,
        generator,
        termination_strategy,
        batch_size,
        dataset,
        num_epochs,
        mu_arg,
        lambda_arg,
        mutation_rate,
        training_size=20000,
        test_size=10000,
    ):
        """
        Evolves the population of individuals using evolutionary algorithms.

        Parameters:
            generator (Generator): The generator object responsible for creating individuals.

            optimalizator (NSGA_II): The optimizer object used for optimizing individuals.

            termination_strategy (callable): The termination strategy to determine when to stop evolution.

            batch_size (int): The batch size for data loading.

            dataset (torchvision.datasets): The dataset type, either CIFAR-10 or MNIST.

            num_epochs (int or list):   The number of epochs for training.
            If int, same number of epochs will be used for all generations.
            If list, each element corresponds to epochs for each generation.

        Raises:
            ValueError: If batch_size is not a positive integer,
                        If termination_strategy is not a method of Terminator class,
                        If not a valid optimalizator,
                        If generator is not from supported generators,
                        If num_epochs contains non-positive integers.
                        If mu_arg is not a positive integer.
                        If lambda_arg is not a positive integer.
                        If mutation_rate is not in the interval (0, 1>.

        """
        # Initialize loaders and parameters
        self.__trainloader = None
        self.__testloader = None
        self.__generator = None
        self.__termination_strategy = None
        self.__num_epochs = 0
        self.__mu_arg = 0
        self.__lambda_arg = 0
        self.__mutation_rate = 0

        # Stores inforamtions about populations
        self.__populations_info = []
        self.__best_solutions = []
        # Supported generators for network architectures
        self.__supported_generators = nas.population.cnn.Generator

        # Validate batch size
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")

        # Validate termination strategy
        if not (
            hasattr(termination_strategy, "__qualname__")
            and termination_strategy.__qualname__.startswith(nas.Terminator.__name__)
        ):
            raise ValueError(
                "The provided 'termination_strategy' is not valid. Please provide a 'termination_strategy'"
                + "belonging to the class 'Terminator'."
            )

        # Validate generator
        if not isinstance(generator, self.__supported_generators):
            raise ValueError("The provided 'generator' is not valid.")

        # Validate num_epochs
        if not (isinstance(num_epochs, int) and num_epochs >= 1):
            raise ValueError(
                "Only positive integers are allowed to specify number of epochs: '{}'.".format(
                    num_epochs
                )
            )

        if not isinstance(mu_arg, int) or mu_arg <= 0:
            raise ValueError("Number of parents must be a positive integer.")

        if not isinstance(training_size, int) or mu_arg <= 0:
            raise ValueError("Size of training dataset must be a positive integer.")

        if not isinstance(test_size, int) or mu_arg <= 0:
            raise ValueError("Size of testing dataset must be a positive integer.")

        if not isinstance(lambda_arg, int) or lambda_arg <= 0:
            raise ValueError("Number of descendants must be a positive integer.")

        if not (isinstance(mutation_rate, (int, float)) and 0 < mutation_rate <= 1):
            raise ValueError("Mutation rate must be in the interval (0, 1>.")

        # Prepare train and test loaders
        self.__trainloader, self.__testloader = self.__get_train_and_test_set(
            batch_size, dataset, train_size=training_size, test_size=test_size
        )
        self.__generator = generator
        self.__termination_strategy = termination_strategy
        self.__num_epochs = num_epochs
        self.__mu_arg = mu_arg
        self.__lambda_arg = lambda_arg
        self.__mutation_rate = mutation_rate

    def evolve(self):
        """
        Evolves the population over generations using the NSGA-II algorithm.

        Returns:
            populations_info (list): List containing information about the populations over generations.
        """

        # Reset populations info
        self.__populations_info = []

        offsprings = []

        parents = []

        # Generate initial individuals
        individuals = self.__generator.get(num_of_individuals=self.__mu_arg)

        # Initialize NSGAE objects for each individual
        for i in individuals:
            parents.append(NSGAE(i))

        # Evaluate fitness of initial generation
        parents = self.__evaluate(parents)

        # Evolution loop until termination criteria met
        while not self.__termination_strategy():
            # Reproduce offspring from parents
            offsprings = self.__replicate(parents)

            # Mutate each offspring
            for nsgae in offsprings:
                nsgae.individual.mutate(self.__mutation_rate)

            # Evaluate fitness of offspring population
            population = self.__evaluate(
                individuals=offsprings, evaluated_individuals=parents
            )

            # Sort population based on non-dominated sorting
            population = sorted(population)

            # Select top individuals for next generation
            parents = population[: self.__mu_arg]

            # Save information about current population
            self.__save(population=population)

        # Store information about final population
        for idx, nsgae in enumerate(population):
            self.__populations_info[-1][idx]["net"] = nsgae.individual.net

        return self.__populations_info

    def __non_dominated_sorting(self, individuals):
        """
        Perform non-dominated sorting to identify Pareto fronts.

        This method updates the internal attributes of the genetic algorithm instance
        based on the provided population.
        """
        for nsgae in individuals:
            nsgae.clear()
        fronts = []
        rank = 1
        # Calculate domination between individuals
        for nsgae_1 in individuals:
            for nsgae_2 in individuals:
                if nsgae_1 == nsgae_2:  # Skip comparing with itself
                    continue
                if nsgae_1.dominates(nsgae_2):
                    nsgae_1.append_dominates(nsgae_2)
                    nsgae_2.increase_dominated()

        # Perform sorting
        while True:
            fronts.append([])  # Create a new front
            for nsgae in individuals:
                # Pick up elements which are not dominated by none other elements
                if nsgae.dominated == 0:
                    fronts[-1].append(nsgae)  # Add NSGAE to the current front
                    # Set the rank of NSGAE based on current front
                    fronts[-1][-1].set_rank(rank)

            # Update domination counts and remove dominated individuals
            for nsgae in fronts[-1]:
                nsgae.decrease_dominates()
                individuals.remove(nsgae)
            rank += 1
            if len(individuals) <= 0:
                return fronts

    def __calculate_crowding_distance(
        self, left_values, right_values, min_values, max_values
    ):
        """
        Calculate the crowding distance for a given set of values.

        Parameters:
            left_values (list): List of fitness values of the individual on the left side compared to the
            current individual.

            right_values (list): List of fitness values of the individual on the right side compared to the
            current individual.

            min_values (list): List of minimum fitness values for normalization.

            max_values (list): List of maximum fitness values for normalization.

        Returns:
            float: The calculated crowding distance.
        """
        crowding_distance = 0

        for left, right, min_val, max_val in zip(
            left_values, right_values, min_values, max_values
        ):
            if max_val - min_val == 0:
                return 0
            crowding_distance += (right - left) / (max_val - min_val)

        return crowding_distance

    def __get_min_max_values(self, front):
        """
        Gets minimum and maximum values of each property in given front.

        Args:
            front (list): List of NSGAEs from witch minimum and maximum values of fitness_value will be
            extracted.

        Returns:
            list: List of minimum and maximum values of each propertie in given front.
        """
        num_values = len(front[0].fitness_value)
        min_values = [float("inf")] * num_values
        max_values = [float("-inf")] * num_values

        for nsgae in front:
            for i, property in enumerate(nsgae.fitness_value):
                min_values[i] = min(min_values[i], property)
                max_values[i] = max(max_values[i], property)

        return min_values, max_values

    def __crowding_distance_sorting(self, fronts):
        """
        Sorts elements by crowding distance.

        Returns:
            list: List of individuals which will be used for creating new population.
        """
        individuals = []
        for front in fronts:
            # Sorts front by first property
            front = sorted(front, key=lambda x: x.fitness_value[0])
            min_values, max_values = self.__get_min_max_values(front)

            # Marginal NSGAEs have infinity crowding distance
            # (NSGAEs with highest value of first property)
            individuals.append(front[0])
            individuals[-1].set_crowding_distance(float("inf"))

            # Calculates crowding distance for inner NSGAEs
            for i_idx, i in enumerate(front[1:-1]):
                crowding_distance = self.__calculate_crowding_distance(
                    front[i_idx - 1].fitness_value,
                    front[i_idx + 1].fitness_value,
                    min_values,
                    max_values,
                )
                individuals.append(i)
                individuals[-1].set_crowding_distance(crowding_distance)

            # Marginal NSGAEs have infinity crowding distance (NSGAEs with lowest value of first property)
            if len(front) <= 1:
                continue

            individuals.append(front[-1])
            individuals[-1].set_crowding_distance(float("inf"))

        # NSGAEs sorted by rank. When two compared NSGAEs have same rank, than crowding distance
        # is used.
        return sorted(individuals)

    def __replicate(self, individuals):
        """
        Perform tournament selection and mutation to generate offspring individuals.

        Returns:
            list: List of individuals which represents the new population.
        """

        offsprings = []

        # Tournament selection to generate offsprings
        for _ in range(self.__lambda_arg):
            nsgae_1 = random.choice(individuals)
            nsgae_2 = random.choice(individuals)
            # NSGAEs compared by rank when two compared individuals have same rank, than crowding
            # distance is used.
            if nsgae_1 < nsgae_2:
                offsprings.append(copy.deepcopy(nsgae_1))
            else:
                offsprings.append(copy.deepcopy(nsgae_2))

        return offsprings

    def __evaluate(self, individuals, evaluated_individuals=None):
        """
        Evaluates the fitness of individuals.

        Parameters:
            individuals (list): Individuals to be evaluated.
            evaluated_individuals (list, optional): Previously evaluated individuals to be considered alongside the new ones.
        """
        # Evaluate fitness of each individual
        for nsgae in individuals:
            nsgae.individual.evaluate(
                self.__trainloader, self.__testloader, self.__num_epochs
            )

        # Combine new and previously evaluated individuals
        if evaluated_individuals is None:
            population = individuals
        else:
            population = individuals + evaluated_individuals

        # Sort individuals based on non-dominated sorting and crowding distance
        return self.__crowding_distance_sorting(
            self.__non_dominated_sorting(population)
        )

    def __save(self, population):
        """
        Saves information about the current population.

        Parameters:
            population (list): List of individuals in the population.
        """
        # Initialize list to store information about each individual
        self.__populations_info.append([])
        # Save information about each individual in the population
        for nsgae in population:
            self.__populations_info[-1].append(
                {"graph_info": None, "fitness": None, "rank": None, "net": None}
            )
            self.__populations_info[-1][-1]["rank"] = nsgae.rank
            self.__populations_info[-1][-1]["graph_info"] = nsgae.individual.graph_info
            self.__populations_info[-1][-1]["fitness"] = nsgae.fitness_value

    def __evaluate(self, individuals, evaluated_individuals=None):
        """
        Evaluates individuals.

        Parameters:
            offsprings (list): Individuals to be evaluated.
        """
        for nsgae in individuals:
            nsgae.individual.evaluate(
                self.__trainloader, self.__testloader, self.__num_epochs
            )

        if evaluated_individuals is None:
            population = individuals
        else:
            population = individuals + evaluated_individuals

        return self.__crowding_distance_sorting(
            self.__non_dominated_sorting(population)
        )

    def __save(self, population):
        self.__populations_info.append([])
        for nsgae in population:
            self.__populations_info[-1].append(
                {"graph_info": None, "fitness": None, "rank": None, "net": None}
            )
            self.__populations_info[-1][-1]["rank"] = nsgae.rank
            self.__populations_info[-1][-1]["graph_info"] = nsgae.individual.graph_info
            self.__populations_info[-1][-1]["fitness"] = nsgae.fitness_value

    def __get_train_and_test_set(self, batch_size, dataset, train_size, test_size):
        """
        Prepares training and testing datasets and corresponding data loaders.

        Parameters:
        - batch_size (int): The batch size for data loading.
        - dataset (torchvision.datasets): The dataset type, either CIFAR-10 or MNIST.
        - train_size (int): The size of the training subset.
        - test_size (int): The size of the testing subset.

        Returns:
        - trainloader (torch.utils.data.DataLoader): DataLoader for the training set.
        - testloader (torch.utils.data.DataLoader): DataLoader for the testing set.

        Raises:
        - ValueError: If the dataset is not CIFAR-10 or MNIST.
        """
        if not (
            dataset == torchvision.datasets.CIFAR10
            or dataset == torchvision.datasets.MNIST
        ):
            raise ValueError(
                "Invalid dataset type. The dataset must be either CIFAR-10 or MNIST."
            )

        if dataset == torchvision.datasets.CIFAR10:
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]
            )

        elif dataset == torchvision.datasets.MNIST:
            mean = (0.1307,)
            std = (0.3081,)
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]
            )

        # Load the full training dataset
        full_trainset = dataset(
            root="./data", train=True, download=True, transform=transform
        )

        # Create a subset of the training dataset
        trainset = torch.utils.data.Subset(full_trainset, range(train_size))

        # Create DataLoader for the training set
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=3
        )

        # Load the full testing dataset
        full_testset = dataset(
            root="./data", train=False, download=True, transform=transform
        )

        # Create a subset of the testing dataset
        testset = torch.utils.data.Subset(full_testset, range(test_size))

        # Create DataLoader for the testing set
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=3
        )

        return trainloader, testloader
