"""
@author:

Name: M.Yasin SAGLAM
E-Mail: saglam.yasin.m@gmail.com
Github: https://github.com/myasinsaglam

@version: 1

@todo: In v2; Cluster number k will be dynamic value, Increment and decrement policy update
"""

import sys
sys.path.append(".")
from evoStream.Abstracts.EvoStream import EvoStream
from evoStream.v1.Classes.MicroClusterV1 import MicroClusterV1 as MicroCluster
from evoStream.v1.Utils.Utils import UtilInterface
import numpy as np
import random

## Debug mode macro
DEBUG = False


class EvoStreamV1(EvoStream):

    def __init__(self, *args, **kwargs):
        """
        Construtor of EvoStreamV1 class
        Time variant, adaptive, evolutional stream clustering algorithm
        @param args: Optional
        @param kwargs: radius, decay_rate, cleanup_interval,
        number_of_clusters, crossover_rate, mutation_rate, population_size, initialize_after, recluster_generations
        """
        super().__init__()
        try:
            ## Radius value for cluster
            self.radius = kwargs["radius"]
            ## Decay rate
            self.decay_rate = kwargs["decay_rate"]
            ## Cleanup interval
            self.cleanup_interval = kwargs["cleanup_interval"]
            ## Number of clusters
            self.k = kwargs["number_of_clusters"]
            ## Omega calculation
            self.omega = 2 ** (-self.decay_rate * self.cleanup_interval)

            ## GA PARAMETERS
            ## Crossover rate
            self.crossover_rate = kwargs["crossover_rate"]
            ## Mutation rate
            self.mutation_rate = kwargs["mutation_rate"]
            ## Population size
            self.population_size = kwargs["population_size"]
            ## Initalization after x number of inputs
            self.initialize_after = kwargs["initialize_after"]
            ## Recluster generation interval parameters
            self.recluster_generations = kwargs["recluster_generations"]
            # self.initialization_threshold = kwargs["initialization_threshold"]

        except Exception as e:
            print(e)
            raise e

        self.dict = kwargs
        self.init = False
        self.init_time = 0
        self.warming_current = 0
        self.warming_instance_count = 100

        self.uptodate = 0
        self.current_time = 0

        self.macro_fitness = [0] * self.population_size

        self.micro_clusters = []
        self.macro_clusters = []
        self.utils = UtilInterface()

        # TODO: Time unit and current time variables structure will be determined and implemented as kwargs
        # TODO: Data Connector Class means Stream will be implemented
        # TODO: Implementation of run method and any other return methodologies

    def update_cleanup_interval(self, cleanup_interval):
        """MODULE FOR CLEAN-UP ADAPTIVITY"""
        self.cleanup_interval = cleanup_interval
        self.omega = 2 ** (-self.decay_rate * self.cleanup_interval)

    def update_weights(self):
        """
        DONE
        :return:
        """
        updated_micro_clusters = []
        for mc in self.micro_clusters:
            mc.fade(self.current_time, self.decay_rate)
            if mc.weight > self.omega:
                updated_micro_clusters.append(mc)
        self.micro_clusters = updated_micro_clusters

    def clean_up(self):
        """
        DONE
        A method that cleans up expired(low weighted) microclusters and merges relevant ones
        :return:
        """
        # if DEBUG:
        #     print("Clean-Up Phase...")
        # print("Clean-Up Phase...")

        # Updating weights - microclusters are being older and fixed same update time, low weighted microclusters  are
        # eliminated.
        self.update_weights()

        # Merging close micro clusters by cleaning newest microclusters.
        mc_size = len(self.micro_clusters)
        for i in range(mc_size - 1, 0, -1):
            merge = False
            j = i - 1
            while merge is False and j >= 0:
                dist = self.micro_clusters[i].distance(self.micro_clusters[j])
                # print(i, j, dist)
                if dist <= self.radius:
                    # if DEBUG:
                    #     print("Merging : ", i, ",", j)
                    self.micro_clusters[j].merge(self.micro_clusters[i], self.current_time, self.decay_rate,
                                                 self.radius)
                    del self.micro_clusters[i]
                    merge = True
                else:
                    j -= 1

    def get_distance_vector(self, mc):
        """
        A method that calculates distance vector from given temporary microcluster to all micro cluster.
        (Finding nearest microcluster)
        @param mc: Microcluster object
        @return: Distance vector
        """
        distances = []
        for item in self.micro_clusters:
            distances.append(mc.distance(item))
        return distances
        # for mc in self.micro_clusters:
        #     mc.weight = mc.weight * (2 ** (-self.decay_rate * (self.current_time - mc.t)))
        # self.micro_clusters = [mc for mc in self.micro_clusters if not mc.weight <= 2 ** (-self.cleanup_interval)]
        # TODO: Merge step implementation of clusters within radius

    def insert(self, distances, mc):
        """
        A method that absorbs observation or creates new microcluster.
        @param distances: Distance array to other microclusters
        @param mc: New coming microcluster object.
        @return: None
        """
        merged = False
        for i in range(len(self.micro_clusters)):
            if distances[i] <= self.radius:
                # if DEBUG:
                #     print("Time : ", self.current_time, " Observation merging with Micro Cluster : ", i)
                self.micro_clusters[i].merge(mc, self.current_time, self.decay_rate, self.radius)
                merged = True
        if not merged:
            # if DEBUG:
            #     print("Time: ", self.current_time, "Observation evaluated as new Micro Cluster")
            self.micro_clusters.append(mc)

    def cluster(self, input_data):
        """
        A method that creates new microcluster from given input data and inserts to set according to principles of algorithm.
        @param input_data: Vector, value, centroid  etc.
        @return: None
        """
        self.uptodate = 0
        self.current_time += 1

        # Creating temporary micro cluster.
        mc = MicroCluster(centroid=input_data, last_update=self.current_time)
        distances = self.get_distance_vector(mc)
        self.insert(distances, mc)

        if self.current_time % self.cleanup_interval == 0:
            self.clean_up()

        if self.init is False and len(self.micro_clusters) == self.initialize_after:
            self.initialize()

    def initialize(self):
        """
        Randomly chosen MACRO cluster POPULATION initialization according to defined population size.
        @return: None
        """
        self.init_time = self.current_time
        micro_cluster_size = len(self.micro_clusters)
        choose = list(range(0, micro_cluster_size))

        for i in range(self.population_size):
            self.macro_clusters.append([])
            random.shuffle(choose)
            for j in range(self.k):
                index = j % micro_cluster_size
                self.macro_clusters[i].append(self.micro_clusters[choose[index]].centroid)
        self.init = 1

    """
    INTERFACE MODULES
    """

    def get_microclusters(self):
        return [item.centroid for item in self.micro_clusters]

    def get_microweights(self):
        return [item.weight for item in self.micro_clusters]

    def get_macroclusters(self):

        if not self.init:
            return None

        if self.recluster_generations != 0 and self.uptodate == 0:
            self.recluster(self.recluster_generations)
            self.uptodate = 1

        maxindex = np.argmax(np.array(self.macro_fitness))
        max_fitness = max(self.macro_fitness)
        # print("Max fittest macro cluster id : ", maxindex, " Fitness value: ", max_fitness)
        return self.macro_clusters[maxindex]

    def get_macroweights(self):
        if not self.init:
            return 0
        if self.recluster_generations != 0 and self.uptodate == 0:
            self.recluster(self.recluster_generations)
            self.uptodate = 1

        cluster_assignments, cluster_distances = self.micro_to_macro()
        microweights = self.get_microweights()
        macro_weights = [0] * self.k
        for i in range(len(cluster_assignments)):
            macro_weights[cluster_assignments[i]] += microweights[i]

        return macro_weights

    def micro_to_macro(self):
        if not self.init:
            return 0
        centres = self.get_macroclusters()
        return self.get_assignment(centres)

    def evolution(self):
        if not self.init:
            return

        # Macro fitness evolution
        self.calculate_fitness()

        selected = self.selection()
        offsprings = self.crossing_over(selected)
        mutants = self.mutation(offsprings)

        # mindex = np.argmin(np.array(self.macro_fitness))
        # minval = min(self.macro_fitness)
        # print("\n\n\n\n\nMAX FITNESS : ", max(self.macro_fitness))
        # print("MAX INDEX : ", np.argmax(np.array(self.macro_fitness)))
        # print("MACRO FITNESSES : ", self.macro_fitness)
        for i in range(len(mutants)):
            fit = self.fitness(mutants[i])
            mindex = np.argmin(np.array(self.macro_fitness))
            minval = min(self.macro_fitness)
            if minval < fit:
                # print("Mindex :", mindex, " Mutant : ", i, " Fitness: ", fit)
                self.macro_clusters[mindex] = mutants[i]
                self.macro_fitness[mindex] = fit
            # print("After : ", self.macro_fitness[mindex])

        # print("Best fitness is : ", max(self.macro_fitness))

    def recluster(self, generations):
        if not self.init:
            return

        for i in range(generations):
            # print("Generation : ", i)
            self.evolution()

    """
    HELPER MODULES
    """

    def get_assignment(self, centres):
        """
        A method that returns nearest center values for each micro clusters
        MICRO to MACRO ASSIGNMENT INDEXES
        @param centres:
        @return:
        """

        assignments = []
        assignment_distances = []
        for mc in self.micro_clusters:
            distances = np.array([mc.distance_vector(center) for center in centres])
            mindex = np.argmin(distances)
            assignments.append(mindex)
            assignment_distances.append(distances[mindex])
        return assignments, assignment_distances

    def fitness(self, centres):
        """
        MICRO to MACRO ASSIGNMENT INDEXES
        @param centres:
        @return:
        """
        fitness_value = 0.0
        assignments, assignment_distances = self.get_assignment(centres)
        for i in range(len(self.micro_clusters)):
            # TODO: REVISE FITNESS FUNCTION
            fitness_value += (assignment_distances[i] ** 2) * self.micro_clusters[i].weight
        return 1 / fitness_value

    def calculate_fitness(self):
        """
        FOR EVERY MACRO CLUSTER(k,input_vector_length) CALCULATE FITNESS
        RESULTS ARE ASSIGNED TO POPULATION(ALL MACRO CLUSTERS) FITNESS VARIABLE
        @return:
        """
        self.macro_fitness = [self.fitness(macro_centers) for macro_centers in self.macro_clusters]
        # print(len(self.macro_fitness))

    def selection(self, algorithm="roulette"):
        if algorithm == "roulette":
            indexes = self.utils.fitness_roulette_wheel(self.macro_fitness)
            return [self.macro_clusters[i] for i in indexes]

    def crossing_over(self, selections, algorithm="nebilimolumolcmedimki"):
        """
        The method that makes cross-over on selected fittest 2 macroclusters.
        @param selections: Selected gene indexes
        @param algorithm: future improvement alg switching
        @return:
        """
        rand = random.random()
        if rand < self.crossover_rate:
            nrow = len(selections[0])
            ncolumn = len(selections[0][0])
            size = nrow * ncolumn
            # Random crossover point
            crossover_point = int((size - 1) * random.random())
            pos = 0
            for row in range(nrow):
                for column in range(ncolumn):
                    if pos > crossover_point:
                        selections[0][row][column], selections[1][row][column] = selections[1][row][column], \
                                                                                 selections[0][row][column]
                    pos += 1
        return selections

    def mutation(self, selections):
        """
        A method that mutates selection values
        @param selections: Selected chromosomes
        @return:
        """
        nrow = len(selections[0])
        ncolumn = len(selections[0][0])
        # if DEBUG:
        #     print("SELECTIONS : ", selections)
        for i in range(len(selections)):
            for j in range(nrow):
                for k in range(ncolumn):
                    rand = random.random()
                    if rand < self.mutation_rate:
                        if selections[i][j][k] != 0:
                            val = selections[i][j][k] * 2 * random.random()
                        else:
                            val = random.random() * 2

                        if random.random() < 0.5:
                            selections[i][j][k] += val
                        else:
                            selections[i][j][k] -= val
        return selections

    """
    RUNNER MODULES
    """

    def predict(self, dataset):
        """
        A method that predicts cluster id according to existing centroids in time, relatively .
        @param dataset:
        @return:
        """

        assignment_distances = []
        assignments = []
        centroids = self.get_macroclusters()

        for data in dataset:
            distances = np.array([self.utils.distance_vector(data, centroid) for centroid in centroids])
            mindex = np.argmin(distances)
            assignments.append(mindex)
            assignment_distances.append(distances[mindex])

        return assignments

    def fit(self, dataset):
        """
        Training
        @param dataset:
        @return:
        """

        for i, item in enumerate(dataset):
            self.cluster(item)
            # self.recluster(1)

        self.recluster(self.recluster_generations)
        self.uptodate = 1

        # if i % 2000:
        #     print("Reclustering")
        #     self.recluster(5)

        # centroids = self.get_macroclusters()

        # print("\nMicrocluster length: ", len(self.evo.micro_clusters))
        # print("\nMacro clusters : ", self.evo.get_macroclusters())
        # print("MACRO CLUSTERS : ", self.get_macroclusters())
        # print("Assignments : ", self.evo.micro_to_macro())
