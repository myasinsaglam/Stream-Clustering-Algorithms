import math
from sklearn.metrics import pairwise_distances
from collections import defaultdict
# import numpy as np
import random
import numpy as np


class UtilInterface:
    def __init__(self, *args, **kwargs):
        """
        distance metrics : l1 or manhattan , l2 or euclidean
        @param args:
        @param kwargs:
        """

        if "distance_metric" not in kwargs:
            self.distance_metric = "euclidean"
        else:
            self.distance_metric = kwargs["distance_metric"]

    def gaussian_neighbourhood(self, x_input, c_input, radius):
        """
        A method that calculates gaussian neighbourhood.
        @param x_input: Input 1
        @type x_input: float
        @param c_input: Input 2
        @type c_input: float
        @param radius: radius parameter
        @type radius: float
        @return:
        """
        distance = self.distance_vector([x_input], [c_input])
        return math.exp(-(((distance ** 2) / 3 * radius) ** 2) / 2)

    def distance_vector(self, v1, v2):
        """
        A method that calculates distance between given vectors
        @param v1: vector 1
        @param v2: vector 2
        @return:
        """

        # if type(v1).__name__ != "list" or type(v2).__name__ != "list":
        #     return pairwise_distances(list(v1), list(v2), metric=self.distance_metric)[0][0]
        # #     return pairwise_distances([v1], [v2], metric=self.distance_metric)[0][0]
        # else:
        return pairwise_distances([v1], [v2], metric=self.distance_metric)[0][0]

    def roulette_wheel(self, array, n):
        """

        A method that implements generic version of roulette wheel selection algorithm
        @type array: list
        @param array: Value array
        @type n: int
        @param n: number of elements that will be selected
        @rtype: list
        @return: list of selected elements

        """
        result = []
        freqs = defaultdict(int)
        for item in array:
            freqs[item] += 1
        total = sum(freqs.values())
        tmp = 0
        for k, v in freqs.items():
            freqs[k] /= total
            freqs[k] += tmp
            tmp = freqs[k]
        i = 0
        k = list(freqs.keys())[i]
        rand = random.random()
        while n:
            if rand <= freqs[k]:
                result.append(k)
                n -= 1
                rand = random.random()
                i = 0
                k = list(freqs.keys())[i]
                continue
            i += 1
            i %= len(freqs)
            k = list(freqs.keys())[i]
        return result

    def fitness_roulette_wheel(self, fitness_array):
        """
        A method that implements spesific version of roulette wheel selection algorithm
        @param fitness_array: Value array
        @type fitness_array: list
        @return: list of selected elements
        @rtype: list
        """
        ## fix sized - number of elements that will be selected
        n = 2
        result = []

        freqs = {}
        for i in range(len(fitness_array)):
            freqs[i] = fitness_array[i]

        total = sum(freqs.values())
        tmp = 0
        for k, v in freqs.items():
            freqs[k] /= total
            freqs[k] += tmp
            tmp = freqs[k]

        i = 0
        # k = list(freqs.keys())[i]
        rand = random.random()
        while n and len(freqs) > 0:
            key = list(freqs.keys())[i]
            if freqs[key] > rand:
                # print("ID : ", i, " Freqs: ", freqs, " Rand: ", rand)
                result.append(key)
                n -= 1
                total -= fitness_array[key]
                del freqs[key]
                # print("Deleted", freqs)

                tmp = 0
                for k, v in freqs.items():
                    freqs[k] = fitness_array[k]/total
                    freqs[k] += tmp
                    tmp = freqs[k]

                rand = random.random()
                i = 0
            else:
                i += 1

            # k = list(freqs.keys())[i]
        return result

    def find_min_indexes(self, arr, abs_mode=True, min_count=1):
        """
        A method that find minimum valued indexes(defined size in min_count variable) of given array. It can run on abs mode too.
        @param arr: Array
        @type arr: list
        @param abs_mode: Abs mode option
        @type abs_mode: bool
        @param min_count: Min index number that will be returned
        @type min_count: int
        @return: Minimum valued index or indexes
        @rtype: list
        """
        if abs_mode:
            arr = np.abs(arr)
        sorted_mapping = sorted([*enumerate(arr)], key=lambda x: x[1])
        return [item[0] for item in sorted_mapping[:min_count]]


# utils = UtilInterface(distance_metric="l2")
#
# # print(utils.gaussian_neighbourhood(1, 2, 5))
# # print(utils.distance_vector([6], [1]))
# # print(utils.roulette_wheel([1, 1, 2, 2, 2, 3, 3, 333, 3, 4, 5, 1], 30))
# arr = [2, 3, 3, 2]
# arr = [19, 1, 2, 2, 2, 3, 3, 333, 3, 4, 5, 199]
# print(utils.fitness_roulette_wheel(arr))
# print(arr)
