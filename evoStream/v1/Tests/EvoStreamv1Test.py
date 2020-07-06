"""
@package EvoStreamv1Test


@author: Name: M.Yasin SAGLAM, E-Mail: saglam.yasin.m@gmail.com, Github: https://github.com/myasinsaglam
@version: 1
@note: EvostreamV1 class unittests of python implementation of evoStream Algorithm
"""

import sys
sys.path.append(".")
from evoStream.v1.Algorithm.EvoStreamV1 import EvoStreamV1 as EvoStream
import unittest
import numpy as np
import random
from matplotlib import pyplot as plt
from itertools import cycle, islice

np.random.seed(12)


class Test(unittest.TestCase):

    def setUp(self) -> None:
        """
        Setting up algorithm object and observations for test methods.
        @return:
        """
        self.evo = EvoStream(radius=0.005,
                             decay_rate=0.0004,
                             cleanup_interval=10,
                             number_of_clusters=4,
                             crossover_rate=0.8,
                             mutation_rate=0.001,
                             population_size=10,
                             initialize_after=2 * 4,
                             recluster_generations=10)

        self.observation = [list(np.random.sample(2)) for i in range(100)]
        # print(self.observation)

    def tearDown(self) -> None:
        pass

    def test_pipeline(self):
        """
        Pipeline test of clustering mechanism
        @return:
        """
        try:
            for item in self.observation:
                self.evo.cluster(item)
                self.evo.recluster(1)
                # print("\nMicrocluster length: ", len(self.evo.micro_clusters))
                # print("\nMacro clusters : ", self.evo.get_macroclusters())
            self.evo.get_macroclusters()
            self.evo.micro_to_macro()
            # print("MACRO CLUSTERS : ", self.evo.get_macroclusters())
            # print("Assignments : ", self.evo.micro_to_macro())
            self.assertEqual(True, True)
        except Exception as e_pipeline:
            print(e_pipeline)
            self.assertEqual(False, True)

        # print(self.evo.get_microclusters())
        # self.evo.get_microweights()
        # self.evo.get_macroclusters()
        # self.evo.get_macroweights()
        # self.evo.recluster(10)
        # self.evo.micro_to_macro()

    def test_predictions(self):
        """
        Test of algorithm predictions. If the colored dots in the scatter plot are seperated. Primitive PoC is done.
        @return:
        """
        dataset = []
        plot_set = []
        for i in range(1000):
            # dataset.append([random.random(), random.random()])
            dataset.append([random.randint(0, 3), random.randint(4, 7)])
            dataset.append([random.randint(0, 6), random.randint(4, 9)])
            dataset.append([random.randint(-3, -1), random.randint(-7, -7)])
            dataset.append([random.randint(77, 102), random.randint(-40, 0)])
        # print(dataset)

        plot_set = [item[0] for item in dataset]

        # add black color for outliers (if any)
        self.evo.fit(dataset)
        labels = self.evo.predict(dataset)
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(labels) + 1))))
        colors = np.append(colors, ["#000000"])
        print(colors)
        x = [item[0] for item in dataset]
        y = [item[1] for item in dataset]
        plt.scatter(x, y, s=10, color=colors[labels])
        plt.show()


    # def test_args(self):
    #     print(self.mc.arg)


if __name__ == '__main__':
    unittest.main()
