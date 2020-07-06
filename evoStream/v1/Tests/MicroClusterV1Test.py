"""
@author:

Name: M.Yasin SAGLAM
E-Mail: saglam.yasin.m@gmail.com
Github: https://github.com/myasinsaglam

@version: 1
@note: MicroclusterV1 class unittests of python implementation of evoStream Algorithm
"""

import sys
sys.path.append(".")
from evoStream.v1.Classes.MicroClusterV1 import MicroClusterV1 as MicroCluster
import unittest


class Test(unittest.TestCase):

    def setUp(self):
        """
        Microcluster object initialization for test methods
        @return: None
        """
        # self.current_time = 5
        self.mc = MicroCluster(centroid=[0, 1, 2], weight=1, last_update=2)
        self.mc2 = MicroCluster(centroid=[12, 6, 2], weight=1, last_update=2)

        self.radius = 14
        self.current_time = 3
        self.decay_rate = 0.005

    def tearDown(self) -> None:
        pass

    # self.mc = MicroClusterV1(4, 0, 1, 0, 44, x=4, t=0, w=1)

    def test_microcluster_initialization(self):
        """
        Test for microcluster initialization
        @return:
        """
        unittest.TestCase.assertDictEqual(self, self.mc.jsonify(),
                                          {'centroid': [0, 1, 2], 'weight': 1, 'last_update': 2},
                                          msg="Construction of MicroCluster are done.")

    def test_microcluster_vector_distance(self):
        """
        Test for microcluster vector distance
        @return: boolean test result with message
        """
        unittest.TestCase.assertEqual(self, self.mc.distance_vector([3, 5, 2]), 5,
                                      msg="Vector based distance calculation is successful")

    def test_microcluster_object_distance(self):
        """
        Test for microcluster object distance
        @return: boolean test result with message
        """
        unittest.TestCase.assertEqual(self, self.mc.distance(self.mc2), 13,
                                      msg="Object based distance calculation is successful")

    def test_microcluster_gaussian_neighbourhood(self):
        """
        Test for gaussian neighbourhood
        @return: boolean test result with message
        """
        distance = self.mc.distance(self.mc2)

        gn = self.mc.gaussian_neighbourhood(distance=distance, radius=self.radius)

        unittest.TestCase.assertEqual(self, gn, 0.020648718062161137,
                                      msg="Object based distance calculation is successful")

    def test_microcluster_fade(self):
        """
        Test for fading microcluster
        @return: boolean test result with message
        """
        self.mc.fade(current_time=self.current_time, decay_rate=self.decay_rate)

        result = [self.mc.last_update, self.mc.weight]

        print(result)
        unittest.TestCase.assertListEqual(self, result, [3, 0.9965402628278678])

    def test_microcluster_same_fade(self):
        """
        Test for fading microcluster
        @return: boolean test result with message
        """
        self.mc.fade(current_time=2, decay_rate=self.decay_rate)

        result = [self.mc.last_update, self.mc.weight]

        # print("FADED WITH SAME TIME : ", result)

        unittest.TestCase.assertListEqual(self, result, [2, 1])

    def test_microcluster_merge(self):
        """
        Test for merge microclusters
        @return: boolean test result with message
        """
        for i in range(10):
            self.current_time += 1
            distance = self.mc.distance(self.mc2)
            print(i, distance, self.radius)
            if distance <= self.radius:
                print("Merging")
                self.mc.merge(self.mc2, self.current_time, self.decay_rate, self.radius)
                print(self.mc.jsonify())

        # real = {'centroid': [3.895829608300197, 2.6232623367917487, 2.0], 'last_update': self.current_time, 'weight': 1.659753955386447}
        # unittest.TestCase.assertDictEqual(self, self.mc.jsonify(), real)


if __name__ == '__main__':
    unittest.main()
