"""
@author:

Name: M.Yasin SAGLAM
E-Mail: saglam.yasin.m@gmail.com
Github: https://github.com/myasinsaglam

@version: 1
"""

import sys
sys.path.append(".")
from evoStream.Abstracts.MicroCluster import MicroCluster
from evoStream.v1.Utils.Utils import UtilInterface
import math


class MicroClusterV1(MicroCluster):

    def __init__(self, *args, **kwargs):
        """
        Constructor of MicroClusterV1 class.
        @param args:
        @param kwargs:
        """
        super().__init__()

        ## Class configuration kwargs
        self.config = kwargs
        ## Centroid for microcluster initialization
        self.centroid = kwargs["centroid"]

        if type(self.centroid).__name__ not in ["list", "ndarray"]:
            print("Type", type(self.centroid).__name__)
            self.centroid = [self.centroid]
            # print(self.centroid)
        self.last_update = kwargs["last_update"]
        if "weight" in kwargs:
            self.weight = kwargs["weight"]
        else:
            self.weight = 1

        ## Utility interface modules
        self.utils = UtilInterface()

    def jsonify(self):
        return {"centroid": self.centroid, "last_update": self.last_update, "weight": self.weight}

    def fade(self, current_time, decay_rate):
        """
        Decay weight according to elapsed time, microcluster is being older
        @param current_time: current time step
        @param decay_rate: weight decay rate
        @return:
        """

        ## Weight update formula
        self.weight *= (2 ** (-decay_rate * (current_time - self.last_update)))
        ## Time update
        self.last_update = current_time

    def merge(self, mc, current_time, decay_rate, radius):
        """
        Absorb new observation by merging centroids according to gaussian neighbourhood coefficient(calc. acc. to radius and distance)
        @param mc: microcluster object
        @param current_time: current time step
        @param decay_rate: weight decay rate
        @param radius:
        @return: None, Updated microcluster centroids as inplace
        """

        ## OPTIMIZATION
        ## Weight decay of new instance but its unnecessary bec. result is always 1. initial weights is 1 too.
        ## mc.fade(current_time, decay_rate)
        if mc.last_update != current_time:
            print("Time diff mc ", mc.last_update, current_time)

        self.fade(current_time, decay_rate)

        self.weight += mc.weight

        distance = self.utils.distance_vector(v1=self.centroid, v2=mc.centroid)
        gn_coeff = self.gaussian_neighbourhood(distance, radius)

        # print("Gn Coeff : ", gn_coeff, "Distance: ", distance)
        for i in range(len(self.centroid)):
            self.centroid[i] += gn_coeff * (
                    mc.centroid[i] - self.centroid[i])

    def gaussian_neighbourhood(self, distance, radius):
        """
        A method that calculates gaussian neighbourhood coefficient according to distance of two vectors and radius value
        @param distance: distance between to vectors (default metric is euclidean neighbourhood)
        @type distance: float, list
        @param radius: radius value
        @type radius: float
        @return: gaussian neighbour coefficient
        @rtype: float
        """
        return math.exp(-((distance / radius * 3) ** 2) / 2)

    def distance(self, mc):
        """
        Object based distance calculation wrapper, default distance metric is euclidean
        @param mc: microcluster object
        @type mc : MicroClusterV1
        @return: distance vector
        @rtype : list
        """
        return self.utils.distance_vector(v1=self.centroid, v2=mc.centroid)

    def distance_vector(self, vector):
        """
        Vector based distance calculation wrapper method
        @param vector: same sized input vector
        @type vector: list
        @return: distance vector
        @rtype: list
        """
        if len(vector) != len(self.centroid):
            raise Exception("Vector size mismatch !!! Expected: ", len(self.centroid), "Current", len(vector))
            # return Exception("")
        else:
            # print("Centroid", self.centroid, "Vector", vector)
            return self.utils.distance_vector(self.centroid, vector)
