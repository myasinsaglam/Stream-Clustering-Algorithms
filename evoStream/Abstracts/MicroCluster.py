"""
@author:

Name: M.Yasin SAGLAM
E-Mail: saglam.yasin.m@gmail.com
Github: https://github.com/myasinsaglam

@version: 1
"""

from abc import ABC


class MicroCluster(ABC):
    """
    Abstract base class for MicroCluster
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def jsonify(self):
        pass

    def fade(self, current_time, decay_rate):
        """
        A base method for aging strategy
        @param current_time: Current Time id
        @type current_time: int
        @param decay_rate: Decay rate
        @type decay_rate: float
        @return:
        """
        pass

