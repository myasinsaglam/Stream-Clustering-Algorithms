"""
@author:

Name: M.Yasin SAGLAM
E-Mail: saglam.yasin.m@gmail.com
Github: https://github.com/myasinsaglam

@version: 1
"""

from abc import ABC


class EvoStream(ABC):
    """
    Abstract base class for EvoStream Algorithm
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
