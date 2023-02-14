"""
python -m unittest pte_design/test_select_ptes.py
"""
import sys
import unittest
import numpy as np
import pandas as pd

# import pytest
import pte_design

from pte_design.seqtools import all_mers


"""TODO: load data from ../data and prepare epmers and pepmers for selection"""

class TestSelectPTEs(unittest.TestCase):
    def test_select(self):

        res = pte_design.select_ptes(epmers_df, pepmers_df, K=50, mm_tolerance=0)

        self.assertTrue((drect_dots1 == drect_dots2).all())

if __name__ == '__main__':
    unittest.main()
