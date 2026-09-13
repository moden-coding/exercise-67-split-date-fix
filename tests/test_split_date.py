#!/usr/bin/env python3

import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from src.split_date import split_date, main


class SplitDate(unittest.TestCase):

    def setUp(self):
        self.df = split_date()

    def test_shape(self):
        self.assertEqual(self.df.shape, (37128, 5),
                         msg="The DataFrame has incorrect shape!")

    def test_columns(self):
        np.testing.assert_array_equal(self.df.columns,
                                      ["Weekday", "Day", "Month", "Year", "Hour"],
                                      err_msg="Incorrect column names!")

    def test_dtypes(self):
        correct_types = [object, np.integer, np.integer, np.integer, np.integer]
        for i, (result, correct) in enumerate(zip(self.df.dtypes, correct_types)):
            self.assertTrue(np.issubdtype(result, correct),
                            msg="Types don't match on column %i! Expected %s got %s." % (i, correct, result))

    def test_called(self):
        with patch("src.split_date.pd.read_csv", wraps=pd.read_csv) as prc,\
             patch("src.split_date.split_date", wraps=split_date) as psd:
            main()
            psd.assert_called()

    def test_content(self):
        weekdays = "Mon Tue Wed Thu Fri Sat Sun".split()
        for elem in self.df["Weekday"]:
            self.assertIn(elem, weekdays, msg="Incorrect value '%s' in column Weekday!" % elem)

        for index in self.df.index:
            weekday, day, month, year, hour = self.df.loc[index]
            self.assertIn(weekday, weekdays, msg="Incorrect value '%s' in column Weekday!" % weekday)
            self.assertIn(day, range(1,32), msg="Incorrect value '%s' in column Day!" % day)
            self.assertIn(month, range(1,13), msg="Incorrect value '%s' in column Month!" % month)
            self.assertIn(year, range(2014,2019), msg="Incorrect value '%s' in column Year!" % year)
            self.assertIn(hour, range(0,24), msg="Incorrect value '%s' in column Hour!" % hour)

if __name__ == '__main__':
    unittest.main()
