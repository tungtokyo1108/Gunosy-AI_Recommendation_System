#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:55:29 2020

@author: tungutokyo
"""

from django.test import TestCase
import inspect
from apps.ml.registry import MLRegistry 

from apps.ml.Gunosy_classifier.NaiveBayes import NaiveBayes

class MLTests(TestCase):
    def test_nb_algorithm(self):
        input_data = "https://gunosy.com/articles/Rue0f" 
        my_algo = NaiveBayes()
        response = my_algo.compute_prediction(input_data)
        self.assertEqual('OK', response["status: "])
        self.assertTrue("Group_1st: " in response)
        self.assertEqual('エンタメ', response["Group_1st: "])
        
    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "Gunosy_classifier"
        algorithm_object = NaiveBayes()
        algorithm_name = "Naive Bayes"
        algorithm_status = "production"
        algorithm_version = "1.0"
        algorithm_owner = "Tung Dang"
        algorithm_description = "Navie Bayes with NLP to classify news article"
        algorithm_code = inspect.getsource(NaiveBayes)
        
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)
        