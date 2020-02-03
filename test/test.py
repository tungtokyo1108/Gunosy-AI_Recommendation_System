#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 23:06:36 2020

@author: tungutokyo
"""

import pytest
import requests


def test_metadata():

    model_endpoint = 'http://0.0.0.0:8000/api/v1/'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    metadata = r.json()
    assert metadata['endpoints'] == 'http://0.0.0.0:8000/api/v1/endpoints'
    assert metadata['mlalgorithms'] == 'http://0.0.0.0:8000/api/v1/mlalgorithms'
    assert metadata['mlalgorithmstatuses'] == 'http://0.0.0.0:8000/api/v1/mlalgorithmstatuses'
    assert metadata['mlrequests'] == 'http://0.0.0.0:8000/api/v1/mlrequests'
    
if __name__ == '__main__':
    pytest.main([__file__])