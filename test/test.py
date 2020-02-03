#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 23:06:36 2020

@author: tungutokyo
"""

import pytest
import requests


def test_metadata():

    model_endpoint = 'http://localhost:8000/api/v1/'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    