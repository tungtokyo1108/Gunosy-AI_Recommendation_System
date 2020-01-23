"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

application = get_wsgi_application()

import inspect
from apps.ml.registry import MLRegistry
from apps.ml.Gunosy_classifier.NaiveBayes import NaiveBayes

try:
    registry = MLRegistry()
    
    nb = NaiveBayes()
    # add to ML registry
    registry.add_algorithm(endpoint_name = "Gunosy_classifier",
                           algorithm_object = nb,
                           algorithm_name = "Naive Bayes",
                           algorithm_status = "production",
                           algorithm_version = "1.0",
                           owner = "Tung Dang",
                           algorithm_description = "Navie Bayes with NLP to classify news article",
                           algorithm_code = inspect.getsource(NaiveBayes))
except Exception as e:
    print("Exception while loading the algorithm to the registry", str(e))

