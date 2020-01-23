from django.test import TestCase
from rest_framework.test import APIClient

# Create your tests here.

class EndpointTests(TestCase):
    
    def test_predict_view(self):
        client = APIClient()
        input_data = "https://gunosy.com/articles/a4Nlf"
        classifier_url = "/api/v1/Gunosy_classifier/prediction"
        response = client.post(classifier_url, input_data, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["Group: "], "スポーツ")
        self.assertTrue("request_id" in response.data)
        self.assertTrue("status: " in response.data)