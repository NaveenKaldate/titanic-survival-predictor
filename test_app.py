"""
Unit & Integration Tests for Titanic Survival Prediction App
"""

import warnings
warnings.filterwarnings("ignore")
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

import unittest
from app import app, predict_passenger, load_model

class TitanicAppTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure model is loadable
        cls.model = load_model()
        cls.client = app.test_client()

    def test_model_loaded(self):
        self.assertIsNotNone(self.model)
        self.assertEqual(len(self.model.classes_), 2)

    def test_predict_rose_scenario(self):
        # 1st Class Female, 17yo, with family, high fare -> Should survive
        res = predict_passenger(pclass=1, sex='female', age=17, sibsp=1, parch=1, fare=150.0, embarked='C')
        self.assertTrue(res['survived'])
        self.assertEqual(res['verdict'], 'SURVIVED')
        self.assertGreater(res['survival_probability'], 70.0)

    def test_predict_jack_scenario(self):
        # 3rd Class Male, 20yo, alone, low fare -> Should not survive
        res = predict_passenger(pclass=3, sex='male', age=20, sibsp=0, parch=0, fare=7.8, embarked='S')
        self.assertFalse(res['survived'])
        self.assertEqual(res['verdict'], 'DID NOT SURVIVE')
        self.assertGreater(res['mortality_probability'], 70.0)

    def test_index_get_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Titanic Survival Predictor", response.data)

    def test_index_post_form_route(self):
        payload = {
            'pclass': '1',
            'sex': 'female',
            'age': '22',
            'sibsp': '0',
            'parch': '0',
            'fare': '100.0',
            'embarked': 'C'
        }
        response = self.client.post('/', data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"SURVIVED", response.data)

    def test_api_predict_json_route(self):
        payload = {
            'pclass': 3,
            'sex': 'male',
            'age': 30,
            'sibsp': 0,
            'parch': 0,
            'fare': 8.05,
            'embarked': 'S'
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data['status'], 'success')
        self.assertIn('survived', json_data['data'])
        self.assertIn('survival_probability', json_data['data'])
        self.assertIn('mortality_probability', json_data['data'])
        self.assertAlmostEqual(
            json_data['data']['survival_probability'] + json_data['data']['mortality_probability'],
            100.0,
            places=1
        )

    def test_api_info_route(self):
        response = self.client.get('/api/info')
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data['status'], 'online')
        self.assertEqual(json_data['model_type'], 'RandomForestClassifier')

if __name__ == '__main__':
    unittest.main()
