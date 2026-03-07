import unittest
import json
from app import app

class FlaskTestCase(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()
        # self.app.testing = True

    def test_home(self):
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)

    def test_prediction(self):
        response = self.app.post(
            "/predict",
            data=json.dumps({
                "build_time": 16,
                "failure_rate": 0.5,
                "commit_freq": 3,
                "deploy_delay": 6
            }),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("prediction", data)

if __name__ == "__main__":
    unittest.main()
