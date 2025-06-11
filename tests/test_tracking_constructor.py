import unittest
from src.tracking import Tracking

class TestTrackingConstructor(unittest.TestCase):
    def test_accepts_integer(self):
        t = Tracking(source=0)
        self.assertEqual(t.source, 0)

    def test_accepts_string_digit(self):
        t = Tracking(source='1')
        self.assertEqual(t.source, 1)

    def test_accepts_string_url(self):
        url = 'http://example.com/video'
        t = Tracking(source=url)
        self.assertEqual(t.source, url)

if __name__ == '__main__':
    unittest.main()
