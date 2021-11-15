from unittest import TestCase

import numpy as np

from encoders.encoder import BytesLabelEncoder, StringLabelEncoder


class EncoderTest(TestCase):
    def test_bytes(self):
        stuff = [b"asd", b"qwe", b"zxc"] * 2

        le = BytesLabelEncoder()
        le.partial_fit(stuff)
        le.partial_fit(stuff)
        self.assertEqual(le.classes, {b"asd": 0, b"qwe": 1, b"zxc": 2})
        le.transform(stuff) == np.array([0, 1, 2, 0, 1, 2])

    def test_bytes_typeerror(self):

        with self.assertRaises(TypeError):
            le = BytesLabelEncoder()
            le.partial_fit(["asd"])

    def test_str(self):
        stuff = ["asü", "😀", "zxä"] * 2

        le = StringLabelEncoder()
        le.partial_fit(stuff)
        le.partial_fit(stuff)
        self.assertEqual(le.classes, {"asü": 0, "😀": 1, "zxä": 2})
        le.transform(stuff) == np.array([0, 1, 2, 0, 1, 2])

    def test_str_typeerror(self):

        with self.assertRaises(TypeError):
            le = StringLabelEncoder()
            le.partial_fit([b"asd"])


if __name__ == "__main__":
    import unittest

    unittest.main()
