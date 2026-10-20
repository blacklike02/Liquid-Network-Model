"""Keep language evaluation boundaries and vocabulary independent of test data."""

import unittest

import torch

from benchmark_memory_language import make_splits, prepare_text, train_batch


class LanguageBenchmarkTests(unittest.TestCase):
    def test_train_only_vocabulary_and_nonoverlapping_splits(self):
        text = 'abcdefgh' * 10 + 'V' * 10 + 'T' * 10
        vocabulary, splits = make_splits(text)
        self.assertNotIn('V', vocabulary)
        self.assertNotIn('T', vocabulary)
        self.assertEqual([len(splits[k]) for k in ('train','validation','test')], [80,10,10])
        self.assertTrue(torch.all(splits['validation'] == 0))
        self.assertTrue(torch.all(splits['test'] == 0))
        data = torch.arange(80)
        x, y = train_batch(data, torch.Generator().manual_seed(3), 64, 8)
        torch.testing.assert_close(y, x + 1)
        self.assertGreaterEqual(x.min(), 0)
        self.assertLess(y.max(), 80)

    def test_gutenberg_headers_and_footers_are_excluded(self):
        raw = (b'header\r\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\r\n'
               b'\r\nOnly the story.\r\n\r\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\r\nfooter')
        self.assertEqual(prepare_text(raw), 'Only the story.')
        self.assertEqual(prepare_text(b'plain\r\ntext'), 'plain\ntext')


if __name__ == '__main__':
    unittest.main()
