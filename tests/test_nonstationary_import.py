"""Regression tests for the nonstationary bandit lecture example."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "02._表格解法方法" / "04_2.5_跟踪非平稳问题.py"


class ImportSafetyTests(unittest.TestCase):
    def test_import_does_not_run_experiment_or_write_files(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("os.getcwd", return_value=directory):
                spec = importlib.util.spec_from_file_location("nonstationary_bandit_example", SOURCE)
                module = importlib.util.module_from_spec(spec)
                with patch("pandas.DataFrame.to_csv", side_effect=AssertionError("import wrote CSV")):
                    spec.loader.exec_module(module)
                self.assertTrue(hasattr(module, "NonstationaryBandit"))
                bandit = module.NonstationaryBandit(num_arms=2, alpha=0.5, reward_variation=0.0)
                bandit.true_values[:] = 0
                bandit.update_estimates(0, 2.0)
                self.assertAlmostEqual(bandit.estimated_values[0], 1.0)
                self.assertEqual(bandit.action_counts[0], 1)

    def test_zero_steps_returns_no_samples(self):
        spec = importlib.util.spec_from_file_location("nonstationary_bandit_zero", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.NonstationaryBandit(num_arms=2).run(0), [])


if __name__ == "__main__":
    unittest.main()
