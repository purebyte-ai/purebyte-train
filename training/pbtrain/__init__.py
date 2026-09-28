"""pbtrain: the training stack of PureByte's AI Specialists.

A specialist is built from ONE task.yaml by the factory (`pbtrain.factory`), which runs the stages
data -> baselines -> finetune -> eval -> calibrate -> export -> verify -> report -> release, each one refusing to run
unless every earlier one ran, on the same task, in order (the done.json chain). On the CPU the model (`pbtrain.model`)
and the trainer (`pbtrain.train`) are deterministic and locked by golden fingerprints
(training/tests/test_golden_training.py, compared on the platform that recorded them); the exporter writes the engine's
format (spec/FORMAT.md). Start with docs/training.md.
"""
__version__ = "0.1.0"
