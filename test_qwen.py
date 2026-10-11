"""GPU-backed smoke test for loading the Qwen model with Unsloth."""

import importlib.util
import unittest


def _gpu_unsloth_available():
    """Return whether this environment can run the Unsloth model test."""
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch
        # Probe the actual package import as well as its module spec.  Some
        # Unsloth installations are present but unusable on a CPU-only build.
        import unsloth  # noqa: F401
        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError, OSError):
        return False


@unittest.skipUnless(
    _gpu_unsloth_available(), "requires CUDA and an installed Unsloth package"
)
class QwenUnslothTests(unittest.TestCase):
    def test_qwen_model_loads_with_unsloth(self):
        """Load the 4-bit Qwen checkpoint and report basic GPU diagnostics."""
        import torch
        from unsloth import FastModel

        model_name = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
        print("开始加载Qwen3 1.7B...")

        # Keep the original model-loading smoke-test logic intact.
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            dtype=torch.float16,
            load_in_4bit=True,
        )

        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
        print("Qwen模型加载成功！")
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "显存占用:",
            round(torch.cuda.memory_allocated() / 1024**3, 2),
            "GB",
        )


if __name__ == "__main__":
    unittest.main()
