import os
import unittest
from unittest.mock import patch
from src.agent.public_config import gateway_config

class PublicConfigTests(unittest.TestCase):
    def test_missing_config(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "ECG_BASE_URL"):
                gateway_config()
    def test_explicit_config(self):
        env = dict(ECG_API_KEY="dummy", ECG_BASE_URL="https://example.invalid/v1", ECG_MODEL="test-model")
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(gateway_config(), env)
    def test_unsafe_url_not_echoed(self):
        for url in ["ftp://example.invalid", "https://u:private@example.invalid/v1", "https://example.invalid?key=private", "https://example.invalid:bad"]:
            with patch.dict(os.environ, dict(ECG_API_KEY="dummy", ECG_BASE_URL=url, ECG_MODEL="test"), clear=True):
                with self.assertRaises(ValueError) as ctx: gateway_config()
                self.assertNotIn("private", str(ctx.exception))
    def test_gateway_fails_before_client(self):
        from src.agent.gateway import ToolGateway
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "ECG_MODEL"): ToolGateway()
