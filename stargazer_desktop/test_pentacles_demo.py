"""Zero-network demo API tests; do not import server or start old monitor threads."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flask import Flask, jsonify, request
import pentacles_demo as demo


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.file = Path(self.temp.name) / 'snapshot.json'
        self.mock = patch.object(demo, 'snapshot_path', return_value=self.file)
        self.mock.start()
        demo._running, demo._error = False, None

    def tearDown(self):
        self.mock.stop()
        self.temp.cleanup()

    def test_empty_explicit_not_live(self):
        self.assertIsNone(demo.state()['snapshot'])
        self.assertFalse(demo.state()['live_ready'])

    def test_toggle_isolated_and_bool_only(self):
        self.assertTrue(demo.set_enabled(True)['enabled'])
        self.assertFalse(demo.set_enabled(False)['enabled'])
        for value in ('true', 1, None):
            with self.assertRaises(ValueError):
                demo.set_enabled(value)

    def test_corrupt_snapshot_is_visible(self):
        self.file.write_text('{broken', encoding='utf-8')
        self.assertTrue(demo.state()['error'])
        self.assertIsNone(demo.state()['snapshot'])

    def test_stale_schema_and_size(self):
        self.file.write_text(json.dumps({'schema': 'PAGE_OF_PENTACLES_DEMO_V1',
                                        'data_date': 20250101, 'rows': []}), encoding='utf-8')
        self.assertTrue(demo.state()['stale'])
        self.file.write_text('{}', encoding='utf-8')
        self.assertTrue(demo.state()['error'])
        self.file.write_bytes(b' ' * (demo.MAX_BYTES + 1))
        self.assertIn('1MiB', demo.state()['error'])

    def test_missing_environment_keeps_snapshot(self):
        self.file.write_text('preserve', encoding='utf-8')
        with self.assertRaises(ValueError):
            demo.run({'pentacles_demo': {'project_root': self.temp.name}})
        self.assertEqual(self.file.read_text(), 'preserve')
        self.assertFalse(demo._running)

    def test_actual_api_functions_without_server_side_effects(self):
        tree = ast.parse(Path(__file__).with_name('server.py').read_text(encoding='utf-8-sig'))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name.startswith('api_pentacles_')]
        app = Flask('pentacles-test')
        namespace = dict(app=app, jsonify=jsonify, request=request, pentacles_demo=demo, get_cfg=lambda: {})
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'server.py', 'exec'), namespace)
        client = app.test_client()
        self.assertEqual(client.get('/api/pentacles/state').status_code, 200)
        self.assertEqual(client.get('/api/pentacles/journal').status_code, 200)
        self.assertEqual(client.get('/api/pentacles/journal?day=../../config').status_code, 400)
        self.assertTrue(client.post('/api/pentacles/monitor', json={'enabled': True}).json['enabled'])
        for body in ([], {'enabled': 'true'}):
            self.assertEqual(client.post('/api/pentacles/monitor', json=body).status_code, 400)
        with patch.object(demo, 'run', side_effect=ValueError('environment missing')):
            result = client.post('/api/pentacles/run', json={})
            self.assertEqual(result.status_code, 400)
            self.assertIn('environment missing', result.json['error'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
