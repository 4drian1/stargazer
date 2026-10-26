"""No paid requests or production data writes; all workflow evidence is isolated."""
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import assistant_workflow as workflow
import deepseek_api
import learning_review
import pentacles_review_data


def reply(content='{"commentary":"只观察，未知保留未知。"}', reason='stop', choices=True):
    choice = types.SimpleNamespace(finish_reason=reason, message=types.SimpleNamespace(content=content))
    return types.SimpleNamespace(choices=[choice] if choices else [],
        usage=types.SimpleNamespace(total_tokens=30, prompt_tokens=20, completion_tokens=10))


def facts():
    rows = [dict(code=f'{i:06}', name='样例', setup='A_SDP', status='DAILY_OBSERVATION',
                 daily_touch=(i % 2 == 0), close_change_pct=i/10-10, price_source='STOCKDB_HTTP')
            for i in range(200)]
    return dict(day=20261015, observe_date=20261015, data_date=20261014, candidate_count=200,
                snapshot_hash='f'*64, rows=rows, summary=dict(valid_daily_rows=200,
                daily_touch_count=100, touch_close_mean_pct=-.1), market=dict(status='UNKNOWN'),
                seven_day=[dict(code=r['code'], status='ENTRY_UNKNOWN') for r in rows],
                actual_trades=dict(status='NO_LEDGER', realized_gross_pnl=None, realized_net_pnl=None))


class ReviewFormatTests(unittest.TestCase):
    def test_full_counts_bounded_examples(self):
        s = learning_review.summary(facts())
        self.assertEqual((s['candidate_count'], s['touch_known'], s['touch_count']), (200, 200, 100))
        self.assertEqual(s['seven_day_status_counts'], {'ENTRY_UNKNOWN': 200})
        self.assertLessEqual(len(s['examples']), 6)
        self.assertLess(len(json.dumps(learning_review.prompt_context(facts()), ensure_ascii=False)), 5000)

    def test_local_render_null_and_zero_are_different(self):
        value = facts()
        value['rows'] = []
        value['actual_trades']['realized_gross_pnl'] = 0
        text = learning_review.render(value, '人工复核')
        self.assertIn('NULL（未知）', text)
        self.assertIn('0.00 元', text)
        self.assertIn('净盈亏 NULL', text)

    def test_confirmed_header_and_fact_table(self):
        preview = learning_review.render(facts(), '只观察')
        saved = learning_review.render(facts(), '只观察', confirmed=True)
        self.assertIn('待人工确认', preview)
        self.assertNotIn('待人工确认', saved)
        self.assertIn('有效日K 200/200', saved)
        self.assertIn('触区 100（50.0%）', saved)


class ReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.path_patch = patch('paths.APP_DIR', str(self.base))
        self.path_patch.start()
        self.snap = dict(schema='PAGE_OF_PENTACLES_DEMO_V1', data_date=20261014,
                         monitor_date=20261015, rows=[dict(code='000001')])
        p = self.base / 'data/pentacles_demo/snapshot.json'
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(self.snap), encoding='utf-8')
        self.learning = self.base / 'PENTACLES_LEARNING.md'
        self.learning.write_bytes('旧篇和旧数字不可改写\n'.encode())
        self.before = self.learning.read_bytes()
        workflow.JOB.update(running=False, draft=None, error=None)

    def tearDown(self):
        self.assertFalse(workflow.JOB['running'])
        self.path_patch.stop()
        self.tmp.cleanup()

    def run_review(self, api):
        with patch.object(pentacles_review_data, 'collect', return_value=facts()), patch.object(workflow, 'complete', api):
            workflow.review_start({}, 20261015)
            for _ in range(200):
                if not workflow.JOB['running']:
                    break
                time.sleep(.01)
            self.assertFalse(workflow.JOB['running'])

    def test_short_request_and_new_preview_append_only(self):
        from unittest.mock import Mock
        api = Mock(return_value=reply())
        self.run_review(api)
        self.assertIsNone(workflow.JOB['error'])
        messages = api.call_args.args[1]
        packed = json.dumps(messages, ensure_ascii=False)
        self.assertNotIn('旧篇和旧数字', packed)
        self.assertIn('最多600', packed)
        self.assertLess(len(packed), 6000)
        self.assertEqual(api.call_args.kwargs['purpose'], 'review')
        self.assertEqual(self.learning.read_bytes(), self.before)
        d = workflow.JOB['draft']
        self.assertEqual(d['report_version'], 2)
        self.assertIn('有效日K 200/200', d['report_markdown'])
        workflow.confirm_review(d['id'], True)
        after = self.learning.read_bytes()
        self.assertTrue(after.startswith(self.before))
        self.assertIn('触区 100（50.0%）', after.decode())
        self.assertNotIn('待人工确认', after.decode())
        self.assertTrue(workflow.confirm_review(d['id'], True)['duplicate'])
        self.assertEqual(self.learning.read_bytes(), after)

    def test_truncated_reply_no_draft_or_learning_write(self):
        from unittest.mock import Mock
        self.run_review(Mock(side_effect=ValueError('finish_reason=length')))
        self.assertIn('length', workflow.JOB['error'])
        self.assertIsNone(workflow.JOB['draft'])
        self.assertFalse(list(workflow.root().glob('review_*.json')))
        self.assertEqual(self.learning.read_bytes(), self.before)

    def test_json_invalid_not_saved(self):
        from unittest.mock import Mock
        self.run_review(Mock(return_value=reply('{"commentary":')))
        self.assertIn('JSON格式', workflow.JOB['error'])
        self.assertFalse(list(workflow.root().glob('review_*.json')))

    def test_overlong_commentary_not_silently_cut(self):
        from unittest.mock import Mock
        self.run_review(Mock(return_value=reply(json.dumps({'commentary': '长' * 1201}))))
        self.assertIn('短判读过长', workflow.JOB['error'])
        self.assertFalse(list(workflow.root().glob('review_*.json')))

    def test_legacy_cache_retained_but_not_reused(self):
        from unittest.mock import Mock
        old = workflow.store('review', dict(day=20261015, facts=facts(), facts_hash=workflow.digest(facts()), commentary='旧整篇'))
        file = workflow.root() / f"review_{old['id']}.json"
        original = file.read_bytes()
        api = Mock(return_value=reply())
        self.run_review(api)
        api.assert_called_once()
        self.assertEqual(file.read_bytes(), original)
        self.assertEqual(workflow.JOB['draft']['report_version'], 2)


class DeepSeekResultTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.p = patch('paths.APP_DIR', str(self.base))
        self.p.start()

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def complete(self, result=None, error=None):
        with patch.object(deepseek_api, 'get_key', return_value='sk-test-private-value'), patch('openai.OpenAI') as sdk:
            create = sdk.return_value.chat.completions.create
            create.return_value = result or reply()
            if error:
                create.side_effect = error
            try:
                return deepseek_api.complete({}, [{'role': 'user', 'content': 'PROMPT_PRIVATE_TEXT'}],
                                             structured=True, purpose='review')
            finally:
                self.assertEqual(create.call_count, 1)
                self.assertEqual(sdk.call_args.kwargs['max_retries'], 0)

    def metadata(self):
        with closing(sqlite3.connect(self.base/'data/assistant/deepseek_usage.sqlite')) as conn:
            result = conn.execute('SELECT purpose,finish_reason,outcome,max_tokens FROM results').fetchall()
            calls = conn.execute('SELECT SUM(calls) FROM usage').fetchone()[0]
            raw = json.dumps(conn.execute('SELECT * FROM results').fetchall())
        self.assertNotIn('sk-test-private-value', raw)
        self.assertNotIn('PROMPT_PRIVATE_TEXT', raw)
        return result, calls

    def test_length_specific_and_recorded_no_retry(self):
        with self.assertRaisesRegex(ValueError, 'length.*2000'):
            self.complete(reply(reason='length'))
        rows, calls = self.metadata()
        self.assertEqual(rows, [('review', 'length', 'TRUNCATED', 2000)])
        self.assertEqual(calls, 1)

    def test_empty_choices_separate(self):
        with self.assertRaisesRegex(ValueError, 'NO_CHOICES'):
            self.complete(reply(choices=False))
        self.assertEqual(self.metadata()[0][0][2], 'NO_CHOICES')

    def test_empty_body_separate(self):
        with self.assertRaisesRegex(ValueError, 'EMPTY_CONTENT'):
            self.complete(reply(content=''))
        self.assertEqual(self.metadata()[0][0][2], 'EMPTY_CONTENT')

    def test_content_filter_not_accepted(self):
        with self.assertRaisesRegex(ValueError, '服务端拦截'):
            self.complete(reply(reason='content_filter'))
        self.assertEqual(self.metadata()[0][0][2], 'INCOMPLETE')

    def test_transport_error_no_secret_and_counted(self):
        with self.assertRaises(ValueError) as error:
            self.complete(error=OSError('sk-test-private-value PROMPT_PRIVATE_TEXT'))
        self.assertNotIn('sk-test', str(error.exception))
        self.assertEqual(self.metadata()[0][0][2], 'REQUEST_ERROR')
        self.assertEqual(self.metadata()[1], 1)

    def test_success_no_full_output_in_log(self):
        self.complete(reply(content='PRIVATE_RESPONSE_TEXT'))
        self.assertEqual(self.metadata()[0][0][2], 'OK')
        db = (self.base/'data/assistant/deepseek_usage.sqlite').read_bytes()
        self.assertNotIn(b'PRIVATE_RESPONSE_TEXT', db)


class BrowserRegressionTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node required for isolated UI test')
    def test_review_preview_state_machine(self):
        script = Path(__file__).with_name('test_learning_review_ui.js')
        result = subprocess.run([shutil.which('node'), str(script)], capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
