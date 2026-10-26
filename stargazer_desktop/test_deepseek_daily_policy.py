"""Daily caps removed by user; retain the original ledger and request safeguards."""
from datetime import datetime
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import deepseek_api


class DailyPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.redirect = patch('paths.APP_DIR', str(self.root))
        self.redirect.start()
        self.db = self.root/'data/assistant/deepseek_usage.sqlite'
        self.db.parent.mkdir(parents=True)
        self.day = datetime.now().strftime('%Y%m%d')
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute('CREATE TABLE usage(day TEXT PRIMARY KEY,calls INTEGER,tokens INTEGER)')
            conn.execute('INSERT INTO usage VALUES(?,?,?)', (self.day, 80, 163316))
            conn.execute('INSERT INTO usage VALUES(?,?,?)', ('20000101', 2, 500))

    def tearDown(self):
        self.redirect.stop()
        self.temp.cleanup()

    def invoke(self, error=None, messages=None):
        response = SimpleNamespace(usage=SimpleNamespace(total_tokens=30,prompt_tokens=20,completion_tokens=10),
            choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='本地测试'))])
        with patch.object(deepseek_api,'get_key',return_value='sk-test-only-not-a-real-key'), patch('openai.OpenAI') as sdk:
            create = sdk.return_value.chat.completions.create
            create.return_value = response
            if error: create.side_effect = error
            try:
                return deepseek_api.complete({},messages or [{'role':'user','content':'继续对话'}])
            finally:
                self.assertLessEqual(create.call_count,1)
                if create.call_count: self.assertEqual(sdk.call_args.kwargs['max_retries'],0)

    def usage(self):
        with closing(sqlite3.connect(self.db)) as conn:
            return conn.execute('SELECT calls,tokens FROM usage WHERE day=?',(self.day,)).fetchone()

    def test_existing_over_budget_day_still_calls_and_accumulates(self):
        self.invoke()
        self.assertEqual(self.usage(),(81,163346))
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute('SELECT calls,tokens FROM usage WHERE day=?',('20000101',)).fetchone(),(2,500))

    def test_service_failure_counted_once_without_retry(self):
        with self.assertRaisesRegex(ValueError,'请求失败'):
            self.invoke(OSError('private service error'))
        self.assertEqual(self.usage()[0],81)
        self.assertGreater(self.usage()[1],163316)

    def test_single_request_input_bound_retained(self):
        with self.assertRaisesRegex(ValueError,'单次输入'):
            self.invoke(messages=[{'role':'user','content':'字'*60001}])
        self.assertEqual(self.usage(),(80,163316))


if __name__=='__main__': unittest.main()
