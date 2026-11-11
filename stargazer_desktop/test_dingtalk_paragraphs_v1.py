"""DingTalk markdown collapses single \\n in clients; send_wechat must upgrade
paragraph breaks so pushed reports keep their line structure."""
import json
import types
import unittest
from unittest import mock

import notifier


class DingtalkParagraphTest(unittest.TestCase):
    def _capture(self, text):
        captured = []

        def receive(req, **kwargs):
            captured.append(json.loads(req.data))
            return types.SimpleNamespace(read=lambda: b'{"errcode":0,"errmsg":"ok"}')

        with mock.patch.object(notifier.urllib.request, 'urlopen', side_effect=receive):
            ok, _ = notifier.send_wechat(
                {'push': {'provider': 'dingtalk', 'dingtalk_webhook': 'https://example.invalid/t'}},
                '标题', text)
        self.assertTrue(ok)
        return captured[0]['markdown']['text']

    def test_single_newlines_become_blank_lines(self):
        out = self._capture('第一行\n第二行\n第三行')
        self.assertEqual(out, '第一行\n\n第二行\n\n第三行')

    def test_existing_blank_lines_not_quadrupled(self):
        out = self._capture('甲\n\n乙\n丙')
        self.assertEqual(out, '甲\n\n乙\n\n丙')
        self.assertNotIn('\n\n\n', out)


if __name__ == '__main__':
    unittest.main()
