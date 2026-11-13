import ast
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import sector_rule_scan as scan
import assistant_context
import chat_archive
import conversation_intent as intent
import plan_board_monitor


class SectorRulesTests(unittest.TestCase):
    def setUp(self):
        scan.CACHE.clear();scan.RUNNING=False
        self.tmp=tempfile.TemporaryDirectory()
        self.base=Path(self.tmp.name)
        self.app=patch('paths.APP_DIR',str(self.base));self.app.start()

    def tearDown(self):
        self.app.stop();self.tmp.cleanup()

    def test_ambiguous_names_and_case_insensitive_exact_code(self):
        import pandas as pd
        frame=pd.DataFrame([dict(name=n,code=c,source='ths',type='concept',symbols=['002780']) for n,c in
            [('PCB概念','PCB.TI'),('存储芯片','A.TI'),('数据存储','B.TI')]])
        row,err=scan.resolve_board(frame,'pcb');self.assertIsNone(err);self.assertEqual(row['name'],'PCB概念')
        row,err=scan.resolve_board(frame,'PCB.TI');self.assertEqual(row['code'],'PCB.TI')
        row,err=scan.resolve_board(frame,'存储');self.assertIsNone(row);self.assertEqual(err['status'],'AMBIGUOUS_BOARD')
        self.assertEqual(len(err['choices']),2)
        self.assertEqual(scan.resolve_board(frame,'不存在')[1]['status'],'BOARD_NOT_FOUND')

    def test_completed_dates_never_use_partial_current_daily(self):
        project=HERE.parents[1]
        self.assertEqual(scan.source_context(project,'2026-10-27T10:30:00+08:00')['P'],20261026)
        context=scan.source_context(project,'2026-10-27T16:00:00+08:00')
        self.assertEqual((context['P'],context['E']),(20261027,20261028))
        with self.assertRaisesRegex(ValueError,'TIMEZONE'):scan.source_context(project,'2026-10-27T10:30:00')

    def test_formal_first_ties_and_shortfall_do_not_fill_with_manual(self):
        formal=[dict(code='002780',selection_score=.01),dict(code='000573',selection_score=.01)]
        manual=[dict(code='600362',selection_score=.99)]
        self.assertEqual([r['code'] for r in scan.ranked_candidates(formal,manual,5)],['000573','002780'])
        result=scan.ranked_candidates([],manual,5)
        self.assertEqual(len(result),1);self.assertEqual(result[0]['rank'],1)
        self.assertNotIn('rank',manual[0])

    def test_actual_frozen_manual_features_and_hard_rejections(self):
        from test_manual_display_v1 import ManualDisplayTests
        import tarot_live_v2 as live
        fixture=ManualDisplayTests();fixture.setUp();g=fixture.near_miss()
        model=Mock();model.predict.return_value=[.123]
        row,error=scan.evaluate_stock(g,live,fixture.core,model,'test-sha')
        self.assertIsNone(error);self.assertFalse(row['formally_admitted'])
        self.assertIn('FIRST_LEG_VOLUME_NOT_DECREASING',row['failed_gates'])
        self.assertEqual(list(model.predict.call_args.args[0]),fixture.core.FEATURES)
        missing=g.copy();missing.loc[len(missing)-1,'amount']=float('nan');model.reset_mock()
        self.assertEqual(scan.evaluate_stock(missing,live,fixture.core,model,'test-sha')[1],'MISSING_FINAL_DAILY_FIELDS')
        model.predict.assert_not_called()
        for bad_kind in ('ST','INVALID_PRICE','TRIPLE_TOP'):
            bad=g.copy();model.reset_mock()
            if bad_kind=='ST':bad.loc[len(bad)-1,'is_st']=True
            if bad_kind=='INVALID_PRICE':bad.loc[len(bad)-1,'low']=-1
            with patch.object(fixture.core,'top_structure',return_value=[1,2,3]) if bad_kind=='TRIPLE_TOP' else patch.dict({},{}):
                rejected,reason=scan.evaluate_stock(bad,live,fixture.core,model,'test-sha')
            self.assertIsNone(rejected);model.predict.assert_not_called()

    def test_cache_shared_by_chat_and_monitor_without_repeat_worker(self):
        project=HERE.parents[1]
        context={'key':'source-version'}
        done=SimpleNamespace(returncode=0,stdout=json.dumps({'status':'READY','rows':[],'formal_count':0}))
        with patch.object(scan,'project_root',return_value=project),patch.object(scan,'source_context',return_value=context),patch.object(scan.subprocess,'run',return_value=done) as run:
            first=scan.scan_sector_rules('PCB');second=scan.scan_sector_rules('PCB')
            self.assertEqual(first,second);self.assertEqual(run.call_count,1)
            self.assertEqual(scan.monitor_state(['PCB'],datetime.now(scan.ZONE))[0]['status'],'READY')
            self.assertEqual(run.call_count,1)
            context['key']='changed-version';scan.scan_sector_rules('PCB');self.assertEqual(run.call_count,2)

    def test_monitor_does_not_wait_for_expensive_worker_lock(self):
        context={'key':'pending'}
        scan.SCAN_LOCK.acquire()
        try:
            with patch.object(scan,'source_context',return_value=context),patch.object(scan,'project_root',return_value=HERE.parents[1]),patch.object(scan.threading,'Thread') as thread:
                self.assertEqual(scan.monitor_state(['PCB'],datetime.now(scan.ZONE))[0]['status'],'LOADING')
                self.assertTrue(thread.return_value.start.called)
        finally:scan.SCAN_LOCK.release();scan.RUNNING=False

    def test_quote_monitor_and_rules_are_separate_and_no_plan_writes(self):
        now=datetime(2026,10,28,10,0,tzinfo=scan.ZONE)
        plan={'monitor_date':'20261028','boards':[{'name':'PCB'}]}
        with patch.object(plan_board_monitor.trade_plan_v1,'load_plan',return_value=plan),patch.object(plan_board_monitor,'resolve',return_value=[{'name':'PCB','bk':'BK1234'}]),patch.object(plan_board_monitor,'RUNNING',True),patch.object(scan,'monitor_state',return_value=[{'status':'READY','rows':[]}]) as monitor:
            result=plan_board_monitor.state(now)
        self.assertEqual(result['status'],'OBSERVING');self.assertIn('rule_scans',result)
        monitor.assert_called_once_with(['PCB'],now)

    def test_long_user_input_passes_local_api_gate(self):
        import deepseek_api
        messages=assistant_context.bounded_messages([], [{'role':'user','content':'保留这些条件' * 16000}])
        self.assertGreater(len(json.dumps(messages,ensure_ascii=False)),60000)
        response=SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='收到'))],usage=SimpleNamespace(total_tokens=12))
        with patch.object(deepseek_api,'get_key',return_value='test-only'),patch('openai.OpenAI') as client:
            client.return_value.chat.completions.create.return_value=response
            self.assertIs(deepseek_api.complete({},messages),response)
            self.assertEqual(client.return_value.chat.completions.create.call_count,1)

    def test_large_scan_tool_keeps_all_ranked_bands_in_context(self):
        import assistant_workflow
        result={'scope':'COMPLETED_DAILY_RULE_SCAN_NOT_INTRADAY_TRIGGER','status':'READY',
            'rows':[{'code':'002780','observation_bands':[{'lower':1.,'zone_upper':1.1}]}],
            'excluded':[{'code':str(i),'reason':'DAILY_SOURCE_MISSING'} for i in range(500)]}
        file=assistant_workflow.root()/'chat_session_test.jsonl'
        summary=json.loads(chat_archive.tool_content(result,file))
        self.assertEqual(summary['rows'],result['rows']);self.assertIn('evidence_id',summary)
        self.assertEqual(summary['excluded_counts']['DAILY_SOURCE_MISSING'],500)

    def test_scan_request_never_edits_draft_or_enables_legacy_funnel(self):
        for message in ('帮我扫描PCB板块，有没有符合我规则的股票，买入节点在哪',
                        '筛选煤炭板块股票，没有符合的给我5只','扫描创新药，只要五只'):
            self.assertTrue(intent.asks_for_sector_scan(message));self.assertFalse(intent.plan_work(message,drafting=True))
        self.assertTrue(intent.plan_work('监控板块加上煤炭',drafting=True))
        self.assertTrue(intent.plan_work('5只',drafting=True))

    def test_real_incorrect_counts_fall_back_to_computed_scan_without_retry(self):
        import gzip
        captures=json.loads(gzip.decompress((HERE/'REAL_CHAT_CAPTURE.json.gz').read_bytes()))
        actual=captures[-1]
        result=json.loads(next(m['content'] for m in actual['messages'] if m.get('role')=='tool'))
        text=actual['response']['choices'][0]['message']['content']
        out=scan.checked_chat_reply(text,[result])
        self.assertNotIn('127只',out);self.assertIn('600362',out);self.assertIn('暂无V4观察区',out)
        self.assertIn('产出可用模型分数1只',out);self.assertIn('未自动重试',out)
        self.assertEqual(scan.checked_chat_reply('正式入围0只，人工复核1只',[result]),'正式入围0只，人工复核1只')

    def test_real_brain_exposes_new_tool_and_finishes_after_tool_budget(self):
        from agent import brain as module
        brain=module.Brain.__new__(module.Brain)
        brain.cfg={'agent':{}};brain.max_rounds=1
        brain.messages=[{'role':'system','content':'只读事实，不改计划'}]
        brain.archive=self.base/'session.jsonl'
        call=SimpleNamespace(id='scan-call',function=SimpleNamespace(name='scan_sector_rules',arguments='{"board_name":"PCB"}'))
        msg=SimpleNamespace(tool_calls=[call],content=None,model_dump=lambda **kw:{'role':'assistant','tool_calls':[{'id':'scan-call','type':'function','function':{'name':'scan_sector_rules','arguments':call.function.arguments}}]})
        final=SimpleNamespace(content='扫描完成，暂无正式入围，人工复核不足五只',tool_calls=None)
        with patch('deepseek_api.complete',side_effect=[SimpleNamespace(choices=[SimpleNamespace(message=msg)]),SimpleNamespace(choices=[SimpleNamespace(message=final)])]) as api,patch.object(module.services,'scan_sector_rules',return_value={'status':'READY','rows':[],'formal_count':0,'manual_review_count':0,'scored_count':0}) as tool:
            answer=brain.chat('帮我扫描PCB板块，买入节点在哪')
        tool.assert_called_once_with(board_name='PCB')
        names=[r['function']['name'] for r in api.call_args_list[0].kwargs['tools']]
        self.assertIn('scan_sector_rules',names);self.assertNotIn('scan_sector',names)
        self.assertNotIn('tools',api.call_args_list[-1].kwargs)
        self.assertNotIn('上限',answer);self.assertEqual(api.call_count,2)


if __name__=='__main__':unittest.main()
