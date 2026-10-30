"""Current manual feed: verify recomputation, no old six-ticket fallback."""
import copy,json,tempfile,unittest
from pathlib import Path
from ura.manual_formations import evaluate_race
from ura_haruq_independent.web.live_predict import SYSTEM_ID
from ura_haruq_independent.web.publish import verify_document,publish_batch

class ManualWebTests(unittest.TestCase):
    def document(self):
        ranked=[{'bike':b,'score':.8-i*.1} for i,b in enumerate(range(1,8))]
        src={'race_id':'202610305501','grade':'F2','race_class':'A1/A2','ranked':ranked,
            'confidence':.1,'model_sha256':'independent-sha','model_role':'INDEPENDENT_CHRONOLOGICAL_2026',
            'line_formation':[[1,2,3],[4,5],[6,7]]}
        return {'system':SYSTEM_ID,'race_id':src['race_id'],'date':'2026-10-30',
            'original_flagship_data_used':False,'model_sha256':'independent-sha','ranked':ranked,
            'manual_input':src,'ura_haru_manual_display':evaluate_race(src)}
    def test_missing_independent_probability_shows_no_bet_not_v1(self):
        row=self.document();self.assertEqual(verify_document(row),row['race_id'])
        self.assertEqual(row['ura_haru_manual_display']['tickets'],[])
        self.assertEqual(row['ura_haru_manual_display']['reason'],'MISSING_COMPATIBLE_INDEPENDENT_PROBABILITIES')
    def test_mutated_ticket_rejected(self):
        row=self.document();row['ura_haru_manual_display']['tickets']=['1-2-3']
        with self.assertRaises(ValueError):verify_document(row)
    def test_rank_mismatch_rejected(self):
        row=self.document();row['manual_input']=copy.deepcopy(row['manual_input'])
        row['manual_input']['ranked'][0]['score']=100
        with self.assertRaises(ValueError):verify_document(row)
    def test_only_ura_feed_migrates_from_legacy(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';target=root/'docs/ura-haru/results'
            source.mkdir();target.mkdir(parents=True)
            (target/'index.json').write_text(json.dumps({'system':'URA_HARU_V1_20261024','items':[{'race_id':'old'}]}))
            row=self.document();(source/(row['race_id']+'.json')).write_text(json.dumps(row))
            manifest=publish_batch(source,target,'123')
            self.assertEqual(len(manifest['items']),1);self.assertEqual(manifest['items'][0]['points'],0)
            self.assertFalse((root/'docs/haruq').exists())

class LiveRoutingTests(unittest.TestCase):
    def test_independent_live_run_routes_to_manual_without_legacy_tickets(self):
        from unittest.mock import patch,Mock
        from datetime import datetime,timedelta
        import pandas as pd
        from ura_haruq_independent.web import live_predict as live
        day=(datetime.now(live.ZONE)+timedelta(days=1)).date()
        rid=day.strftime('%Y%m%d')+'5501'
        data={'grade':'F2','race_name':'A級一般','start_time':'23:30',
            'entries':[{'rider_id':str(b),'bike_number':b,'class':'A1','rider_name':str(b)} for b in range(1,8)],
            'line_formation':[[{'bike_number':b} for b in line] for line in [[1,2,3],[4,5],[6,7]]]}
        scraper=Mock();scraper.scrape_race_entry.return_value=data
        builder=Mock();builder.feature_names=['input_feature']
        builder.build_race_features.return_value=pd.DataFrame({'rider_id':[str(b) for b in range(1,8)],'input_feature':[0]*7})
        model=Mock();model.predict.return_value=pd.Series([.8,.7,.6,.5,.4,.3,.2])
        with patch.object(live,'persist_entries',return_value={1:'A',2:'A',3:'A',4:'B',5:'B',6:'C',7:'C'}):
            out=live.one_race(rid,scraper,model,builder,'independent-test-sha',{'last_date':day.isoformat()})
        self.assertNotIn('ura_haru_v1_display',out)
        self.assertEqual(out['ura_haru_manual_display']['reason'],'MISSING_COMPATIBLE_INDEPENDENT_PROBABILITIES')
        self.assertEqual(out['ura_haru_manual_display']['tickets'],[])
        self.assertEqual(out['line_pattern'],'3-2-2')
        self.assertEqual(verify_document(out),rid)
