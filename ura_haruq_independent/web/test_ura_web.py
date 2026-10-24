"""Independent UI/feed assertions. Never queries flagship, writes only temp dirs."""
from __future__ import annotations
import copy
import json
import tempfile
from pathlib import Path
import unittest

from ura_haruq_independent.web.live_predict import SYSTEM_ID
from ura_haruq_independent.web.publish import publish_batch, verify_document
from ura_haruq_independent.recommendation.low_quarter_advisory import evaluate_record

class UraWebTests(unittest.TestCase):
    def _document(self, gap=0.31, entries=7):
        bikes=[3,5,1,2,6,7,4]
        data={"score_gap":gap,"predicted_bikes_by_rank":bikes,
          "line_groups_by_bike":"3:A;5:A;1:B;2:B;6:C;7:C;4:D"}
        ev=evaluate_record(data)
        return {"system":SYSTEM_ID,"original_flagship_data_used":False,
          "race_id":"202610255501","date":"2026-10-25","velodrome":"佐世保",
          "race_number":1,"grade":"F2","start_time":"20:00","status":"発走前",
          "score_gap":gap,"line_pattern":"2-2-2-1",
          "rider_count":entries,"ranked":[{"rank":i+1,"bike":b,"name":"Rider "+str(i+1),"score":10-i}
              for i,b in enumerate(bikes)],
          "ura_haru_v1_display":ev}
    def test_v1_ok(self):
        row=self._document()
        self.assertEqual(verify_document(row), "202610255501")
        self.assertEqual(row["ura_haru_v1_display"]["points"],6)
    def test_non_eligible_has_no_recommendation(self):
        row=self._document(gap=1.25)
        self.assertEqual(row["ura_haru_v1_display"]["status"],"v1_not_applicable")
        self.assertEqual(verify_document(row),row["race_id"])
        self.assertIsNone(row["ura_haru_v1_display"]["recommendation"])
    def test_refuse_original_data(self):
        row=self._document()
        row["original_flagship_data_used"]=True
        with self.assertRaisesRegex(ValueError,"ORIGINAL_MODEL"):
            verify_document(row)
    def test_reject_mutated_tickets(self):
        row=self._document()
        row["ura_haru_v1_display"]["tickets"].append("1-2-3")
        with self.assertRaisesRegex(ValueError,"TICKET_COUNT"):
            verify_document(row)
    def test_publisher_makes_independent_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"source"
            dest=Path(tmp)/"docs/ura-haru/results"
            source.mkdir()
            row=self._document()
            (source/"202610255501.json").write_text(json.dumps(row,ensure_ascii=False))
            manifest=publish_batch(source,dest,"123")
            self.assertEqual(len(manifest["items"]),1)
            summary=manifest["items"][0]
            self.assertEqual(summary["recommendation"]["stars"],5)
            self.assertEqual(summary["points"],6)
            self.assertTrue((dest/"202610255501.json").is_file())
            self.assertFalse((Path(tmp)/"docs/haruq").exists())
    def test_frontend_does_not_reference_flagship_data(self):
        page=Path(__file__).resolve().parents[2]/"docs/ura-haru/index.html"
        s=page.read_text(encoding="utf-8")
        self.assertIn("裏ハルキュー",s)
        self.assertIn("URA_HARU_V1_20261024",s)
        self.assertNotIn("supabase",s.lower().replace("supabase／本番予測", ""))
        self.assertNotIn("prediction_detail",s)
        self.assertNotIn('src="https://cdn.jsdelivr.net/npm/@supabase/',s)

if __name__=="__main__":
    unittest.main()
