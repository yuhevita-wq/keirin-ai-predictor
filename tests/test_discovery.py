import unittest
from unittest.mock import patch
from data.discovery import race_ids_from_html, discover_day

class DiscoveryTests(unittest.TestCase):
    def test_link_queries_and_html_entities(self):
        html = '<a href="/db/result/?rf=calendar&amp;race_id=202309067301">1R</a><a href="/race/result/?race_id=202309067301&amp;rf=x">again</a><script>race_id=202309067302</script>'
        self.assertEqual(race_ids_from_html(html), ['202309067301'])

    def test_second_day_uses_opening_not_current_day(self):
        rows = {('20230905','73'):'小松島 初日', ('20230906','73'):'小松島 2日目'}
        with patch('data.discovery.calendar_rows', return_value=rows), patch('scripts.init_db.get_race_ids_from_program', return_value=['202309057301','202309067301','202309077301']) as get:
            ids, errors = discover_day(object(), '20230906', ['73'])
        self.assertEqual(ids, {'202309067301'})
        self.assertFalse(errors)
        self.assertEqual(get.call_args.args[1], '2023090573')
