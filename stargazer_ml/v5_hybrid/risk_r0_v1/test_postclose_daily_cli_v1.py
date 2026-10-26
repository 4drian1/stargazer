import unittest
from postclose_daily_cli_v1 import require_completed_day


class ClockTests(unittest.TestCase):
    def test_completed_day_allowed(self):
        require_completed_day(20261016,'2026-10-18T09:00:00+08:00')
        require_completed_day(20261019,'2026-10-19T15:00:00+08:00')

    def test_future_day_and_intraday_refused(self):
        for now in ['2026-10-18T09:00:00+08:00','2026-10-19T14:59:59+08:00']:
            with self.assertRaisesRegex(ValueError,'NO_FETCH'):require_completed_day(20261019,now)

    def test_wrong_or_naive_timezone_refused(self):
        for now in ['2026-10-19T16:00:00','2026-10-19T16:00:00+00:00']:
            with self.assertRaisesRegex(ValueError,'EXCHANGE_CLOCK'):require_completed_day(20261019,now)


if __name__=='__main__':unittest.main()
