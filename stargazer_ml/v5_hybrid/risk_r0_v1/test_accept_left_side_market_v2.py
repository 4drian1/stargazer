import unittest
import pandas as pd
from accept_left_side_market_v2 import normalize_text


class TextSchemaTests(unittest.TestCase):
    def test_string_and_object_readback_are_equal_after_explicit_normalization(self):
        a=pd.DataFrame(dict(symbol=['X'],stage=['POSTCLOSE'],signal_status=['CANDIDATE'],value=[1.]))
        b=a.copy()
        for name in ['symbol','stage','signal_status']:b[name]=b[name].astype('string')
        pd.testing.assert_frame_equal(normalize_text(a),normalize_text(b))

    def test_different_values_remain_detectable(self):
        a=pd.DataFrame(dict(symbol=['X'],stage=['POSTCLOSE'],signal_status=['CANDIDATE'],value=[1.]))
        for column,value in [('symbol','Y'),('value',2.)]:
            b=a.copy();b.loc[0,column]=value
            with self.assertRaises(AssertionError):
                pd.testing.assert_frame_equal(normalize_text(a),normalize_text(b))

    def test_empty_schema_matches_nonempty_prefix(self):
        a=pd.DataFrame(dict(symbol=['X'],stage=['POSTCLOSE'],signal_status=['CANDIDATE'],value=[1.]))
        b=pd.DataFrame(dict(symbol=pd.Series(dtype=object),stage=pd.Series(dtype=object),
                            signal_status=pd.Series(dtype=object),value=pd.Series(dtype=float)))
        pd.testing.assert_frame_equal(normalize_text(a.iloc[:0]),normalize_text(b))


if __name__=='__main__':unittest.main()
