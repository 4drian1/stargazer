"""Pure synthetic tests; no SDK imports, queries, or production writes."""
import unittest
from inventory_ths_desktop_local_v1 import bridge, parse_name_table


class LocalInventoryTests(unittest.TestCase):
    def test_gbk_versions_and_names(self):
        names, versions, rejected = parse_name_table('[name_48_48]\nConfigVer=20261016_x\n885921=储能\n886015=创新药'.encode('gb18030'))
        self.assertEqual(names, {'储能': ['885921'], '创新药': ['886015']})
        self.assertEqual(versions, ['20261016_x'])
        self.assertEqual(rejected, [])

    def test_duplicate_identical_row_not_ambiguity(self):
        names, _, _ = parse_name_table('885921=储能\n885921=储能'.encode('gb18030'))
        self.assertEqual(names['储能'], ['885921'])

    def test_duplicate_name_different_codes_stays_ambiguous(self):
        names, _, _ = parse_name_table('885921=储能\n886015=储能'.encode('gb18030'))
        row = bridge([dict(code='306380.TI', name='储能')], names, {'885921'})[0]
        self.assertEqual(row['bridge_status'], 'AMBIGUOUS_NAME')
        self.assertEqual(len(row['desktop_code_candidates']), 2)
        self.assertFalse(row['training_admitted'])

    def test_no_substring_or_alias_matching(self):
        names, _, _ = parse_name_table('885921=储能设备|储能'.encode('gb18030'))
        row = bridge([dict(code='306380.TI', name='储能')], names, {'885921'})[0]
        self.assertEqual(row['bridge_status'], 'UNMAPPED')

    def test_exact_name_and_file_not_identity_admission(self):
        row = bridge([dict(code='306380.TI', name='储能')], {'储能': ['885921']}, {'885921'})[0]
        self.assertEqual(row['cached_day_candidates'], ['885921'])
        self.assertFalse(row['identity_verified'])
        self.assertFalse(row['historical_membership_verified'])

    def test_malformed_lines_are_explicit(self):
        names, _, rejected = parse_name_table(b'bad line\n123=wrong\n885921=')
        self.assertEqual(names, {})
        self.assertEqual(rejected, [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
