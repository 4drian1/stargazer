from copy import deepcopy
import unittest
from unittest.mock import patch

import common_candidate_fixed_fit_v1 as fit


class CommonFitEntryTests(unittest.TestCase):
    def record(self):
        return dict(status='CHAT_APPROVED_ONE_FIXED_RESEARCH_FIT', scope=fit.SCOPE,
            config_sha256='config', entry_manifest_sha256='entry', sidecar_manifest_sha256='sidecar',
            user_quote='好的，请继续', approval_context='after real sidecar acceptance, one fixed pair',
            models=['ridge', 'lightgbm'], fits_per_model=1, http_authorized=False,
            deployment_authorized=False, parameter_search_authorized=False, calibration_authorized=False)

    def test_exact_authorization_accepted(self):
        fit.authorize(self.record(), 'config', 'entry', 'sidecar')

    def test_wrong_scope_hash_count_and_context_rejected(self):
        for key, value in [('scope', fit.SCOPE+'_EXTRA'), ('config_sha256', 'other'),
                           ('entry_manifest_sha256', 'other'), ('sidecar_manifest_sha256', 'other'),
                           ('fits_per_model', 2), ('models', ['lightgbm']), ('user_quote', ''),
                           ('approval_context', ''), ('http_authorized', True),
                           ('parameter_search_authorized', True), ('calibration_authorized', True),
                           ('deployment_authorized', True)]:
            record = deepcopy(self.record()); record[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                fit.authorize(record, 'config', 'entry', 'sidecar')

    def test_bare_flag_cannot_bypass_missing_authorization(self):
        with patch.object(fit, 'verify_manifest'), patch.object(fit, 'OUT') as out:
            out.exists.return_value = False
            with patch.object(fit, 'ENTRY') as entry:
                entry.__truediv__.return_value.read_bytes.return_value = b'{"status":"PASS_COMMON_CANDIDATE_ENTRY_NO_FIT"}'
                with patch.object(fit, 'AUTH') as auth, patch.object(fit.previous, 'fit_model') as model:
                    auth.read_bytes.side_effect = FileNotFoundError('authorization absent')
                    with self.assertRaises(FileNotFoundError):
                        fit.train(True)
                    model.assert_not_called()

    def test_missing_flag_and_existing_fit_stop_before_model(self):
        with patch.object(fit.previous, 'fit_model') as model:
            with self.assertRaisesRegex(ValueError, 'EXPLICIT_FIXED'):
                fit.train(False)
            with patch.object(fit, 'OUT') as out:
                out.exists.return_value = True
                with self.assertRaisesRegex(ValueError, 'OUTPUT_EXISTS'):
                    fit.train(True)
            model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
