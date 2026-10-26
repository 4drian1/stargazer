"""Live preparation overlay; retains the original checker's historical snapshot unchanged."""
import json
from check_historical_preparation_v1 import check
from market_index_source_v1 import bind_from_experiment_config
from research_budget_v2 import TOTAL_CAP,bytes_used


if __name__=='__main__':
    try:
        old=check()
        index=bind_from_experiment_config()
        report=dict(original_preparation_hashes=old['package_hashes'],
                    index_identity='SH000300/CSI300_PRICE_INDEX',market_index_source_bound=True,
                    index_dates=len(index),index_first=min(index),index_last=max(index),
                    registered_cap=TOTAL_CAP,registered_actual_bytes=bytes_used(),
                    training_ready=False,fit_authorized=False,
                    blockers=['stock_price_scale_effective_dates_and_corporate_action_admission',
                              'historical_security_coverage_scope',
                              'corrected_matrix_and_class_date_gates'])
        print(json.dumps(report,ensure_ascii=False))
        raise SystemExit(2)
    except Exception as exc:
        print(json.dumps(dict(training_ready=False,error=str(exc)),ensure_ascii=False))
        raise SystemExit(1)
