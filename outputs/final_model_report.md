# Final Model Report (Phase 9)

* generated: 2026-09-24 21:22
* ensemble: [('lgbm', 'log', 1.0, {'n_estimators': 1500, 'learning_rate': 0.05562808369457181, 'num_leaves': 164, 'max_depth': 9, 'min_child_samples': 26, 'subsample': 0.8325763287231696, 'colsample_bytree': 0.6044790682863366, 'reg_lambda': 0.6356327963736789, 'reg_alpha': 0.27335643347367566})]
* blend space: pred
* training rows (supervised blocks): computed on full train up to 2015-06-19
* test rows: 46830
* validation RMSLE (holdout): 0.06248

## Submission checks
* rows_match_sample: PASS
* ids_match_sample_order: PASS
* no_missing: PASS
* no_negative: PASS
* columns_ok: PASS

## Prediction summary
count    46830.000000
mean      5925.942474
std       3599.827123
min          0.000000
25%       4105.220098
50%       5849.260141
75%       7876.250215
max      29347.819732

* zero predictions (closed hubs): 6548