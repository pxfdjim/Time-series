# Metrics Summary: label_contrastive_withoutmseloss

Generated at: 2026-06-15 10:13:19

Selection rule: for each dataset, the latest `MindTS*.csv` and latest `test_report*.csv` in that dataset folder are used.

## Key Metrics

| dataset | affiliation_f | VUS_ROC | VUS_PR | f_score | precision | recall | auc_roc | auc_pr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KR | 0.755593 | 0.758649 | 0.401326 | 0.336968 | 0.288929 | 0.724387 | 0.850980 | 0.601765 |
| MDT | 0.659807 | 0.775658 | 0.489156 | 0.454197 | 0.523375 | 0.584699 | 0.863039 | 0.641366 |
| EWJ | 0.632481 | 0.594695 | 0.283696 | 0.322431 | 0.318779 | 0.522462 | 0.702170 | 0.417299 |
| Environment | 0.709948 | 0.834911 | 0.414461 | 0.435890 | 0.407870 | 0.666923 | 0.896825 | 0.563800 |
| Energy | 0.655838 | 0.624166 | 0.341967 | 0.365254 | 0.323643 | 0.642007 | 0.677576 | 0.306410 |
| Weather | 0.674906 | 0.772356 | 0.434184 | 0.402516 | 0.424174 | 0.524543 | 0.794508 | 0.423351 |

## Best Threshold By Aff-F

| dataset | threshold | Aff-F | VUS_ROC | VUS_PR | F1 | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KR | 23 | 0.812146 | 0.758649 | 0.401326 | 0.325000 | 0.204724 | 0.787879 |
| MDT | 24 | 0.765143 | 0.775658 | 0.489156 | 0.545455 | 0.451613 | 0.688525 |
| EWJ | 44 | 0.690339 | 0.594695 | 0.283696 | 0.241611 | 0.146939 | 0.679245 |
| Environment | 22 | 0.767631 | 0.834911 | 0.414461 | 0.512428 | 0.397626 | 0.720430 |
| Energy | 23 | 0.739552 | 0.624166 | 0.341967 | 0.414508 | 0.291971 | 0.714286 |
| Weather | 35 | 0.749949 | 0.772356 | 0.434184 | 0.465257 | 0.341463 | 0.729858 |

## All Mean Metrics Across Thresholds

| dataset | accuracy | f_score | precision | recall | adjust_accuracy | adjust_f_score | adjust_precision | adjust_recall | rrecall | rprecision | precision_at_k | rf | affiliation_f | affiliation_precision | affiliation_recall | auc_roc | auc_pr | R_AUC_ROC | R_AUC_PR | VUS_ROC | VUS_PR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KR | 0.772307 | 0.336968 | 0.288929 | 0.724387 | 0.773742 | 0.344897 | 0.293160 | 0.747475 | 0.729571 | 0.358322 | 0.059163 | 0.406978 | 0.755593 | 0.754426 | 0.807036 | 0.850980 | 0.601765 | 0.750548 | 0.402274 | 0.758649 | 0.401326 |
| MDT | 0.830848 | 0.454197 | 0.523375 | 0.584699 | 0.834511 | 0.483457 | 0.534239 | 0.617486 | 0.573260 | 0.540083 | 0.151054 | 0.467637 | 0.659807 | 0.795843 | 0.630897 | 0.863039 | 0.641366 | 0.751100 | 0.470724 | 0.775658 | 0.489156 |
| EWJ | 0.751343 | 0.322431 | 0.318779 | 0.522462 | 0.752730 | 0.331403 | 0.324837 | 0.536388 | 0.546861 | 0.353555 | 0.074573 | 0.364002 | 0.632481 | 0.666009 | 0.665348 | 0.702170 | 0.417299 | 0.557998 | 0.262175 | 0.594695 | 0.283696 |
| Environment | 0.881138 | 0.435890 | 0.407870 | 0.666923 | 0.884041 | 0.467724 | 0.426063 | 0.716846 | 0.658134 | 0.393719 | 0.091526 | 0.423468 | 0.709948 | 0.734713 | 0.742856 | 0.896825 | 0.563800 | 0.825424 | 0.410352 | 0.834911 | 0.414461 |
| Energy | 0.622418 | 0.365254 | 0.323643 | 0.642007 | 0.633333 | 0.406810 | 0.351812 | 0.705357 | 0.639036 | 0.238755 | 0.043793 | 0.260806 | 0.655838 | 0.639591 | 0.749915 | 0.677576 | 0.306410 | 0.635926 | 0.339004 | 0.624166 | 0.341967 |
| Weather | 0.758837 | 0.402516 | 0.424174 | 0.524543 | 0.804305 | 0.592435 | 0.544200 | 0.790454 | 0.551961 | 0.346541 | 0.096987 | 0.354193 | 0.674906 | 0.667991 | 0.744414 | 0.794508 | 0.423351 | 0.778302 | 0.432066 | 0.772356 | 0.434184 |

## Test Report Metrics

| dataset | affiliation_f | VUS_ROC | VUS_PR |
| --- | --- | --- | --- |
| KR | 0.755593 | 0.758649 | 0.401326 |
| MDT | 0.659807 | 0.775658 | 0.489156 |
| EWJ | 0.632481 | 0.594695 | 0.283696 |
| Environment | 0.709948 | 0.834911 | 0.414461 |
| Energy | 0.655838 | 0.624166 | 0.341967 |
| Weather | 0.674906 | 0.772356 | 0.434184 |

## Source Files

| dataset | metrics_csv | test_report_csv | align_loss_type | recon_loss_type |
| --- | --- | --- | --- | --- |
| KR | MindTS.20260615_001656.bs16_dm256_df16_ep5_seq24_patch8_stride8_mask0p4_r0p9_cin1.pc-b319-X11DPi-N-T.3744676.csv | test_report.20260615_001656.bs16_dm256_df16_ep5_seq24_patch8_stride8_mask0p4_r0p9_cin1.pc-b319-X11DPi-N-T.3744676.csv | contrastive | gaussian_nll |
| MDT | MindTS.20260615_001801.bs32_dm256_df32_ep5_seq24_patch6_stride6_mask0p4_r0p9_cin1.pc-b319-X11DPi-N-T.3744681.csv | test_report.20260615_001801.bs32_dm256_df32_ep5_seq24_patch6_stride6_mask0p4_r0p9_cin1.pc-b319-X11DPi-N-T.3744681.csv | contrastive | gaussian_nll |
| EWJ | MindTS.20260615_003016.bs16_dm256_df512_ep5_seq48_patch6_stride6_mask0p3_r0p5_cin1.pc-b319-X11DPi-N-T.3747745.csv | test_report.20260615_003016.bs16_dm256_df512_ep5_seq48_patch6_stride6_mask0p3_r0p5_cin1.pc-b319-X11DPi-N-T.3747745.csv | contrastive | gaussian_nll |
| Environment | MindTS.20260615_013254.bs64_dm64_df64_ep5_seq72_patch6_stride6_mask0p4_r0p9_cin1.pc-b319-X11DPi-N-T.3744679.csv | test_report.20260615_013254.bs64_dm64_df64_ep5_seq72_patch6_stride6_mask0p4_r0p9_cin1.pc-b319-X11DPi-N-T.3744679.csv | contrastive | gaussian_nll |
| Energy | MindTS.20260615_004739.bs32_dm256_df8_ep5_seq24_patch6_stride6_mask0p4_r0p2_cin9.pc-b319-X11DPi-N-T.3747133.csv | test_report.20260615_004739.bs32_dm256_df8_ep5_seq24_patch6_stride6_mask0p4_r0p2_cin9.pc-b319-X11DPi-N-T.3747133.csv | contrastive | gaussian_nll |
| Weather | MindTS.20260615_014211.bs64_dm64_df8_ep5_seq24_patch6_stride6_mask0p4_r0p8_cin4.pc-b319-X11DPi-N-T.3751902.csv | test_report.20260615_014211.bs64_dm64_df8_ep5_seq24_patch6_stride6_mask0p4_r0p8_cin4.pc-b319-X11DPi-N-T.3751902.csv | contrastive | gaussian_nll |
