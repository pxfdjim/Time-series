if [[ "${MINDTS_GPU_CLI_ARGS+x}" != "x" ]]; then
  MINDTS_GPU_CLI_ARGS="--gpus 0 1"
fi

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-name-list "KR.csv" --model-name "MindTS.MindTS" --model-hyper-params '{"batch_size": 16, "d_ff": 16, "d_model": 256, "e_layers": 1, "horizon": 0, "norm": true, "num_epochs": 5, "seq_len": 24, "patch_size": 8, "stride": 8, "mask_ratio": 0.4, "enc_in_time": 1, "parallel_strategy": null, "stl_period": 5, "dataset_description": "KR is a financial stock series. Numerical data come from Yahoo Finance; related financial news comes from NASDAQ, Bloomberg, and other financial news websites."}' ${MINDTS_GPU_CLI_ARGS} --num-workers 1 --timeout 60000 --save-path "${MINDTS_SAVE_ROOT:-label}/KR"
