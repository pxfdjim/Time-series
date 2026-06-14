if [[ "${MINDTS_GPU_CLI_ARGS+x}" != "x" ]]; then
  MINDTS_GPU_CLI_ARGS="--gpus 0 1"
fi

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-name-list "EWJ.csv" --model-name "MindTS.MindTS" --model-hyper-params '{"batch_size": 16, "d_ff": 512, "d_model": 256, "e_layers": 1, "horizon": 0, "norm": true, "num_epochs": 5, "seq_len": 48, "patch_size": 6, "stride": 6, "mask_ratio": 0.3, "r":0.5, "enc_in_time": 1, "parallel_strategy": null, "stl_period": 5, "dataset_description": "EWJ is a financial stock series. Numerical data come from Yahoo Finance; related financial news comes from NASDAQ, Bloomberg, and other financial news websites."}' ${MINDTS_GPU_CLI_ARGS} --num-workers 1 --timeout 60000 --save-path "${MINDTS_SAVE_ROOT:-label}/EWJ" --text-name-list "EWJ_text.csv"
