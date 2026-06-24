if [[ "${MINDTS_GPU_CLI_ARGS+x}" != "x" ]]; then
  MINDTS_GPU_CLI_ARGS="--gpus 0 1"
fi

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --data-name-list "Weather.csv" --model-name "MindTS.MindTS" --model-hyper-params '{"batch_size": 64, "d_ff": 8, "d_model": 64, "e_layers": 1, "horizon": 0, "norm": true, "num_epochs": 5, "seq_len": 24, "patch_size": 6, "stride": 6, "mask_ratio": 0.4, "enc_in_time": 4, "parallel_strategy": null, "stl_period": 3, "dataset_description": "Weather contains temperature and humidity statistics and reports collected from government websites."}' ${MINDTS_GPU_CLI_ARGS} --num-workers 1 --timeout 60000 --save-path "${MINDTS_SAVE_ROOT:-label}/Weather"
