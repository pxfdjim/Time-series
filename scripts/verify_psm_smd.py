#!/usr/bin/env python3
"""Read-only check of PSM/SMD long-table conversion against original raw datasets."""
from pathlib import Path
import numpy as np
import pandas as pd

base = Path('dataset/anomaly_detect')
raw = base / 'raw'
out = base / 'data'
meta = pd.read_csv(base / 'DETECT_META.csv', dtype=str).fillna('')
expected_files = {'PSM.csv'} | {f'SMD_{p.stem}.csv' for p in (raw / 'SMD/train').glob('machine-*.txt')}
assert len(expected_files) == 29, f'Expected 28 SMD machines + PSM, got {len(expected_files)} files'


def check_one(filename, x_train, x_test, y_test, feature_count):
    train_len, test_len = len(x_train), len(x_test)
    total_len = train_len + test_len
    assert x_train.ndim == x_test.ndim == 2
    assert x_train.shape[1] == x_test.shape[1] == feature_count
    assert len(y_test) == test_len
    assert np.isfinite(x_train).all() and np.isfinite(x_test).all()
    assert np.isin(y_test, [0, 1]).all()

    matching = meta.loc[meta['file_name'] == filename]
    assert len(matching) == 1, f'{filename}: metadata entry count = {len(matching)}'
    m = matching.iloc[0]
    assert int(m['train_lens']) == train_len, f'{filename}: train_lens mismatch'
    assert int(m['total_len']) == total_len, f'{filename}: total_len mismatch'
    assert abs(float(m['train/total']) - train_len / total_len) < 1e-8

    data = np.concatenate([x_train, x_test], axis=0)
    labels = np.concatenate([np.zeros(train_len, dtype=np.int8), y_test.astype(np.int8)])
    observed = 0
    path = out / filename
    assert path.is_file(), f'Missing {path}'
    for chunk in pd.read_csv(path, chunksize=200_000):
        assert list(chunk.columns) == ['date', 'data', 'cols'], f'{filename}: header mismatch'
        vals = pd.to_numeric(chunk['data'], errors='raise').to_numpy(dtype=np.float64)
        dates = pd.to_numeric(chunk['date'], errors='raise').to_numpy(dtype=np.int64)
        channel_names = chunk['cols'].to_numpy()
        assert np.isfinite(vals).all(), f'{filename}: contains NaN/Inf'
        cursor = 0
        while cursor < len(chunk):
            global_i = observed + cursor
            group = global_i // total_len
            time_i = global_i % total_len
            n = min(len(chunk) - cursor, total_len - time_i)
            assert group < feature_count + 1, f'{filename}: too many channel groups'
            expected_name = f'channel{group + 1}' if group < feature_count else 'label'
            assert np.all(channel_names[cursor:cursor+n] == expected_name), (filename, expected_name, global_i)
            assert np.array_equal(dates[cursor:cursor+n], np.arange(time_i + 1, time_i + n + 1)), (filename, 'date index', global_i)
            want = data[time_i:time_i+n, group] if group < feature_count else labels[time_i:time_i+n]
            assert np.allclose(vals[cursor:cursor+n], want, rtol=1e-9, atol=1e-9), (filename, 'raw values / labels mismatch', expected_name, global_i)
            cursor += n
        observed += len(chunk)
    assert observed == total_len * (feature_count + 1), f'{filename}: row count {observed}'
    print(f'[PASS] {filename}: {feature_count} channels, train={train_len}, test={test_len}, rows={observed:,}', flush=True)


# PSM: official timestamp column is not a feature. Validate ZERO-imputation against raw.
a = pd.read_csv(raw / 'PSM/train.csv')
b = pd.read_csv(raw / 'PSM/test.csv')
ydf = pd.read_csv(raw / 'PSM/test_label.csv')
features = [f'feature_{i}' for i in range(25)]
assert list(a.columns[1:]) == list(b.columns[1:]) == features
assert len(ydf) == len(b)
if 'timestamp_(min)' in ydf:
    assert np.array_equal(ydf['timestamp_(min)'].to_numpy(), b['timestamp_(min)'].to_numpy())
a_values = a[features].to_numpy(dtype=np.float64)
b_values = b[features].to_numpy(dtype=np.float64)
assert not np.isinf(a_values).any() and not np.isinf(b_values).any()
print('[PSM] raw missing:', int(np.isnan(a_values).sum()), 'train cells;', int(np.isnan(b_values).sum()), 'test cells')
check_one('PSM.csv', np.nan_to_num(a_values, nan=0.0), np.nan_to_num(b_values, nan=0.0), ydf.iloc[:, -1].to_numpy(dtype=np.int8), 25)

# SMD: each machine is independently checked against its official raw files.
for src in sorted((raw / 'SMD/train').glob('machine-*.txt')):
    train = np.loadtxt(src, delimiter=',', ndmin=2)
    test = np.loadtxt(raw / 'SMD/test' / src.name, delimiter=',', ndmin=2)
    labels = np.loadtxt(raw / 'SMD/test_label' / src.name, delimiter=',').reshape(-1)
    check_one(f'SMD_{src.stem}.csv', train, test, labels, 38)

# Check no PSM/SMD metadata entries were accidentally added under other names.
new_rows = meta.loc[(meta['file_name'].str.startswith('SMD_machine-')) | (meta['file_name'] == 'PSM.csv')]
assert set(new_rows['file_name']) == expected_files, 'metadata PSM/SMD file set mismatch'
backup = base / 'DETECT_META.csv.backup'
if backup.is_file():
    old_meta = pd.read_csv(backup, dtype=str).fillna('')
    old_rows = old_meta.loc[~old_meta['file_name'].isin(expected_files)].reset_index(drop=True)
    current_rows = meta.loc[~meta['file_name'].isin(expected_files)].reset_index(drop=True)
    pd.testing.assert_frame_equal(old_rows, current_rows, check_dtype=False)
    print('[PASS] Old metadata records unchanged (compared to backup).')
print('\n[ALL PASS] 29 files match raw data, channel order, long-table format, labels, and official split metadata.')
