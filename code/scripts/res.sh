cd ../trajnetplusplusbaselines-1/
cp trajnetbaselines/lstm_files/lstm_orig.py trajnetbaselines/lstm/lstm.py
python -m trajnetbaselines.lstm.trainer --type directional --augment --residual