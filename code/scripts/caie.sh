cd ../trajnetplusplusbaselines-1/
cp trajnetbaselines/lstm_files/lstm_ca.py trajnetbaselines/lstm/lstm.py
python -m trajnetbaselines.lstm.trainer --type directional --augment --intent