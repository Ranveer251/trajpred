#!/usr/bin/env bash

cd ../trajnetplusplusbaselines-1
pip install -e .
cd ../trajnetpluslusdataset
pip install -e .
pip install -e '.[test, plot]'
cd ../socialforce-main
pip install -e .
cd ../
wget https://github.com/sybrenstuvel/Python-RVO2/archive/master.zip
unzip master.zip
rm master.zip
cd Python-RVO2-main/
pip install cmake
pip install cython
python setup.py build
python setup.py install
cd ../trajnetplusplusbaselines-1
