#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: addSubject label"
    exit 1
fi

cd capture
./takeFaces ../custom_faces/ $1
cd ..

python create_csv.py custom_faces > custom_faces.csv

./lbp custom_faces.csv custom_model
