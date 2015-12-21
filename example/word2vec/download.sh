#!/bin/sh
FILE=text8

if [ ! -f $FILE ]; then
  mkdir -p data
  wget http://mattmahoney.net/dc/${FILE}.zip -O ./data/${FILE}.zip
  cd data/
  unzip ${FILE}.zip
  rm -rf ${FILE}.zip
  cd ..
fi

