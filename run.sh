#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./pyin_hz_data/train.pyin --train-tgt=./pyin_hz_data/train.hz --dev-src=./pyin_hz_data/dev.pyin --dev-tgt=./pyin_hz_data/dev.hz --vocab=vocab.json --cuda
elif [ "$1" = "test" ]; then
  if [ "$2" = "" ]; then
      CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./pyin_hz_data/test.pyin ./pyin_hz_data/test.hz outputs/test_outputs.txt --cuda
  else
      CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./pyin_hz_data/test.pyin ./pyin_hz_data/test.hz outputs/test_outputs.txt --cuda --beam-size="$2"
  fi
elif [ "$1" = "train_local" ]; then
	python run.py train --train-src=./pyin_hz_data/train.pyin --train-tgt=./pyin_hz_data/train.hz --dev-src=./pyin_hz_data/dev.pyin --dev-tgt=./pyin_hz_data/dev.hz --vocab=vocab.json
elif [ "$1" = "test_local" ]; then
    python run.py decode model.bin ./pyin_hz_data/test.pyin ./pyin_hz_data/test.hz outputs/test_outputs.txt
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=./pyin_hz_data/train.pyin --train-tgt=./pyin_hz_data/train.hz vocab.json
elif [ "$1" = "decode" ]; then
    if [ "$2" != "" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin "$2" --cuda
    else echo "error parameters"
    fi
else
	echo "Invalid Option Selected"
fi
