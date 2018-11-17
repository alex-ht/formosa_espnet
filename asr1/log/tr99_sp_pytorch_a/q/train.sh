#!/bin/bash
cd /workspace/fsw-system/asr1
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
asr_train.py --ngpu 1 --backend pytorch --outdir exp/tr99_sp_pytorch_a/results --debugmode 0 --dict data/lang_1char/tr99_sp_units.txt --debugdir exp/tr99_sp_pytorch_a --minibatches 0 --verbose 0 --resume --train-json dump/tr99_sp/deltatrue/data.json --valid-json dump/cv01/deltatrue/data.json --etype blstmp --elayers 4 --eunits 320 --eprojs 160 --subsample 1_2_2_2 --dlayers 1 --dunits 320 --atype location --aconv-chans 10 --aconv-filts 100 --mtlalpha 0.5 --batch-size 40 --maxlen-in 800 --maxlen-out 150 --sampling-probability 0.0 --opt adadelta --epochs 16 
EOF
) >exp/tr99_sp_pytorch_a/train.log
time1=`date +"%s"`
 ( asr_train.py --ngpu 1 --backend pytorch --outdir exp/tr99_sp_pytorch_a/results --debugmode 0 --dict data/lang_1char/tr99_sp_units.txt --debugdir exp/tr99_sp_pytorch_a --minibatches 0 --verbose 0 --resume --train-json dump/tr99_sp/deltatrue/data.json --valid-json dump/cv01/deltatrue/data.json --etype blstmp --elayers 4 --eunits 320 --eprojs 160 --subsample 1_2_2_2 --dlayers 1 --dunits 320 --atype location --aconv-chans 10 --aconv-filts 100 --mtlalpha 0.5 --batch-size 40 --maxlen-in 800 --maxlen-out 150 --sampling-probability 0.0 --opt adadelta --epochs 16  ) 2>>exp/tr99_sp_pytorch_a/train.log >>exp/tr99_sp_pytorch_a/train.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/tr99_sp_pytorch_a/train.log
echo '#' Finished at `date` with status $ret >>exp/tr99_sp_pytorch_a/train.log
[ $ret -eq 137 ] && exit 100;
touch exp/tr99_sp_pytorch_a/q/sync/done.8910
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH,CUDA_VISIBLE_DEVICES -cwd -S /bin/bash -j y -l arch=*64* -o exp/tr99_sp_pytorch_a/q/train.log -l mem_free=2G,ram_free=2G -l gpu=1   /workspace/fsw-system/asr1/exp/tr99_sp_pytorch_a/q/train.sh >>exp/tr99_sp_pytorch_a/q/train.log 2>&1
