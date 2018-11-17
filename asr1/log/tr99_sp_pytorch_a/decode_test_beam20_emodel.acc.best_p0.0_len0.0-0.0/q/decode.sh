#!/bin/bash
cd /workspace/fsw-system/asr1
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
asr_recog.py --ngpu 1 --backend pytorch --debugmode 0 --verbose 1 --recog-json dump/test/deltatrue/split30utt/data.${SGE_TASK_ID}.json --result-label exp/tr99_sp_pytorch_a/decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0/data.${SGE_TASK_ID}.json --model exp/tr99_sp_pytorch_a/results/model.acc.best --beam-size 20 --penalty 0.0 --maxlenratio 0.0 --minlenratio 0.0 --ctc-weight 0.0 
EOF
) >exp/tr99_sp_pytorch_a/decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0/log/decode.$SGE_TASK_ID.log
time1=`date +"%s"`
 ( asr_recog.py --ngpu 1 --backend pytorch --debugmode 0 --verbose 1 --recog-json dump/test/deltatrue/split30utt/data.${SGE_TASK_ID}.json --result-label exp/tr99_sp_pytorch_a/decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0/data.${SGE_TASK_ID}.json --model exp/tr99_sp_pytorch_a/results/model.acc.best --beam-size 20 --penalty 0.0 --maxlenratio 0.0 --minlenratio 0.0 --ctc-weight 0.0  ) 2>>exp/tr99_sp_pytorch_a/decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0/log/decode.$SGE_TASK_ID.log >>exp/tr99_sp_pytorch_a/decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0/log/decode.$SGE_TASK_ID.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/tr99_sp_pytorch_a/decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0/log/decode.$SGE_TASK_ID.log
echo '#' Finished at `date` with status $ret >>exp/tr99_sp_pytorch_a/decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0/log/decode.$SGE_TASK_ID.log
[ $ret -eq 137 ] && exit 100;
touch exp/tr99_sp_pytorch_a/decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0/q/sync/done.23874.$SGE_TASK_ID
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH,CUDA_VISIBLE_DEVICES -cwd -S /bin/bash -j y -l arch=*64* -o exp/tr99_sp_pytorch_a/decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0/q/decode.log -l gpu=1 -l mem_free=4G,ram_free=4G  -t 1:30 /workspace/fsw-system/asr1/exp/tr99_sp_pytorch_a/decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0/q/decode.sh >>exp/tr99_sp_pytorch_a/decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0/q/decode.log 2>&1
