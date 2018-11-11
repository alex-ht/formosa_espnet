#!/bin/bash
. ./cmd.sh

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=0
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=true

# network archtecture
# encoder related
etype=blstmp # encoder architecture type
elayers=4
eunits=320
eprojs=160
subsample=1_2_2_2 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=320
# attention related
atype=location
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=40
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=16

# rnnlm related
lm_layers=2
lm_units=320
lm_opt=sgd        # or adam
lm_batchsize=256  # batch size in LM training
lm_epochs=8      # if the data size is large, we can reduce this
lm_maxlen=100     # if sentence length > lm_maxlen, lm_batchsize is automatically reduced
lm_resume=        # specify a snapshot file to resume LM training
lmtag="a"         # tag for managing LMs

# decoding parameter
lm_weight=1.0
beam_size=5
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.5
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# scheduled sampling option
samp_prob=0.0

# exp tag
tag="a" # tag for managing experiments.

. ./path.sh
. ./utils/parse_options.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -o pipefail

train_set=tr99
train_dev=cv01
recog_set="test"
if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/prepare_data.sh
    utils/subset_data_dir_tr_cv.sh --cv-spk-percent 1 data/train data/$train_set data/$train_dev
    utils/data/perturb_data_dir_speed_3way.sh data/$train_set data/${train_set}_sp
    utils/data/perturb_data_dir_volume.sh --scale-high 1.5 --scale-low 0.5 data/${train_set}_sp
    ### download CLMAD
    [ ! -d ./lm_data ] && mkdir -p lm_data
    local/download_and_untar.sh ./lm_data http://www.openslr.org/resources/55 train
fi
train_set=${train_set}_sp
feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; [ ! -d ${feat_tr_dir} ] && mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; [ ! -d ${feat_dt_dir} ] && mkdir -p ${feat_dt_dir}

if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    # Generate the mfcc features
    for x in $train_set $train_dev ${recog_set}; do
        steps/make_mfcc.sh --cmd "$train_cmd" --nj 32 \
          --write-utt2num-frames true data/${x}
        utils/fix_data_dir.sh data/${x}
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 10 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 10 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"

if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char
    echo "make a non-linguistic symbol list"
    touch ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    fi
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

expdir=exp/${train_set}_${backend}_${tag}
mkdir -p ${expdir}

if [ ${stage} -le 3 ]; then
    echo "stage 3: Network Training"
    ${train_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --sampling-probability ${samp_prob} \
        --opt ${opt} \
        --epochs ${epochs}
fi

if [ ${stage} -le 4 ]; then
    nj=30
    echo "stage 4: Decoding"
    for rtask in ${recog_set}; do
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu 0 \
            --backend ${backend} \
            --debugmode 0 \
            --verbose 1 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight}

        score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}
    done
fi

lmexpdir=exp/train_rnnlm_${backend}_${lmtag}
mkdir -p ${lmexpdir}

if [ ${stage} -le 5 ]; then
    echo "stage 5: LM Preparation"
    lmdatadir=data/local/lm_train
    mkdir -p ${lmdatadir}
    text2token.py -s 1 -n 1 -l ${nlsyms} --space '' data/${train_set}/text | cut -f 2- -d" " \
        > ${lmdatadir}/train.txt
    cat lm_data/CLMAD/train/*.train | cconv -f UTF8 -t UTF8-TW > ${lmdatadir}/clmad.txt \
    text2token.py -s 1 -n 1 -l ${nlsyms} --space '' ${lmdatadir}/clmad.txt \
        >> ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1 -l ${nlsyms} --space '' data/${train_dev}/text | cut -f 2- -d" " \
        > ${lmdatadir}/valid.txt
    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi
    ${train_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --layer ${lm_layers} \
        --unit ${lm_units} \
        --opt ${lm_opt} \
        --batchsize ${lm_batchsize} \
        --epoch ${lm_epochs} \
        --maxlen ${lm_maxlen} \
        --dict ${dict}
fi

if [ ${stage} -le 6 ]; then
    nj=30
    echo "stage 6: Decoding"
    for rtask in ${recog_set}; do
        decode_dir=decode_rnnlm_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode 0 \
            --verbose 1 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            ${recog_opts}

        score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}
    done
fi
echo "Finished"
exit 0;
