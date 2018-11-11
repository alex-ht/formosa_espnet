#!/bin/bash

set -e -o pipefail

corpus=NER-Trs-Vol1/Train

if [ -z "$corpus" ] ; then
    echo >&2 "The script $0 expects one parameter -- the location of the NER-Trs-Vol1 corpus"
    exit 1
fi
if [ ! -d "$corpus" ] ; then
    echo >&2 "The directory $corpus does not exist"
fi

echo "Preparing train and test data"

# have to remvoe previous files to avoid filtering speakers according to cmvn.scp and feats.scp
rm -rf        data/all data/train data/test data/local/train
mkdir -p data data/all data/train data/test data/local/train

#
# make utt2spk, wav.scp and text
#

rm -f data/all/utt2spk
rm -f data/all/wav.scp
rm -f data/all/text
touch data/all/utt2spk
touch data/all/wav.scp
touch data/all/text
for x in $corpus/*/Wav/*/*/*.wav; do
  key=$(basename -s .wav $x);
  echo "$key $key" >> data/all/utt2spk
  echo "$key $x"   >> data/all/wav.scp
  txt_path=$(echo $x |  sed 's:Wav:Text:g' |  sed 's:.wav:.txt:g')
  sed '1s/^\uFEFF//' $txt_path | \
    uconv -x any-nfc | dos2unix | \
    awk -v key=$key '{print key " " $n}' >> data/all/text
done
utils/fix_data_dir.sh data/all

#
# test set: JZ, GJ, KX, YX 
#
rm -f data/all/testset.txt
touch data/all/testset.txt
echo "JZ"  > data/all/testset.txt
echo "GJ" >> data/all/testset.txt
echo "KX" >> data/all/testset.txt
echo "YX" >> data/all/testset.txt

# training set
rm -f data/train/utt2spk
rm -f data/train/wav.scp
rm -f data/train/text
touch data/train/utt2spk
touch data/train/wav.scp
touch data/train/text
grep -v -F -f data/all/testset.txt data/all/text    > data/train/text    || exit 1;
grep -v -F -f data/all/testset.txt data/all/wav.scp > data/train/wav.scp || exit 1;
grep -v -F -f data/all/testset.txt data/all/utt2spk > data/train/utt2spk || exit 1;

# test set
rm -f data/test/utt2spk
rm -f data/test/wav.scp
rm -f data/test/text
touch data/test/utt2spk
touch data/test/wav.scp
touch data/test/text
grep    -F -f data/all/testset.txt data/all/text    > data/test/text    || exit 1;
grep    -F -f data/all/testset.txt data/all/wav.scp > data/test/wav.scp || exit 1;
grep    -F -f data/all/testset.txt data/all/utt2spk > data/test/utt2spk || exit 1;

#
# fix data format
#
for x in train test; do
    # fix_data_dir.sh fixes common mistakes (unsorted entries in wav.scp,
    # duplicate entries and so on). Also, it regenerates the spk2utt from
    # utt2sp
    utils/fix_data_dir.sh data/$x
done

echo "Data preparation completed."
exit 0;
