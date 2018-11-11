#!/bin/bash
# Apache 2.0

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 3 ]; then
  echo "Usage: $0 [--remove-archive] <data-base> <url-base> <corpus-part>"
  echo "e.g.: $0 $PWD/lm_data http://www.openslr.org/resources/55 train"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
  echo "<corpus-part> can be one of: train test."
fi

data=$1
url=$2
part=$3

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data"
  exit 1;
fi

part_ok=false
list="train test"
for x in $list; do 
  if [ "$part" == $x ]; then part_ok=true; fi
done
if ! $part_ok; then
  echo "$0: expected <corpus-part> to be one of $list, but got '$part'"
  exit 1;
fi

if [ -z "$url" ]; then
  echo "$0: empty URL base."
  exit 1;
fi

if [ -f $data/CLMAD/$part/.complete ]; then
  echo "$0: data part $part was already successfully extracted, nothing to do."
  exit 0;
fi

data=$data/CLMAD/$part
[ ! -d $data ] && mkdir -p $data

# sizes of the archive files in bytes.  This is some older versions.
sizes_old="1972193280"
# sizes_new is the archive file sizes of the final release.  Some of these sizes are of
# things we probably won't download.
sizes_new="1972193280"

if [ -f $data/$part.tgz ]; then
  size=$(/bin/ls -l $data/$part.tgz | awk '{print $5}')
  size_ok=false
  for s in $sizes_old $sizes_new; do if [ $s == $size ]; then size_ok=true; fi; done
  if ! $size_ok; then
    echo "$0: removing existing file $data/$part.tgz because its size in bytes $size"
    echo "does not equal the size of one of the archives."
    rm $data/$part.tgz
  else
    echo "$data/$part.tgz exists and appears to be complete."
  fi
fi

pushd $data

if [ ! -f $part.tgz ]; then
  if ! which axel >/dev/null; then
    echo "$0: axel is not installed."
    exit 1;
  fi
  full_url=$url/$part.tgz
  echo "$0: downloading data from $full_url.  This may take some time, please be patient."

  if ! axel -n 30 $full_url; then
    echo "$0: error executing axel -n 30 $full_url"
    exit 1;
  fi
fi

if ! tar -xvf $part.tgz; then
  echo "$0: error un-tarring archive $data/$part.tgz"
  exit 1;
fi

popd >&/dev/null

touch $data/.complete

echo "$0: Successfully downloaded and un-tarred $data/$part.tgz"

if $remove_archive; then
  echo "$0: removing $data/$part.tgz file since --remove-archive option was supplied."
  rm $data/$part.tgz
fi
