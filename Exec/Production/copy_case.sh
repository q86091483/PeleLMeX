#! /usr/bin/env bash

SRC=$1/
DST=$2/

rsync -aP --exclude=archive* \
          --exclude=tmp* \
          --exclude=core* \
          --exclude=Backtrace* \
          --exclude=*.{o,e}[0-9]* \
          --exclude=chk* \
          --exclude=plt* \
          $SRC $DST
