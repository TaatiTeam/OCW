#!/bin/sh
DIR="./OCW_wordnet"

wget --content-disposition https://www.cs.toronto.edu/~taati/OCW/OCW_wordnet.tar.gz
tar -xf OCW_wordnet.tar.gz
cd $DIR