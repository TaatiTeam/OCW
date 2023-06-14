#!/bin/sh
DIR="./dataset"

wget --content-disposition https://www.cs.toronto.edu/~taati/OCW/OCW.tar.gz
tar -xf OCW.tar.gz
mv OCW $DIR
rm -rf OCW.tar.gz OCW
cd $DIR