#!/bin/sh
DIR="./dataset"

wget --content-disposition https://www.cs.toronto.edu/~taati/OCW/OCW.tar.gz
tar -xf OCW.tar.gz
cd $DIR