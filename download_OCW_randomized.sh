#!/bin/sh
DIR="./OCW_randomized"

wget --content-disposition https://www.cs.toronto.edu/~taati/OCW/OCW_randomized.tar.gz
tar -xf OCW_randomized.tar.gz
cd $DIR