#!/bin/bash

for n in 3 8 13 19 24 30 36 42 47 52 58 64 70;
do 
	source train_transformer.sh $n
done
