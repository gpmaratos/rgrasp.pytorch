#!/bin/bash

mkdir DATA
W=http://pr.cs.cornell.edu/grasping/rect_data/temp
for F in 01 02 03 04 05 06 07 08 09 10; do curl -O $W/data$F.tar.gz; done
for FILE in *.tar.gz; do tar xf $FILE; done
for FILE in *.tar.gz; do rm $FILE; done
for DIR in 01 02 03 04 05 06 07 08 09 10; do mv $DIR/* DATA/; done
for DIR in 01 02 03 04 05 06 07 08 09 10; do rm -r $DIR; done
