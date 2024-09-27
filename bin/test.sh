#!/bin/bash

for file in data/input/*.zipzip; do
    if [ -f $file ]; then
        echo $file
    fi
done
