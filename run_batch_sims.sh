#!/bin/bash

for i in $(seq 1000); do
    out=run-${i}.sqlite
    cyclus -o $out $1
done;

