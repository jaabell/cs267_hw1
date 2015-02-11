#!/bin/bash

make clean
make -f Makefile_hopper benchmark-blocked &> compile_log
export jobname=$(qsub job-blocked)
echo $jobname
mv compile_log job-blocked.$jobname.compile_log
