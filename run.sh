#!/bin/bash

#make clean
make -f Makefile_hopper benchmark-blocked_new &> compile_log
export jobname=$(qsub job-foo)
echo $jobname
mv compile_log job-foo.$jobname.compile_log
