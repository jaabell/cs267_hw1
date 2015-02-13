#!/bin/bash

#make clean
make -f Makefile_hopper benchmark-foo &> compile_log
export jobname=$(qsub job-foo)
echo $jobname
mv compile_log job-foo.$jobname.compile_log
