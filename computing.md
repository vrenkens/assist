# Computing

Assist currently has three computing modes: local, condor and condor_gpu

## Local

this is the simplest computing model. Local computing will run on the machine
that is calling the run command

## Condor

Condor computing will use HTConcor to find a machine to run the
code. The job description file can be found in assist/condor/run_script.job.
To use condor you should create a create_environment.sh file which is a script
that will set up your environment (for example sourcing your bashrc file). Next
the original command should be called.

An example of a create_environment.sh file is:

```
#!/bin/sh

#source the bashrc
source ~/.bashrc

#run the original command
$@
```

## Condor gpu

Condor_gpu computing is the same as condor computing but it will look for a
machine that has a GPU. The job description file can be found in
assist/condor/run_script_GPU.job.
