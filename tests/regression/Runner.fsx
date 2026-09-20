#!/usr/bin/env dotnet fsi
#load "RunnerCore.fsx"

exit (RunnerCore.main fsi.CommandLineArgs.[1..])
