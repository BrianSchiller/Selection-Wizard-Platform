# Format description
In all cases an empty field represents something is unknown.

## algorithms.csv
Descriptions of fields in algorithms.csv and their possible values.
### ID
Unique integer identifier of the algorithm.
### name
String name of the algorithm.
### implementation
URL to exact commit in a public repository.
### scientific reference
DOI or URL to primary scientific reference for this algorithm.
### deterministic
Boolean (true/false) indicating whether the algorithm is deterministic or not.
### parameter values
String of parameter values.
### Properties
Additional properties are described in the properties section, since they are shared with problems.

## problems.csv
Descriptions of fields in problems.csv and their possible values.
### ID
Unique integer identifier of the problem.
### name
String name of the problem.
### implementation
URL to exact commit in a public repository.
### scientific reference
DOI or URL to primary scientific reference for this problem.
### Properties
Additional properties are described in the properties section, since they are shared with algorithms.

## Properties
Descrition of property fields shared between algorithms.csv and problems.csv and their possible values.
### continuous
Boolean (true/false) indicating whether the algorithm/problem handles/includes continuous variables.
### number of variables
Integer indicating the number of variables; for algorithms this is assumed to be the maximum number of variables it can handle.
Alternatively, it can be an integer range denoted as 1-2, where the first number is the minimum number of variables, and the second number the maximum (both inclusive).
### evaluation budget
Integer indicating the number of function evaluations the algorithm/problem allows.
Alternatively, it can be an integer range denoted as 1-2, where the first number is the minimum number of evaluations, and the second number the maximum (both inclusive).
### number of objectives
Integer indicating the number of objectives the algorithm/problem handles/has.
Alternatively, it can be an integer range denoted as 1-2, where the first number is the minimum number of objectives, and the second number the maximum (both inclusive).

## performance_data.csv
Description of fields in performance_data.csv.
### problem ID
ID of the problem, must match one found in problems.csv.
### algorithm ID
ID of the algorithm, must match one found in algorithms.csv.
### algorithm seed
Seed or other unique identifier for the algorithm run.

## runs/<PID>_<AID>_<seed>.csv
Description of fields of files in the runs/ directory. Each of these files is named according to the <PID>_<AID>_<seed>.csv format, which associates each file with a specific problem, algorithm, and algorithm seed. These files only record rows where performance improved over the previous best found quality.
### Evaluation number
Number of evaluations used so far.
### Quality
Best quality found at this number of evaluations.
