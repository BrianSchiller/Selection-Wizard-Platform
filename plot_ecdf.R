#!/usr/bin/env Rscript
library('IOHanalyzer')
library('plotly')
reticulate::py_run_string("import sys")

dsList <- DataSetList('data_seeds_organised/f1_Sphere/CMA/')

ds <- subset(dsList, DIM == 5, funcId == 1)

fig = Plot.RT.ECDF_Single_Func(ds, fstart = 0, fstop = 0.5, fstep = 10)
save_image(fig, 'test.pdf')
