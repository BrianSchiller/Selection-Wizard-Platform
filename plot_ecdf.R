#!/usr/bin/env Rscript
library('IOHanalyzer')
library('plotly')
reticulate::py_run_string('import sys')

# Load all data for f1
dsList_f1 <- DataSetList('data_seeds_organised/f1_Sphere/', verbose = FALSE)
dim <- 10

# Limit to the 6 'normal' algorithms
ds_f1 <- subset(dsList_f1, DIM == dim, funcId == 1, algId == 'CMA' | algId == 'ChainMetaModelPowell' | algId == 'Cobyla' | algId == 'MetaModel' | algId == 'MetaModelOnePlusOne' | algId == 'ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)')

get_FV_overview(ds_f1)
format = '.pdf'
targets = get_ECDF_targets(ds_f1, type = 'bbob', 51)
print(targets)

fig = Plot.RT.ECDF_Single_Func(ds_f1, fstart = 0.00000001, fstop = 100, fstep = 0.5)
name = 'test_D'
file = paste(name, dim, format, sep="")
save_image(fig, file)

fig = Plot.RT.ECDF_Single_Func(ds_f1, fstart = 0.00000001, fstop = 100, fstep = 0.5, show.per_target = TRUE)
name = 'f1_targets_D'
file = paste(name, dim, format, sep="")
save_image(fig, file)

fig = Plot.RT.ECDF_Per_Target(ds_f1, ftarget = 0.00000001)
name = 'f1_target10-8_D'
file = paste(name, dim, format, sep="")
save_image(fig, file)

fig = Plot.RT.ECDF_Per_Target(ds_f1, ftarget = targets)
name = 'f1_targetBBOB_D'
file = paste(name, dim, format, sep="")
save_image(fig, file)
