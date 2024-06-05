# PresentationsYu
Collection of Jupyter notebooks used for presentation at the Institute of Computational Mathematics

For the Fortran notes use the fortranenviron.yml environment

    conda env create -f fortranenviron.yml
    source activate fortran
    python -m ipykernel install --user --name fortran
    jupyter notebook&

The line with ipykernel makes the fortran environment available from Jupyter. When Jupyter starts you should
enable the fortran environment under `Kernel->Change kernel`

For the shenfun presentations use the shenfunenvironment.yml instead of fortranenviron.yml

    conda deactivate
    conda env create -f shenfunenviron.yml
    source activate shenfun
    python -m ipykernel install --user --name shenfun
    jupyter notebook&
