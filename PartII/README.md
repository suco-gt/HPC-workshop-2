Assuming you have ssh'd into a login node on ice, your next steps are to request a compute node with a CUDA capable GPU, where  ``[GPU_name]`` is filled with one of ```H100, A100, V100, RTX6000, A40```:
```
    salloc --nodes=1 --ntasks-per-node=1 --gres=gpu:[GPU_name]:1 --mem=0 --time=0:15:00
```
Once you are in, load appropriate modules:
```
module load anaconda3/2022.05.0.1 
module load cuda/11
module load gcc/10
module load intel-oneapi-mkl
source ${INTEL_ONEAPI_MKLROOT}/setvars.sh
```
You can paste these commands into the terminal every time to load the correct modules. Try to log on to a cluster using the same GPU each time.

The first time that you log onto the cluster, please run the following commands to install the correct dependencies for the project.

```
python -m pip install numpy --user
python -m pip install numba --user
python -m pip install --upgrade numpy numba llvmlite --user
python -m pip install -qqq git+https://github.com/danoneata/chalk@srush-patch-1
```

If you're unable to access this through a GPU, a Colaboratory file will be available for you to use otherwise.
