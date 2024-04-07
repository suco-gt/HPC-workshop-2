Assuming you have ssh'd into a login node on ice, your next steps are to request a compute node with a CUDA capable GPU, where  ``[GPU_name]`` is filled with one of ```H100, A100, V100, RTX6000, A40```:
```
    salloc --nodes=1 --ntasks-per-node=1 --gres=gpu:[GPU_name]:1 --mem=0 --time=0:15:00
```
Once you are in, load appropriate modules:
```
    module load cuda/11

    module load gcc/10

    module load intel-oneapi-mkl

    source ${INTEL_ONEAPI_MKLROOT}/setvars.sh

    pip install numpy --user

    pip install numba --user

    pip install --upgrade numpy numba llvmlite --user
```
You can paste these commands into the terminal every time, just to make sure that your dependencies are clear. Try to log on to a cluster using the same GPU each time.

If you're unable to access this through a GPU, a Colaboratory file will be available for you to use otherwise.
