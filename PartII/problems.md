# Problems

For each of the following problems, you'll be writing the core logic of the GPU Kernel function and decide how data streams & threads will be organized. This sounds complicated, but is not too hard (as far as these examples go!). 

You'll be writing ``Python`` code that closely parallels the algorithmic approach that you would use with low-level ``C`` code written for CUDA. Some of the later examples contain material that we'll discuss later on in the problem session, so don't worry too much if you don't already know the relevant methods.

**Warning** This code looks like Python but it is really CUDA! You cannot use standard python tools like list comprehensions or ask for Numpy properties like shape or size (if you need the size, it is given as an argument). The puzzles only require doing simple operations, basically +, *, simple array indexing, for loops, and if statements. You are allowed to use local variables. If you get an error it is probably because you did something fancy.

Tip: Think of the function call as being run 1 time for each thread. The only difference is that ``cuda.threadIdx.x`` changes each time.

Although this is not relevant until later, recall that a **global read** occurs when you read an array from memory, and similarly a **global write** occurs when you write an array to memory. Ideally, you should reduce the extent to which you perform ``global`` operations, so that your code will have minimal overhead. Additionally, our grader (may) throw an error if you do something not nice!

# Grading

For each problem, you can complete the code directly in ``nano`` or a code editor of your choice. Afterwards, once you've logged onto a node that contains a NVIDIA GPU that is CUDA-capable, you may run your code for the **n**th question as follows.
```
python qn_file.py
```
Make sure to not remove the ```lib.py``` file, as this contains some of the driver code for our grader!

1. Implement a kernel that adds 10 to each position of vector ``a`` and stores it in vector out. You have 1 thread per position.

2. Implement a kernel that adds together each position of ``a`` and ``b`` and stores it in out. You have 1 thread per position.

3. Implement a kernel that adds 10 to each position of ``a`` and stores it in out. You have more threads than positions.

4. Implement a kernel that adds 10 to each position of ``a`` and stores it in out. Input ``a`` is 2D and square. You have more threads than positions.

5. Implement a kernel that adds ``a`` and ``b`` and stores it in out. Inputs ``a`` and ``b`` are vectors. You have more threads than positions.

6. Implement a kernel that adds 10 to each position of ``a`` and stores it in out. You have fewer threads per block than the size of ``a``.

Tip: A block is a group of threads. The number of threads per block is limited, but we can have many different blocks. Variable cuda.blockIdx tells us what block we are in.

7. Implement the same kernel (as the previous question), but in 2d.  You have fewer threads per block than the size of ``a`` in both directions.

8. Implement a kernel that adds 10 to each position of a and stores it in out. You have fewer threads per block than the size of ``a``.

*Warning: Each block can only have a constant amount of shared memory that threads in that block can read and write to. This needs to be a literal python constant not a variable. After writing to shared memory you need to call cuda.syncthreads to ensure that threads do not cross. (This example does not really need shared memory or syncthreads, but it is a demo.)*

9. Implement a kernel that sums together the last 3 position of a and stores it in out. You have 1 thread per position. You only need 1 global read and 1 global write per thread.

*Tip: Remember to be careful about syncing.*

10. Implement a kernel that computes the dot-product of a and b and stores it in out. You have 1 thread per position. You only need 2 global reads and 1 global write per thread.

*Note: For this problem you don't need to worry about number of shared reads. We will handle that challenge later.*