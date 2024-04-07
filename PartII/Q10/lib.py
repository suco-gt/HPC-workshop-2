from dataclasses import dataclass
import numpy as np
from chalk import *
from colour import Color
import chalk
from dataclasses import dataclass
from typing import List, Any
from collections import Counter
from numba import cuda
import numba
import random

@dataclass
class ScalarHistory:
    last_fn: str
    inputs: list

    def __radd__(self, b):
        return self + b

    def __add__(self, b):
        if isinstance(b, (float, int)):
            return self
        if isinstance(b, Scalar):
            return ScalarHistory(self.last_fn, self.inputs + [b])

        return ScalarHistory(self.last_fn, self.inputs + b.inputs)
        
class Scalar:
    def __init__(self, location):
        self.location = location

    def __mul__(self, b):
        if isinstance(b, (float, int)):
            return ScalarHistory("id", [self])
        return ScalarHistory("*", [self, b])


    def __radd__(self, b):
        return self + b
        
    def __add__(self, b):
        if isinstance(b, (float, int)):
            return ScalarHistory("id", [self])
        return ScalarHistory("+", [self, b])

    def __iadd__(self, other):
        assert False, "Instead of `out[] +=` use a local variable `acc + =`"
    
class Table:
    def __init__(self, name, array):
        self.name = name
        self.incoming = []
        self.array = array

        self.size = array.shape
    
    def __getitem__(self, index):
        self.array[index]
        if isinstance(index, int):
            index = (index,)
        assert len(index) == len(self.size), "Wrong number of indices"
        if index[0] >= self.size[0]:
            assert False, "bad size"

        return Scalar((self.name,) + index)

    def __setitem__(self, index, val):
        self.array[index]
        if isinstance(index, int):
            index = (index,)
        assert len(index) == len(self.size), "Wrong number of indices"
        if index[0] >= self.size[0]:
            assert False, "bad size"
        if isinstance(val, Scalar):
            val = ScalarHistory("id", [val])
        if isinstance(val, float):
            return
        self.incoming.append((index, val))

@dataclass(frozen=True, eq=True)
class Coord:
    x: int
    y: int

    def enumerate(self):
        k = 0
        for i in range(self.y):
            for j in range(self.x):
                yield k, Coord(j, i)
                k += 1

    def tuple(self):
        return (self.x, self.y)


class RefList:
    def __init__(self):
        self.refs = []
        
    def __getitem__(self, index):
        return self.refs[-1][index]

    def __setitem__(self, index, val):
        self.refs[-1][index] = val


class Shared:
    def __init__(self, cuda):
        self.cuda = cuda

    def array(self, size, ig):
        if isinstance(size, int):
            size = (size,)
        s = np.zeros(size)
        cache = Table("S" + str(len(self.cuda.caches)), s)
        # self.caches.append(cache)
        self.cuda.caches.append(RefList())
        self.cuda.caches[-1].refs = [cache]
        self.cuda.saved.append([])
        return self.cuda.caches[-1]


class Cuda:
    blockIdx: Coord
    blockDim: Coord
    threadIdx: Coord
    caches: list
    shared: Shared

    def __init__(self, blockIdx, blockDim, threadIdx):
        self.blockIdx = blockIdx
        self.blockDim = blockDim
        self.threadIdx = threadIdx
        self.caches = []
        self.shared = Shared(self)
        self.saved = []

    def syncthreads(self):
        for i, c in enumerate(self.caches):
            old_cache = c.refs[-1]
            # self_links = cache.self_links()
            # cache.clean()
            temp = old_cache.incoming
            old_cache.incoming = self.saved[i]
            self.saved[i] = temp
            cache = Table(old_cache.name + "'", old_cache.array)

            c.refs.append(cache)

    def finish(self):
        for i, c in enumerate(self.caches):
            old_cache = c.refs[-1]
            old_cache.incoming = self.saved[i]

    def rounds(self):
        if len(self.caches) > 0:
            return len(self.caches[0].refs)
        else:
            return 0

@dataclass
class CudaProblem:
    name: str
    fn: Any
    inputs: List[np.ndarray]
    out: np.ndarray
    args: Tuple[int] = ()
    blockspergrid: Coord = Coord(1, 1)
    threadsperblock: Coord = Coord(1, 1)
    spec: Any = None
        
    def run_cuda(self):
        fn = self.fn
        fn = fn(numba.cuda)
        jitfn = numba.cuda.jit(fn)
        jitfn[self.blockspergrid.tuple(), self.threadsperblock.tuple()](
            self.out, *self.inputs, *self.args
        )
        return self.out

    def run_python(self):
        results = {}
        fn = self.fn
        for _, block in self.blockspergrid.enumerate():
            results[block] = {}
            for tt, pos in self.threadsperblock.enumerate():
                a = []
                args = ["a", "b", "c", "d"]
                for i, inp in enumerate(self.inputs):
                    a.append(Table(args[i], inp))
                out = Table("out", self.out)

                c = Cuda(block, self.threadsperblock, pos)
                fn(c)(out, *a, *self.args)
                c.finish()
                results[block][pos] =  (tt, a, c, out)
        return results

    def score(self, results):

        total = 0
        full = Counter()
        for pos, (tt, a, c, out) in results[Coord(0, 0)].items():
            total += 1
            count = Counter()
            for out, tab in [(False, c2.refs[i]) for i in range(1, c.rounds()) for c2 in c.caches] + [(True, out)]:
                for inc in tab.incoming:
                    if out:
                        count["out_writes"] += 1
                    else:
                        count["shared_writes"] += 1
                    for ins in inc[1].inputs:
                        if ins.location[0].startswith("S"):
                            count["shared_reads"] += 1
                        else:
                            count["in_reads"] += 1
            for k in count:
                if count[k] > full[k]:
                    full[k] = count[k]
        print(f"""# {self.name}
 
   Score (Max Per Thread):
   | {'Global Reads':>13} | {'Global Writes':>13} | {'Shared Reads' :>13} | {'Shared Writes' :>13} |
   | {full['in_reads']:>13} | {full['out_writes']:>13} | {full['shared_reads']:>13} | {full['shared_writes']:>13} | 
""") 
    
    def show(self, sparse=False):
        results = self.run_python()
        self.score(results)
        return draw_results(results, self.name,
                            self.threadsperblock.x, self.threadsperblock.y, sparse)
    
    def check(self):
        x = self.run_cuda()
        y = self.spec(*self.inputs)
        try:
            np.testing.assert_allclose(x, y)
            print("Passed Tests!")
            from IPython.display import HTML
            pups = [
            "2m78jPG",
            "pn1e9TO",
            "MQCIwzT",
            "udLK6FS",
            "ZNem5o3",
            "DS2IZ6K",
            "aydRUz8",
            "MVUdQYK",
            "kLvno0p",
            "wScLiVz",
            "Z0TII8i",
            "F1SChho",
            "9hRi2jN",
            "lvzRF3W",
            "fqHxOGI",
            "1xeUYme",
            "6tVqKyM",
            "CCxZ6Wr",
            "lMW0OPQ",
            "wHVpHVG",
            "Wj2PGRl",
            "HlaTE8H",
            "k5jALH0",
            "3V37Hqr",
            "Eq2uMTA",
            "Vy9JShx",
            "g9I2ZmK",
            "Nu4RH7f",
            "sWp0Dqd",
            "bRKfspn",
            "qawCMl5",
            "2F6j2B4",
            "fiJxCVA",
            "pCAIlxD",
            "zJx2skh",
            "2Gdl1u7",
            "aJJAY4c",
            "ros6RLC",
            "DKLBJh7",
            "eyxH0Wc",
            "rJEkEw4"]
            return HTML("""
            <video alt="test" controls autoplay=1>
                <source src="https://openpuppies.com/mp4/%s.mp4"  type="video/mp4">
            </video>
            """%(random.sample(pups, 1)[0]))
            
        except AssertionError:
            print("Failed Tests.")
            print("Yours:", x)
            print("Spec :", y)