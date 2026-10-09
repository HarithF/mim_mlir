<p align="center">
  <h2 align="center">MLIR Mim Emitter</h2>
</p>

<p align="center">
  <b>MLIR Code Generation</b> for <b>MimIR</b>
</p>

The **mlir** plugin emits [MLIR](https://mlir.llvm.org/) from a partially optimized **MimIR** world instead of going through MimIR's LLVM backend (`ll`).
It lowers tensor operations and `affine.For` loops into the `func`, `arith`, `math`, `scf`, `linalg`, and `tensor` dialects, so MimIR programs can continue through the MLIR ecosystem.

## Usage

```bash
./mim <file>.mim -p mlir
```

The plugin defines its own `_compile` pipeline, which runs the usual simplification phases, lowers and fuses tensor operations, and then emits MLIR.
Given `file.mim`, it writes `file.mlir` to the current working directory (or `a.mlir` if the world has no name).

Tensor parameters of extern functions should be wrapped in `tensor.buf`; otherwise `compile.scalarize` may split small arrays into scalars.

```bash
./mim lit/matmul.mim -p mlir
cat matmul.mlir
```

### Example

```mlir
module {
  func.func public @main() -> tensor<2x2xf32> {
    %A = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %B = arith.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf32>
    %result.empty = tensor.empty() : tensor<2x2xf32>
    %v0 = arith.constant 0.0 : f32
    %result.buf = linalg.fill ins(%v0 : f32) outs(%result.empty : tensor<2x2xf32>) -> tensor<2x2xf32>
    %result.shape = tensor.empty() : tensor<2x1xf32>
    %result.buf.out = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2 + d3)>, affine_map<(d0, d1, d2, d3) -> (d2 + d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
      ins(%A, %B, %result.shape : tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x1xf32>)
      outs(%result.buf : tensor<2x2xf32>) {
        ^bb0(%in_2: f32, %in_3: f32, %shape_1: f32, %acc_4: f32):
          %v5 = arith.mulf %in_2, %in_3 : f32
          %v6 = arith.addf %v5, %acc_4 : f32
          linalg.yield %v6 : f32
      }
    -> tensor<2x2xf32>
    func.return %result.buf.out : tensor<2x2xf32>
  }
}
```

## Installation

**1. Clone the `mimir` repository if you haven't already**

```bash
git clone --recursive https://github.com/mimir/mimir.git
```

**2. Clone this repository into `mimir/extra`**

```bash
cd mimir/extra
git clone https://github.com/HarithF/mim_mlir
cd ..
```

MimIR picks up any directory in `extra/` with a `CMakeLists.txt`; no changes to the main project are needed.

**3. Build the project according to the [instructions](https://mimir.github.io/index.html#autotoc_md92)**

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DMIM_BUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
```

This builds `mim_mlir` alongside MimIR's other plugins; it is loaded at runtime via `-p mlir`.

## Supported Constructs

| MimIR construct                                  | MLIR target                        |
|--------------------------------------------------|------------------------------------|
| Functions (`extern fun ...`)                     | `func.func` / `func.return`        |
| `core.wrap` (add/sub/mul/shl)                    | `arith` binary integer ops         |
| `core.div` (sdiv/udiv/srem/urem)                 | `arith` binary integer ops         |
| `core.icmp`                                      | `arith.cmpi`                       |
| `math.arith` (add/sub/mul/div/rem)               | `arith` binary float ops           |
| `math.tri` (tanh/sin/cos)                        | `math` unary ops                   |
| `math.rt` (sq/cb)                                | `math.sqrt` / `math.cbrt`          |
| `math.exp` (exp/exp2/log/log2/log10)             | `math` unary ops                   |
| `math.extrema` (fmax/fmin/ieee754max/ieee754min) | `arith` binary float ops           |
| `affine.For`                                     | `scf.for`                          |
| `tensor.map_reduce`                              | `linalg.generic`                   |
| `tensor.broadcast`                               | `linalg.broadcast`                 |
| Extract with a runtime index                     | `tensor.extract`                   |
| Literal tensors                                  | `arith.constant` (dense attribute) |

This list will grow as more of MimIR's plugin surface is covered.

## Limitations

- The plugin can be loaded alongside others (e.g. `-p opt -p mlir`): its `_compile` takes precedence over a plugin's `_default_compile`, but a file or plugin that defines its own `_compile` clashes with it.
- Coverage of MimIR's axioms is partial (see the table above); unhandled constructs abort with a diagnostic naming the unsupported `Def`.
