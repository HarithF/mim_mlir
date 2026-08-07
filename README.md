<p align="center">
  <h2 align="center">MLIR Mim Emitter</h2>
</p>

<p align="center">
  <b>MLIR Code Generation</b> for <b>MimIR</b>
</p>

The **mlir** plugin emits [MLIR](https://mlir.llvm.org/) from a partially optimized
**MimIR** world, instead of going through MimIR's native LLVM backend (`ll`). It lowers
MimIR's functional, CPS-style IR, mainly targeting tensor `map_reduce`/`broadcast` operations
and `affine.For` loops, into the `func`, `arith`, `math`, `scf`, `linalg`, and `tensor`
MLIR dialects, so MimIR programs can flow into the MLIR ecosystem (further dialect
lowering, bufferization, MLIR's own optimizer, GPU/accelerator backends, etc.) instead of straight to
machine code.

Because this is a third-party plugin, it does not run as part of MimIR's default
compilation pipeline. It is opt-in: pass `-p mlir` to `mim` and it will be loaded and
used in place of the standard pipeline.

## Usage

```bash
./mim <file>.mim -p mlir
```

 The plugin overrides the standard `_default_compile` pipeline with its own, 
 which runs the usual front-end simplification passes and then emits MLIR instead.

Given an input `file.mim`, the plugin writes the result to `file.mlir` in the
current working directory (or `a.mlir` if the world has no name).

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
    %result.buf = tensor.empty() : tensor<2x2xf32>
    %result.buf.out = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%A, %B : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%result.buf : tensor<2x2xf32>) {
        ^bb0(%in_0: f32, %in_1: f32, %acc_2: f32):
          %v3 = arith.mulf %in_0, %in_1 : f32
          %v4 = arith.addf %v3, %acc_2 : f32
          linalg.yield %v4 : f32
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
git https://github.com/HarithF/mim_mlir
cd ..
```

MimIR automatically picks up any repository in `extra/` that has a `CMakeLists.txt`
as a direct child — no changes to the main project are needed.

**3. Build the project according to the [instructions](https://mimir.github.io/index.html#autotoc_md92)**

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DMIM_BUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
```

This builds `mim_mlir` alongside the rest of MimIR's plugins. The plugin 
is loaded dynamically at runtime via `-p mlir`.
## Supported Constructs

| MimIR construct                                   | MLIR target                          |
|----------------------------------------------------|---------------------------------------|
| Functions (`fun extern ...`)                       | `func.func` / `func.return`           |
| `%core.wrap` (add/sub/mul/shl)                      | `arith` binary integer ops            |
| `%core.div` (sdiv/udiv/srem/urem)                   | `arith` binary integer ops            |
| `%core.icmp`                                        | `arith.cmpi`                          |
| `%math.arith` (add/sub/mul/div/rem)                 | `arith` binary float ops              |
| `%math.tri` (tanh/sin/cos)                          | `math` unary ops                      |
| `%math.extrema` (fmax/fmin/ieee754max/ieee754min)   | `arith` binary float ops              |
| `%affine.For`                                       | `scf.for`                             |
| `%tensor.map_reduce`                                | `linalg.generic`                      |
| `%tensor.broadcast`                                 | `linalg.broadcast`                    |
| Literal tensors                                     | `arith.constant` (dense attribute)    |

This list will grow as more of MimIR's plugin surface is covered.

## Limitations

- This plugin is loaded standalone via `-p mlir` and overrides `_default_compile`
  entirely. It is not designed to be combined with other plugins that also
  redefine the default pipeline.
- Coverage of MimIR's axiom set is partial (see the table above). Unhandled
  constructs will currently abort with a diagnostic identifying the
  unsupported `Def`.

