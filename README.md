# Current benchmarks

```
Benchmark 1: ~/software/josh/target/release/josh
  Time (mean ± σ):     23.240 s ±  0.327 s    [User: 23.191 s, System: 0.007 s]
  Range (min … max):   22.497 s … 23.775 s    10 runs

Benchmark 2: /lhome/vincentva/software/gyre/bin/gyre inlist_gl2
  Time (mean ± σ):     33.866 s ±  1.675 s    [User: 33.308 s, System: 0.019 s]
  Range (min … max):   31.970 s … 38.282 s    10 runs

  Warning: The first benchmarking run for this command was significantly slower than the rest (38.282 s). This could be caused by (filesystem) caches that were not filled until after the first run. You are already using the '--warmup' option which helps to fill these caches before the actual benchmark. You can either try to increase the warmup count further or re-run this benchmark on a quiet system in case it was a random outlier. Alternatively, consider using the '--prepare' option to clear the caches before each timing run.

Benchmark 3: /lhome/vincentva/software/gyre/bin/gyre inlist_gl2_noband
  Time (mean ± σ):     34.624 s ±  0.226 s    [User: 34.015 s, System: 0.037 s]
  Range (min … max):   34.326 s … 35.016 s    10 runs

Benchmark 4: /lhome/vincentva/software/gyre/bin/gyre inlist_gl6
  Time (mean ± σ):     42.397 s ±  8.731 s    [User: 38.933 s, System: 0.020 s]
  Range (min … max):   38.704 s … 67.139 s    10 runs

  Warning: Statistical outliers were detected. Consider re-running this benchmark on a quiet system without any interferences from other programs. It might help to use the '--warmup' or '--prepare' options.

Summary
  '~/software/josh/target/release/josh' ran
    1.46 ± 0.07 times faster than '/lhome/vincentva/software/gyre/bin/gyre inlist_gl2'
    1.49 ± 0.02 times faster than '/lhome/vincentva/software/gyre/bin/gyre inlist_gl2_noband'
    1.82 ± 0.38 times faster than '/lhome/vincentva/software/gyre/bin/gyre inlist_gl6'
```
