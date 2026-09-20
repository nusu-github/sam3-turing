# All image candidates

Latency is truck.jpg + truck; output differences cover the listed number of cases.

| Candidate | ms | CUDA allocated GiB | NVML GiB | Cases | Masks | Mean mask IoU vs stock | Changed pixels |
|---|---:|---:|---:|---:|---:|---:|---:|
| [final_fast560](final_fast560.json) | 95.08 | 1.964 | 3.062 | 5 | 15 | 0.933582 | 18311 |
| [final560](final560.json) | 96.13 | 2.002 | 3.003 | 5 | 15 | 0.933510 | 18324 |
| [final_fast672](final_fast672.json) | 101.61 | 1.999 | 3.088 | 5 | 15 | 0.954241 | 13419 |
| [final672](final672.json) | 106.59 | 2.051 | 3.052 | 5 | 15 | 0.954231 | 13431 |
| [final_fast784](final_fast784.json) | 125.12 | 2.035 | 3.161 | 5 | 15 | 0.971110 | 9333 |
| [combined784](combined784.json) | 143.16 | 2.111 | 3.177 | 5 | 15 | 0.971126 | 9333 |
| [autotune](autotune.json) | 149.62 | 2.141 | 3.409 | 5 | 15 | 0.999207 | 281 |
| [final_fixed_compiled](final_fixed_compiled.json) | 152.42 | 1.228 | 2.429 | 5 | 15 | 0.999197 | 286 |
| [final_compiled_early](final_compiled_early.json) | 153.06 | 2.124 | 3.345 | 5 | 15 | 0.999197 | 286 |
| [final_compiled_pixel](final_compiled_pixel.json) | 153.24 | 2.126 | 3.345 | 5 | 15 | 0.999197 | 286 |
| [final_compiled_channels](final_compiled_channels.json) | 153.47 | 2.122 | 3.339 | 5 | 15 | 0.999190 | 286 |
| [final_compiled](final_compiled.json) | 153.53 | 2.125 | 3.345 | 5 | 15 | 0.999197 | 286 |
| [compiled_trunk](compiled_trunk.json) | 158.71 | 2.260 | 3.577 | 5 | 15 | 0.999194 | 284 |
| [combined896](combined896.json) | 160.26 | 2.170 | 3.313 | 5 | 15 | 0.974548 | 8411 |
| [compiled_head](compiled_head.json) | 168.14 | 2.125 | 3.345 | 5 | 15 | 0.999181 | 292 |
| [combined_pixel](combined_pixel.json) | 172.46 | 2.248 | 3.325 | 5 | 15 | 0.999177 | 289 |
| [combined_noearly](combined_noearly.json) | 173.66 | 2.248 | 3.405 | 5 | 15 | 0.999177 | 289 |
| [combined](combined.json) | 174.23 | 2.248 | 3.405 | 5 | 15 | 0.999177 | 289 |
| [resolution784](resolution784.json) | 174.79 | 2.115 | 3.218 | 3 | 11 | 0.978435 | 7122 |
| [final_eager](final_eager.json) | 174.94 | 2.249 | 3.405 | 5 | 15 | 0.999177 | 289 |
| [combined_tanh](combined_tanh.json) | 175.34 | 2.249 | 3.405 | 5 | 15 | 0.999200 | 287 |
| [fixed_text](fixed_text.json) | 175.48 | 1.350 | 2.444 | 5 | 15 | 0.999177 | 289 |
| [combined_fused_channels](combined_fused_channels.json) | 175.99 | 2.248 | 3.382 | 5 | 15 | 0.999189 | 291 |
| [combined_channels](combined_channels.json) | 176.07 | 2.247 | 3.372 | 5 | 15 | 0.999193 | 286 |
| [combined_fused](combined_fused.json) | 176.15 | 2.248 | 3.399 | 5 | 15 | 0.999193 | 287 |
| [final_compiled_uncached](final_compiled_uncached.json) | 176.90 | 2.126 | 3.345 | 5 | 15 | 0.999197 | 286 |
| [text_cache](text_cache.json) | 177.59 | 2.248 | 3.405 | 3 | 11 | 0.999215 | 225 |
| [resolution896](resolution896.json) | 187.44 | 2.176 | 3.325 | 3 | 11 | 0.982213 | 5803 |
| [early](early.json) | 194.17 | 2.248 | 3.405 | 3 | 11 | 0.999215 | 225 |
| [tanh_mlp](tanh_mlp.json) | 196.24 | 2.248 | 3.405 | 3 | 11 | 0.999204 | 232 |
| [channels_last](channels_last.json) | 196.30 | 2.247 | 3.372 | 3 | 11 | 0.999237 | 222 |
| [stock_before_round3](stock_before_round3.json) | 196.77 | 4.994 | 6.017 | 3 | 11 | — | — |
| [fused](fused.json) | 198.29 | 2.249 | 3.356 | 3 | 11 | 0.999199 | 230 |
| [inplace_mlp](inplace_mlp.json) | 198.38 | 2.248 | 3.399 | 3 | 11 | 0.999215 | 225 |
| [half](half.json) | 200.49 | 2.248 | 3.405 | 3 | 11 | 0.999215 | 225 |
| [nocache](nocache.json) | 201.88 | 3.812 | 4.944 | 3 | 11 | 1.000000 | 0 |
| [compiled_mlp](compiled_mlp.json) | 202.41 | 2.248 | 3.405 | 3 | 11 | 0.999223 | 220 |
| [static_head](static_head.json) | 202.95 | 2.248 | 3.415 | 3 | 11 | 0.999203 | 227 |
| [combined_uncached](combined_uncached.json) | 202.99 | 2.249 | 3.323 | 5 | 15 | 0.999177 | 289 |
| [arena](arena.json) | 203.83 | 2.248 | 3.464 | 3 | 11 | 0.999215 | 225 |
| [stock](stock.json) | 204.13 | 4.994 | 6.017 | 5 | 15 | — | — |
| [bf16_lean](bf16_lean.json) | 205.75 | 3.530 | 4.689 | 3 | 11 | 1.000000 | 0 |
| [output_inplace](output_inplace.json) | 205.85 | 2.249 | 3.323 | 5 | 15 | 0.999177 | 289 |
| [dynamic_head](dynamic_head.json) | 206.28 | 2.248 | 3.405 | 3 | 11 | 0.999225 | 224 |
| [pixel_inplace](pixel_inplace.json) | 206.99 | 2.248 | 3.325 | 3 | 11 | 0.999215 | 225 |
| [cudnn_search](cudnn_search.json) | 207.87 | 2.248 | 3.333 | 3 | 11 | 0.999213 | 223 |
| [fp16](fp16.json) | 210.77 | 3.811 | 4.952 | 5 | 15 | 0.999177 | 289 |
| [fp16_before_round3](fp16_before_round3.json) | 210.87 | 3.811 | 4.952 | 3 | 11 | 0.999215 | 225 |
| [fp16_lean](fp16_lean.json) | 213.31 | 3.530 | 4.696 | 3 | 11 | 0.999215 | 225 |
| [chunk512](chunk512.json) | 214.32 | 2.248 | 3.399 | 3 | 11 | 0.999212 | 223 |
| [efficient](efficient.json) | 217.43 | 2.248 | 3.405 | 3 | 11 | 0.999210 | 226 |
| [rope_real](rope_real.json) | 226.96 | 2.248 | 3.472 | 3 | 11 | 0.999238 | 220 |
