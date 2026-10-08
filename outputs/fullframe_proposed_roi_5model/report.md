# Full-frame vs Motion Multi-ROI

같은 generation setting으로 생성한 결과를 video별로 나란히 기록한다. 어느 condition이 더 좋은지는 판정하지 않는다.

## Setup

- Prompts:
  - Full-frame: `prompts/frames_full.txt`
  - Full-frame 2 fps: `prompts/frames_full.txt`
  - Motion ROI: `prompts/frames_roi_per_tube.txt`
- Sampling: full-frame uniform 1.0 fps; ROI sequence는 저장된 Stage3 tube의 1 fps crop.
- Motion ROI packing: each tube is `[Global]` once at the ROI start, then `[ROI#]` and that ROI video. No anchor.
- Models: 8b, 8b-int8, 8b-fp8, 30b, 30b-fp8

Generation:

```json
{
  "max_new_tokens": 256,
  "do_sample": false,
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "repetition_penalty": 1.2
}
```

## Metrics

| video_id | model | condition | frame/image count | visual tokens | total pixels | inference time (s) | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| VIRAT_S_010114_01_000235_000311 | 8b | Full-frame | 73 | 8880 | 67276800 | 819.300 | ok |
| VIRAT_S_010114_01_000235_000311 | 8b | Full-frame 2 fps | 144 | 17280 | 132710400 | 132.985 | ok |
| VIRAT_S_010114_01_000235_000311 | 8b | Motion ROI | 68 | 8956 | 28364800 | 817.362 | ok |
| VIRAT_S_010207_04_000929_000954 | 8b | Full-frame | 17 | 2160 | 15667200 | 120.680 | ok |
| VIRAT_S_010207_04_000929_000954 | 8b | Full-frame 2 fps | 32 | 3840 | 29491200 | 30.190 | ok |
| VIRAT_S_010207_04_000929_000954 | 8b | Motion ROI | 11 | 332 | 1085440 | 46.716 | ok |
| VIRAT_S_000200_00_000100_000171 | 8b | Full-frame | 72 | 8640 | 66355200 | 716.910 | ok |
| VIRAT_S_000200_00_000100_000171 | 8b | Full-frame 2 fps | 142 | 17040 | 130867200 | 197.258 | ok |
| VIRAT_S_000200_00_000100_000171 | 8b | Motion ROI | 73 | 9468 | 22228992 | 950.005 | ok |
| VIRAT_S_010001_03_000537_000563 | 8b | Full-frame | 22 | 2640 | 20275200 | 205.062 | ok |
| VIRAT_S_010001_03_000537_000563 | 8b | Full-frame 2 fps | 43 | 5280 | 39628800 | 55.547 | ok |
| VIRAT_S_010001_03_000537_000563 | 8b | Motion ROI | 52 | 3660 | 9117696 | 248.193 | ok |
| VIRAT_S_010003_03_000219_000259 | 8b | Full-frame | 41 | 5040 | 37785600 | 384.178 | ok |
| VIRAT_S_010003_03_000219_000259 | 8b | Full-frame 2 fps | 81 | 9840 | 74649600 | 34.487 | ok |
| VIRAT_S_010003_03_000219_000259 | 8b | Motion ROI | 110 | 5860 | 17246208 | 483.060 | ok |
| VIRAT_S_010003_10_000901_000934 | 8b | Full-frame | 28 | 3360 | 25804800 | 200.060 | ok |
| VIRAT_S_010003_10_000901_000934 | 8b | Full-frame 2 fps | 55 | 6720 | 50688000 | 80.570 | ok |
| VIRAT_S_010003_10_000901_000934 | 8b | Motion ROI | 58 | 4872 | 13828096 | 335.557 | ok |
| VIRAT_S_010003_06_000526_000560 | 8b | Full-frame | 30 | 3600 | 27648000 | 297.803 | ok |
| VIRAT_S_010003_06_000526_000560 | 8b | Full-frame 2 fps | 58 | 6960 | 53452800 | 73.441 | ok |
| VIRAT_S_010003_06_000526_000560 | 8b | Motion ROI | 88 | 4832 | 11069440 | 365.232 | ok |
| VIRAT_S_010106_02_000656_000692 | 8b | Full-frame | 36 | 4320 | 33177600 | 328.139 | ok |
| VIRAT_S_010106_02_000656_000692 | 8b | Full-frame 2 fps | 71 | 8640 | 65433600 | 23.055 | ok |
| VIRAT_S_010106_02_000656_000692 | 8b | Motion ROI | 73 | 4800 | 11411456 | 336.403 | ok |
| VIRAT_S_010113_05_000776_000805 | 8b | Full-frame | 24 | 2880 | 22118400 | 159.089 | ok |
| VIRAT_S_010113_05_000776_000805 | 8b | Full-frame 2 fps | 47 | 5760 | 43315200 | 22.606 | ok |
| VIRAT_S_010113_05_000776_000805 | 8b | Motion ROI | 57 | 3956 | 10761216 | 284.941 | ok |
| VIRAT_S_040000_01_000042_000099 | 8b | Full-frame | 58 | 6960 | 120268800 | 474.949 | ok |
| VIRAT_S_040000_01_000042_000099 | 8b | Full-frame 2 fps | 115 | 13920 | 238464000 | 33.945 | ok |
| VIRAT_S_040000_01_000042_000099 | 8b | Motion ROI | 87 | 5856 | 18206720 | 409.565 | ok |
| VIRAT_S_040005_08_001225_001276 | 8b | Full-frame | 52 | 6240 | 107827200 | 399.837 | ok |
| VIRAT_S_040005_08_001225_001276 | 8b | Full-frame 2 fps | 103 | 12480 | 213580800 | 25.238 | ok |
| VIRAT_S_040005_08_001225_001276 | 8b | Motion ROI | 99 | 7300 | 19412480 | 571.203 | ok |
| VIRAT_S_010114_01_000235_000311 | 8b-int8 | Full-frame | 73 | 8880 | 67276800 | 212.772 | ok |
| VIRAT_S_010114_01_000235_000311 | 8b-int8 | Full-frame 2 fps | 144 | 17280 | 132710400 | 29.367 | ok |
| VIRAT_S_010114_01_000235_000311 | 8b-int8 | Motion ROI | 68 | 8956 | 28364800 | 152.396 | ok |
| VIRAT_S_010207_04_000929_000954 | 8b-int8 | Full-frame | 17 | 2160 | 15667200 | 47.903 | ok |
| VIRAT_S_010207_04_000929_000954 | 8b-int8 | Full-frame 2 fps | 32 | 3840 | 29491200 | 7.622 | ok |
| VIRAT_S_010207_04_000929_000954 | 8b-int8 | Motion ROI | 11 | 332 | 1085440 | 33.755 | ok |
| VIRAT_S_000200_00_000100_000171 | 8b-int8 | Full-frame | 72 | 8640 | 66355200 | 141.085 | ok |
| VIRAT_S_000200_00_000100_000171 | 8b-int8 | Full-frame 2 fps | 142 | 17040 | 130867200 | 39.597 | ok |
| VIRAT_S_000200_00_000100_000171 | 8b-int8 | Motion ROI | 73 | 9468 | 22228992 | 270.802 | ok |
| VIRAT_S_010001_03_000537_000563 | 8b-int8 | Full-frame | 22 | 2640 | 20275200 | 81.599 | ok |
| VIRAT_S_010001_03_000537_000563 | 8b-int8 | Full-frame 2 fps | 43 | 5280 | 39628800 | 7.271 | ok |
| VIRAT_S_010001_03_000537_000563 | 8b-int8 | Motion ROI | 52 | 3660 | 9117696 | 83.798 | ok |
| VIRAT_S_010003_03_000219_000259 | 8b-int8 | Full-frame | 41 | 5040 | 37785600 | 51.112 | ok |
| VIRAT_S_010003_03_000219_000259 | 8b-int8 | Full-frame 2 fps | 81 | 9840 | 74649600 | 12.620 | ok |
| VIRAT_S_010003_03_000219_000259 | 8b-int8 | Motion ROI | 110 | 5860 | 17246208 | 112.369 | ok |
| VIRAT_S_010003_10_000901_000934 | 8b-int8 | Full-frame | 28 | 3360 | 25804800 | 36.030 | ok |
| VIRAT_S_010003_10_000901_000934 | 8b-int8 | Full-frame 2 fps | 55 | 6720 | 50688000 | 9.484 | ok |
| VIRAT_S_010003_10_000901_000934 | 8b-int8 | Motion ROI | 58 | 4872 | 13828096 | 136.894 | ok |
| VIRAT_S_010003_06_000526_000560 | 8b-int8 | Full-frame | 30 | 3600 | 27648000 | 78.250 | ok |
| VIRAT_S_010003_06_000526_000560 | 8b-int8 | Full-frame 2 fps | 58 | 6960 | 53452800 | 14.671 | ok |
| VIRAT_S_010003_06_000526_000560 | 8b-int8 | Motion ROI | 88 | 4832 | 11069440 | 140.395 | ok |
| VIRAT_S_010106_02_000656_000692 | 8b-int8 | Full-frame | 36 | 4320 | 33177600 | 41.771 | ok |
| VIRAT_S_010106_02_000656_000692 | 8b-int8 | Full-frame 2 fps | 71 | 8640 | 65433600 | 16.490 | ok |
| VIRAT_S_010106_02_000656_000692 | 8b-int8 | Motion ROI | 73 | 4800 | 11411456 | 100.959 | ok |
| VIRAT_S_010113_05_000776_000805 | 8b-int8 | Full-frame | 24 | 2880 | 22118400 | 39.227 | ok |
| VIRAT_S_010113_05_000776_000805 | 8b-int8 | Full-frame 2 fps | 47 | 5760 | 43315200 | 7.707 | ok |
| VIRAT_S_010113_05_000776_000805 | 8b-int8 | Motion ROI | 57 | 3956 | 10761216 | 123.152 | ok |
| VIRAT_S_040000_01_000042_000099 | 8b-int8 | Full-frame | 58 | 6960 | 120268800 | 38.786 | ok |
| VIRAT_S_040000_01_000042_000099 | 8b-int8 | Full-frame 2 fps | 115 | 13920 | 238464000 | 15.215 | ok |
| VIRAT_S_040000_01_000042_000099 | 8b-int8 | Motion ROI | 87 | 5856 | 18206720 | 53.223 | ok |
| VIRAT_S_040005_08_001225_001276 | 8b-int8 | Full-frame | 52 | 6240 | 107827200 | 31.092 | ok |
| VIRAT_S_040005_08_001225_001276 | 8b-int8 | Full-frame 2 fps | 103 | 12480 | 213580800 | 13.807 | ok |
| VIRAT_S_040005_08_001225_001276 | 8b-int8 | Motion ROI | 99 | 7300 | 19412480 | 76.167 | ok |
| VIRAT_S_010114_01_000235_000311 | 8b-fp8 | Full-frame | 73 | 8880 | 67276800 | 78.525 | ok |
| VIRAT_S_010114_01_000235_000311 | 8b-fp8 | Full-frame 2 fps | 144 | 17280 | 132710400 | 43.900 | ok |
| VIRAT_S_010114_01_000235_000311 | 8b-fp8 | Motion ROI | 68 | 8956 | 28364800 | 30.138 | ok |
| VIRAT_S_010207_04_000929_000954 | 8b-fp8 | Full-frame | 17 | 2160 | 15667200 | 7.768 | ok |
| VIRAT_S_010207_04_000929_000954 | 8b-fp8 | Full-frame 2 fps | 32 | 3840 | 29491200 | 4.589 | ok |
| VIRAT_S_010207_04_000929_000954 | 8b-fp8 | Motion ROI | 11 | 332 | 1085440 | 4.448 | ok |
| VIRAT_S_000200_00_000100_000171 | 8b-fp8 | Full-frame | 72 | 8640 | 66355200 | 46.981 | ok |
| VIRAT_S_000200_00_000100_000171 | 8b-fp8 | Full-frame 2 fps | 142 | 17040 | 130867200 | 24.775 | ok |
| VIRAT_S_000200_00_000100_000171 | 8b-fp8 | Motion ROI | 73 | 9468 | 22228992 | 27.113 | ok |
| VIRAT_S_010001_03_000537_000563 | 8b-fp8 | Full-frame | 22 | 2640 | 20275200 | 15.007 | ok |
| VIRAT_S_010001_03_000537_000563 | 8b-fp8 | Full-frame 2 fps | 43 | 5280 | 39628800 | 8.103 | ok |
| VIRAT_S_010001_03_000537_000563 | 8b-fp8 | Motion ROI | 52 | 3660 | 9117696 | 11.674 | ok |
| VIRAT_S_010003_03_000219_000259 | 8b-fp8 | Full-frame | 41 | 5040 | 37785600 | 19.850 | ok |
| VIRAT_S_010003_03_000219_000259 | 8b-fp8 | Full-frame 2 fps | 81 | 9840 | 74649600 | 9.167 | ok |
| VIRAT_S_010003_03_000219_000259 | 8b-fp8 | Motion ROI | 110 | 5860 | 17246208 | 19.622 | ok |
| VIRAT_S_010003_10_000901_000934 | 8b-fp8 | Full-frame | 28 | 3360 | 25804800 | 11.009 | ok |
| VIRAT_S_010003_10_000901_000934 | 8b-fp8 | Full-frame 2 fps | 55 | 6720 | 50688000 | 6.887 | ok |
| VIRAT_S_010003_10_000901_000934 | 8b-fp8 | Motion ROI | 58 | 4872 | 13828096 | 24.483 | ok |
| VIRAT_S_010003_06_000526_000560 | 8b-fp8 | Full-frame | 30 | 3600 | 27648000 | 18.080 | ok |
| VIRAT_S_010003_06_000526_000560 | 8b-fp8 | Full-frame 2 fps | 58 | 6960 | 53452800 | 11.928 | ok |
| VIRAT_S_010003_06_000526_000560 | 8b-fp8 | Motion ROI | 88 | 4832 | 11069440 | 24.179 | ok |
| VIRAT_S_010106_02_000656_000692 | 8b-fp8 | Full-frame | 36 | 4320 | 33177600 | 20.916 | ok |
| VIRAT_S_010106_02_000656_000692 | 8b-fp8 | Full-frame 2 fps | 71 | 8640 | 65433600 | 11.378 | ok |
| VIRAT_S_010106_02_000656_000692 | 8b-fp8 | Motion ROI | 73 | 4800 | 11411456 | 14.209 | ok |
| VIRAT_S_010113_05_000776_000805 | 8b-fp8 | Full-frame | 24 | 2880 | 22118400 | 14.971 | ok |
| VIRAT_S_010113_05_000776_000805 | 8b-fp8 | Full-frame 2 fps | 47 | 5760 | 43315200 | 6.234 | ok |
| VIRAT_S_010113_05_000776_000805 | 8b-fp8 | Motion ROI | 57 | 3956 | 10761216 | 22.567 | ok |
| VIRAT_S_040000_01_000042_000099 | 8b-fp8 | Full-frame | 58 | 6960 | 120268800 | 13.555 | ok |
| VIRAT_S_040000_01_000042_000099 | 8b-fp8 | Full-frame 2 fps | 115 | 13920 | 238464000 | 11.569 | ok |
| VIRAT_S_040000_01_000042_000099 | 8b-fp8 | Motion ROI | 87 | 5856 | 18206720 | 17.126 | ok |
| VIRAT_S_040005_08_001225_001276 | 8b-fp8 | Full-frame | 52 | 6240 | 107827200 | 12.071 | ok |
| VIRAT_S_040005_08_001225_001276 | 8b-fp8 | Full-frame 2 fps | 103 | 12480 | 213580800 | 10.624 | ok |
| VIRAT_S_040005_08_001225_001276 | 8b-fp8 | Motion ROI | 99 | 7300 | 19412480 | 17.965 | ok |
| VIRAT_S_010114_01_000235_000311 | 30b | Full-frame | 73 | 8880 | 67276800 | 245.295 | ok |
| VIRAT_S_010114_01_000235_000311 | 30b | Full-frame 2 fps | 144 | 17280 | 132710400 | 146.080 | ok |
| VIRAT_S_010114_01_000235_000311 | 30b | Motion ROI | 68 | 8956 | 28364800 | 368.659 | ok |
| VIRAT_S_010207_04_000929_000954 | 30b | Full-frame | 17 | 2160 | 15667200 | 204.883 | ok |
| VIRAT_S_010207_04_000929_000954 | 30b | Full-frame 2 fps | 32 | 3840 | 29491200 | 53.501 | ok |
| VIRAT_S_010207_04_000929_000954 | 30b | Motion ROI | 11 | 332 | 1085440 | 138.816 | ok |
| VIRAT_S_000200_00_000100_000171 | 30b | Full-frame | 72 | 8640 | 66355200 | 422.101 | ok |
| VIRAT_S_000200_00_000100_000171 | 30b | Full-frame 2 fps | 142 | 17040 | 130867200 | 646.784 | ok |
| VIRAT_S_000200_00_000100_000171 | 30b | Motion ROI | 73 | 9468 | 22228992 | 320.327 | ok |
| VIRAT_S_010001_03_000537_000563 | 30b | Full-frame | 22 | 2640 | 20275200 | 137.109 | ok |
| VIRAT_S_010001_03_000537_000563 | 30b | Full-frame 2 fps | 43 | 5280 | 39628800 | 76.517 | ok |
| VIRAT_S_010001_03_000537_000563 | 30b | Motion ROI | 52 | 3660 | 9117696 | 168.516 | ok |
| VIRAT_S_010003_03_000219_000259 | 30b | Full-frame | 41 | 5040 | 37785600 | 137.633 | ok |
| VIRAT_S_010003_03_000219_000259 | 30b | Full-frame 2 fps | 81 | 9840 | 74649600 | 104.438 | ok |
| VIRAT_S_010003_03_000219_000259 | 30b | Motion ROI | 110 | 5860 | 17246208 | 159.006 | ok |
| VIRAT_S_010003_10_000901_000934 | 30b | Full-frame | 28 | 3360 | 25804800 | 127.588 | ok |
| VIRAT_S_010003_10_000901_000934 | 30b | Full-frame 2 fps | 55 | 6720 | 50688000 | 77.712 | ok |
| VIRAT_S_010003_10_000901_000934 | 30b | Motion ROI | 58 | 4872 | 13828096 | 144.391 | ok |
| VIRAT_S_010003_06_000526_000560 | 30b | Full-frame | 30 | 3600 | 27648000 | 138.635 | ok |
| VIRAT_S_010003_06_000526_000560 | 30b | Full-frame 2 fps | 58 | 6960 | 53452800 | 95.170 | ok |
| VIRAT_S_010003_06_000526_000560 | 30b | Motion ROI | 88 | 4832 | 11069440 | 150.744 | ok |
| VIRAT_S_010106_02_000656_000692 | 30b | Full-frame | 36 | 4320 | 33177600 | 291.538 | ok |
| VIRAT_S_010106_02_000656_000692 | 30b | Full-frame 2 fps | 71 | 8640 | 65433600 | 84.269 | ok |
| VIRAT_S_010106_02_000656_000692 | 30b | Motion ROI | 73 | 4800 | 11411456 | 162.469 | ok |
| VIRAT_S_010113_05_000776_000805 | 30b | Full-frame | 24 | 2880 | 22118400 | 198.364 | ok |
| VIRAT_S_010113_05_000776_000805 | 30b | Full-frame 2 fps | 47 | 5760 | 43315200 | 57.715 | ok |
| VIRAT_S_010113_05_000776_000805 | 30b | Motion ROI | 57 | 3956 | 10761216 | 155.758 | ok |
| VIRAT_S_040000_01_000042_000099 | 30b | Full-frame | 58 | 6960 | 120268800 | 138.006 | ok |
| VIRAT_S_040000_01_000042_000099 | 30b | Full-frame 2 fps | 115 | 13920 | 238464000 | 84.054 | ok |
| VIRAT_S_040000_01_000042_000099 | 30b | Motion ROI | 87 | 5856 | 18206720 | 346.559 | ok |
| VIRAT_S_040005_08_001225_001276 | 30b | Full-frame | 52 | 6240 | 107827200 | 149.774 | ok |
| VIRAT_S_040005_08_001225_001276 | 30b | Full-frame 2 fps | 103 | 12480 | 213580800 | 94.015 | ok |
| VIRAT_S_040005_08_001225_001276 | 30b | Motion ROI | 99 | 7300 | 19412480 | 315.776 | ok |
| VIRAT_S_010114_01_000235_000311 | 30b-fp8 | Full-frame | 73 | 8880 | 67276800 | 123.424 | ok |
| VIRAT_S_010114_01_000235_000311 | 30b-fp8 | Full-frame 2 fps | 144 | 17280 | 132710400 | 27.713 | ok |
| VIRAT_S_010114_01_000235_000311 | 30b-fp8 | Motion ROI | 68 | 8956 | 28364800 | 196.396 | ok |
| VIRAT_S_010207_04_000929_000954 | 30b-fp8 | Full-frame | 17 | 2160 | 15667200 | 45.000 | ok |
| VIRAT_S_010207_04_000929_000954 | 30b-fp8 | Full-frame 2 fps | 32 | 3840 | 29491200 | 3.731 | ok |
| VIRAT_S_010207_04_000929_000954 | 30b-fp8 | Motion ROI | 11 | 332 | 1085440 | 41.529 | ok |
| VIRAT_S_000200_00_000100_000171 | 30b-fp8 | Full-frame | 72 | 8640 | 66355200 | 98.762 | ok |
| VIRAT_S_000200_00_000100_000171 | 30b-fp8 | Full-frame 2 fps | 142 | 17040 | 130867200 | 21.885 | ok |
| VIRAT_S_000200_00_000100_000171 | 30b-fp8 | Motion ROI | 73 | 9468 | 22228992 | 127.514 | ok |
| VIRAT_S_010001_03_000537_000563 | 30b-fp8 | Full-frame | 22 | 2640 | 20275200 | 40.963 | ok |
| VIRAT_S_010001_03_000537_000563 | 30b-fp8 | Full-frame 2 fps | 43 | 5280 | 39628800 | 4.734 | ok |
| VIRAT_S_010001_03_000537_000563 | 30b-fp8 | Motion ROI | 52 | 3660 | 9117696 | 45.985 | ok |
| VIRAT_S_010003_03_000219_000259 | 30b-fp8 | Full-frame | 41 | 5040 | 37785600 | 42.473 | ok |
| VIRAT_S_010003_03_000219_000259 | 30b-fp8 | Full-frame 2 fps | 81 | 9840 | 74649600 | 7.711 | ok |
| VIRAT_S_010003_03_000219_000259 | 30b-fp8 | Motion ROI | 110 | 5860 | 17246208 | 46.461 | ok |
| VIRAT_S_010003_10_000901_000934 | 30b-fp8 | Full-frame | 28 | 3360 | 25804800 | 37.568 | ok |
| VIRAT_S_010003_10_000901_000934 | 30b-fp8 | Full-frame 2 fps | 55 | 6720 | 50688000 | 6.321 | ok |
| VIRAT_S_010003_10_000901_000934 | 30b-fp8 | Motion ROI | 58 | 4872 | 13828096 | 42.696 | ok |
| VIRAT_S_010003_06_000526_000560 | 30b-fp8 | Full-frame | 30 | 3600 | 27648000 | 33.559 | ok |
| VIRAT_S_010003_06_000526_000560 | 30b-fp8 | Full-frame 2 fps | 58 | 6960 | 53452800 | 5.585 | ok |
| VIRAT_S_010003_06_000526_000560 | 30b-fp8 | Motion ROI | 88 | 4832 | 11069440 | 95.401 | ok |
| VIRAT_S_010106_02_000656_000692 | 30b-fp8 | Full-frame | 36 | 4320 | 33177600 | 37.364 | ok |
| VIRAT_S_010106_02_000656_000692 | 30b-fp8 | Full-frame 2 fps | 71 | 8640 | 65433600 | 6.594 | ok |
| VIRAT_S_010106_02_000656_000692 | 30b-fp8 | Motion ROI | 73 | 4800 | 11411456 | 50.258 | ok |
| VIRAT_S_010113_05_000776_000805 | 30b-fp8 | Full-frame | 24 | 2880 | 22118400 | 81.018 | ok |
| VIRAT_S_010113_05_000776_000805 | 30b-fp8 | Full-frame 2 fps | 47 | 5760 | 43315200 | 5.399 | ok |
| VIRAT_S_010113_05_000776_000805 | 30b-fp8 | Motion ROI | 57 | 3956 | 10761216 | 45.223 | ok |
| VIRAT_S_040000_01_000042_000099 | 30b-fp8 | Full-frame | 58 | 6960 | 120268800 | 29.928 | ok |
| VIRAT_S_040000_01_000042_000099 | 30b-fp8 | Full-frame 2 fps | 115 | 13920 | 238464000 | 10.829 | ok |
| VIRAT_S_040000_01_000042_000099 | 30b-fp8 | Motion ROI | 87 | 5856 | 18206720 | 121.916 | ok |
| VIRAT_S_040005_08_001225_001276 | 30b-fp8 | Full-frame | 52 | 6240 | 107827200 | 43.805 | ok |
| VIRAT_S_040005_08_001225_001276 | 30b-fp8 | Full-frame 2 fps | 103 | 12480 | 213580800 | 9.427 | ok |
| VIRAT_S_040005_08_001225_001276 | 30b-fp8 | Motion ROI | 99 | 7300 | 19412480 | 91.868 | ok |

## Descriptions

### VIRAT_S_010114_01_000235_000311

#### 8b

**Full-frame**

A white sedan drives into the parking lot from the left side, parks near other vehicles, then exits on the right after maneuvering around parked cars. A person walks along the sidewalk adjacent to the building at the far end of the lot while another individual approaches a vehicle nearby before walking away toward the center area. Two individuals walk together through the middle section of the parking lot heading towards the rear entrance of the buildings beyond the field. Another pedestrian enters the frame from the bottom-right corner carrying an object and proceeds straight ahead past several parked cars without stopping. The sun casts long shadows over parts of the asphalt surface throughout the entire duration of this footage.

**Full-frame 2 fps**

A red car drives through the parking lot from left to right while several pedestrians walk along the sidewalk near the buildings in the background. A white van exits the frame on the far right side after turning into an adjacent street. Two individuals cross the road at the intersection behind the parked vehicles before walking away toward the building entrance. Another person walks past the parked cars heading towards the exit gate located on the lower right corner of the screen. The sun casts long shadows over most parts of the asphalt surface throughout the entire duration of this footage.

**Motion ROI**

Two cyclists ride across the parking lot near the center, moving from right to left past parked cars before exiting the frame on the far side. A silver sedan drives through the lower-left section of the parking lot, entering from off-screen and driving toward the bottom edge of the view. Another car enters from the upper-right corner, navigating around several vehicles before stopping briefly near the white vans before continuing out of sight. Two pedestrians walk along the sidewalk adjacent to the building on the right side of the image, strolling slowly while passing behind some trees.

#### 8b-int8

**Full-frame**

A white car drives into the parking lot from the left side and parks near other vehicles. A person walks through the parking area carrying something on their shoulder before disappearing behind parked cars. Another individual exits a building nearby and walks toward the right edge of the frame while holding an object. Two people walk together along the sidewalk adjacent to the parking lot’s far end, moving away from the camera's view. A black SUV pulls up at the bottom-right corner of the screen and comes to rest after stopping briefly next to some trees. Throughout these moments, several pedestrians can be seen walking around various parts of the parking lot without interacting directly with any vehicle or another pedestrian. The sun casts long shadows over most areas due to early morning lighting conditions. No significant changes occur within this timeframe except for minor movements such as slight adjustments made by drivers seated inside stationary cars. There appears no noticeable activity beyond routine movement among individuals navigating between buildings and open spaces surrounding the parking structure. All observed activities remain confined strictly to non-interacting motions involving either vehicular positioning or human traversal paths established throughout the environment depicted in the footage.

**Full-frame 2 fps**

A red car drives through the parking lot from left to right while several pedestrians walk along the sidewalk near the buildings in the background. A white van exits the frame on the far right side after turning around at the edge of the parking area. Two individuals cross the street diagonally toward the upper-left corner before disappearing behind some trees. Another person walks past parked cars heading towards the center-right portion of the lot. The sun casts long shadows over most vehicles throughout this period, indicating early morning light conditions.

**Motion ROI**

Two cyclists ride across the parking lot near the center, moving from right to left past parked cars. A person walks along the sidewalk on the far right side of the frame, heading toward the white vans parked there. Another individual exits a building entrance located behind the central cluster of vehicles and begins walking towards the lower-left corner of the parking lot. Several cars drive through the intersection within the parking lot; first, a silver sedan enters from the bottom left before exiting out of view, followed shortly after by another car driving straight ahead from the upper middle section. Two people walk together down the path adjacent to the large beige building on the right-hand side of the image, approaching closer to the foreground while remaining parallel with the row of parked trucks.

#### 8b-fp8

**Full-frame**

A white car drives into view from the left side of the parking lot, parks near other vehicles, then exits the frame on the right after stopping briefly. A person walks along the sidewalk adjacent to the building at the far end of the parking lot, moving steadily toward the center before disappearing behind parked cars. Another individual appears walking towards the right edge of the frame near some trees, pauses momentarily beside a black SUV, and continues their path out of sight. Two people walk together through the middle section of the parking lot, heading generally toward the upper-right corner while maintaining close proximity until exiting the frame. A silver sedan enters from the bottom-left area, navigates around an empty space between two parked cars, stops for several moments, and proceeds forward slowly past more parked vehicles without changing direction significantly. The sun casts long shadows over parts of the asphalt surface throughout these movements, indicating consistent lighting conditions during this period.

**Full-frame 2 fps**

A red car drives through the parking lot from left to right near the center while several pedestrians walk along sidewalks surrounding the area. A white van exits on the far right side after turning into an adjacent road. Two individuals cross paths walking toward opposite directions at the upper middle section before continuing their respective routes. Another person walks past parked vehicles towards the bottom-right corner without stopping. The sun casts long shadows over most parts of the asphalt surface throughout the entire duration.

**Motion ROI**

Two cyclists ride across the parking lot near the center, moving leftward past parked vehicles before exiting the frame on the far side. A silver sedan drives through the lower-left section of the parking lot, turning right toward the edge of the frame after passing several cars. Another car enters from the upper-right corner, driving diagonally down towards the bottom-center of the lot before stopping among other parked vehicles. Two pedestrians walk along the sidewalk adjacent to the building on the right side of the image, heading away from the viewer’s perspective while remaining close together throughout their movement.

#### 30b

**Full-frame**

A red car drives through the parking lot from left to right while several other cars remain parked throughout the clip; people walk along the sidewalk at the far end of the lot near the buildings.

**Full-frame 2 fps**

A red car drives through the parking lot from left to right while several other cars remain parked throughout the clip; people walk near the buildings in the background on the far side of the lot.

**Motion ROI**

In the parking lot near a building with large windows and trees along the perimeter, several cars are parked while others move through the space; on the left side, vehicles enter and exit the frame, including a red car that drives across the lot, and further back, people walk along the sidewalk adjacent to the green field enclosed by a fence. On the right side, white vans remain stationary throughout, but two individuals appear walking together toward the bottom-right corner of the frame after briefly appearing near the center background. A person riding a bicycle moves horizontally across the middle section of the parking lot from left to right around mid-sequence.

#### 30b-fp8

**Full-frame**

A person walks from left to right near the center-left edge of the parking lot while several cars remain parked throughout the scene; no significant movement or activity occurs elsewhere in the frame during the duration of the clip.

**Full-frame 2 fps**

A person walks from left to right near the center-left area of the parking lot while several cars remain parked throughout the clip; at 0:42, two people briefly appear walking together on the far side of the parking lot before moving out of view.

**Motion ROI**

In the parking lot near a building with large windows and trees on the right side, several cars are parked while others move through the space; a red car drives across the middle section from left to right around 0:37, followed shortly after by another vehicle moving in the opposite direction. On the far left edge of the frame, two people walk together along the sidewalk adjacent to a sports field enclosed by a fence. A white van remains stationary on the lower-right portion of the lot throughout most of the clip. In front of the main building's entrance, individuals can be seen walking toward the door during various moments between 0:09 and 0:14. Additionally, multiple vehicles pass through the central part of the parking lot intermittently—some entering from the top-left corner (e.g., at 0:00), some exiting towards the bottom-right (like the black sedan leaving at approximately 0:02). The overall activity includes both pedestrian movement within designated pathways and vehicular traffic navigating available spaces under bright daylight conditions.

### VIRAT_S_010207_04_000929_000954

#### 8b

**Full-frame**

A person walks along a paved pathway toward a building labeled "STUDENT SERVICES," passing under an awning marked “CHICANO” while other individuals remain stationary near trash receptacles and informational posters on adjacent walls.

**Full-frame 2 fps**

A person walks along a paved pathway toward a building labeled "STUDENT SERVICES," passing under an awning marked “CHICANO” while other individuals remain stationary near trash receptacles and informational posters on adjacent walls.

**Motion ROI**

A person walks toward the entrance of a building with glass doors on the left side of the frame, passing under a green awning labeled "CHICAGO" before entering.

#### 8b-int8

**Full-frame**

A person walks along a paved pathway toward a building labeled "STUDENT SERVICES," passing under an awning marked “CHICANO” while other individuals remain stationary near trash receptacles and informational signs on the sidewalk.

**Full-frame 2 fps**

A person walks along a paved pathway toward a building labeled "STUDENT SERVICES," passing under an awning marked “CHICANO” while other individuals remain stationary near trash bins and informational signs on the sidewalk.

**Motion ROI**

A person walks toward the entrance of a building with glass doors on the left side of the frame, passing under a green awning labeled "CHICAGO" before entering.

#### 8b-fp8

**Full-frame**

A person walks along a paved pathway toward a building labeled "STUDENT SERVICES," passing under an awning marked “CHICANO” while other individuals remain stationary near trash receptacles and informational posters on adjacent walls.

**Full-frame 2 fps**

A person walks along a paved pathway toward a building labeled "STUDENT SERVICES," passing under an awning marked “CHICANO” while moving away from the camera's viewpoint.

**Motion ROI**

A person walks toward the entrance of a building with glass doors while carrying multiple bags, moving along a paved walkway adjacent to greenery on the left side of the frame.

#### 30b

**Full-frame**

A person walks along a pathway on the left side of the frame near the "Student Services" building while another individual briefly appears at the entrance under the green awning labeled "Chicano." A white van drives past in the background from right to left behind the buildings. The rest of the scene remains largely unchanged throughout the clip.

**Full-frame 2 fps**

A person walks along a covered walkway on the left side of the frame near the "STUDENT SERVICES" building before moving out of view.

**Motion ROI**

A person walks across the paved courtyard toward the entrance of the building marked "STUDENT SERVICES," passing under a covered walkway on the left side of the frame before entering through glass doors near the center-right portion of the image.

#### 30b-fp8

**Full-frame**

A person walks along a pathway on the left side of the frame near the "Student Services" building while another individual briefly appears at an entrance further back under the overhang before moving out of view; no other significant activity occurs throughout the clip.

**Full-frame 2 fps**

A person walks along a pathway on the left side of the frame near the "Student Services" building while another individual briefly appears at the entrance of the adjacent structure labeled "Chicano."

**Motion ROI**

A person walks along a covered walkway on the left side of the frame near glass doors, moving away from the camera toward the background while carrying items.

### VIRAT_S_000200_00_000100_000171

#### 8b

**Full-frame**

A silver sedan drives into view from the bottom right corner of the frame, stops at an empty parking space near the center-left area, and remains stationary for several moments before being joined briefly by another vehicle exiting the frame on the far right side while two individuals walk past the parked car toward the road's edge.

**Full-frame 2 fps**

A silver sedan drives into view from the bottom right corner of the frame, stops at an intersection near a grassy area marked with signs, and then proceeds forward along the road before exiting the frame on the left side. A white pickup truck follows behind the sedan, also stopping briefly at the intersection before continuing straight ahead down the road until it disappears out of sight. Shortly after, two individuals emerge from inside the parked silver sedan; one person exits first followed closely by another individual, both stepping onto the sidewalk adjacent to the parking lot entrance while carrying bags. They walk together toward the lower edge of the screen, moving away from their vehicle towards the street level outside the immediate vicinity of the parking structure. The second pedestrian appears slightly delayed compared to the initial exit but maintains pace alongside his companion throughout their journey off-screen. Throughout this entire duration, no other vehicles enter or leave the designated roadway within close proximity to these activities except for those already mentioned above. No additional movement beyond these described occurrences takes place during any part of the observed period shown here.

**Motion ROI**

A silver sedan drives up the entrance ramp of the parking lot, turns right onto a paved road adjacent to the lot, parks parallel to other spaces near a grassy median with signs indicating “PARKING” and “NO PARKING,” then stops briefly before pulling away again after someone exits the vehicle on foot. A person wearing light-colored clothing steps out of the parked car, walks around to the driver’s side door, opens it, retrieves something inside, closes the door, stands beside the open passenger-side window momentarily, then proceeds toward the sidewalk along the edge of the parking lot while carrying a bag slung across their shoulder. The individual continues walking down this path until they exit the frame entirely, leaving behind the stationary silver sedan still positioned within the marked parking space.

#### 8b-int8

**Full-frame**

A silver sedan drives into view from the bottom right corner of the frame, stops at an empty parking space near the center-left area, and remains stationary for several moments before driving off-screen toward the left side of the frame. A white pickup truck then enters from the bottom right, parks briefly next to another vehicle parked further down on the road, and subsequently departs towards the top-right direction. Throughout these movements, other vehicles remain either parked or moving along roads outside the main focus areas without interacting directly with any individuals present within the immediate vicinity. The overall environment consists primarily of large open asphalt lots bordered by trees and brick buildings under overcast skies.

**Full-frame 2 fps**

A silver sedan drives into view from the bottom right corner of the frame, stops at an empty parking space near the center-right side of the lot, and remains stationary for several moments before driving off-screen toward the left edge of the frame. A white pickup truck then enters from the bottom right, moving along the road adjacent to the parking lot until exiting on the far right side of the screen. Shortly after, two individuals appear walking together; they begin their journey beside the parked silver car located centrally within the lower half of the frame, proceed forward past this vehicle while maintaining close proximity, continue straight ahead crossing over towards the middle section of the asphalt area beneath them, eventually reaching the opposite end of the paved surface without stopping or changing direction during transit. The entire duration captures these movements against a backdrop featuring multiple brick buildings situated behind dense greenery surrounding both sides of the expansive open-air parking structure under cloudy skies overhead.

**Motion ROI**

A silver sedan drives up to the parking lot entrance, turns right onto the paved road, parks parallel to other cars near a grassy median with signs indicating “PARKING” and “NO PARKING,” then stops briefly before pulling away again after another vehicle passes through the intersection. A person exits the parked car, walks around to the driver’s side door, opens it, gets out carrying a bag, closes the door behind them, stands momentarily beside their vehicle looking toward the street, then begins walking along the sidewalk adjacent to the curb towards the left edge of frame while holding something small in hand that appears like keys or phone case. The individual continues moving steadily across this path until they disappear off-screen beyond view on the far-left corner of the image boundary line separating two sections of pavement marked white lines running horizontally down center dividing lanes within roadway surface itself which has yellow painted stripes marking edges alongside sidewalks flanked greenery patches interspersed between asphalt surfaces throughout entire expanse shown under cloudy sky conditions above distant buildings situated further back past treeline horizon stretching wide open field space extending outward indefinitely into background distance fading gradually due atmospheric perspective effect diminishing visibility clarity reducing contrast sharpness detail resolution quality overall impression created visually perceived environment experienced observer watching unfold real-time footage recorded high vantage point overlooking

#### 8b-fp8

**Full-frame**

A silver sedan drives into view from the bottom right corner of the frame, stops at an intersection near a grassy area marked with signs, and remains stationary for several moments before beginning to move again toward the left side of the screen. A white pickup truck follows behind the sedan along the road, passing through without stopping. Shortly after, another vehicle appears on the far-right edge of the frame moving away from the viewer's perspective down the street. In parallel, two individuals emerge from within the parked cars located closer to the foreground; one person exits first followed shortly thereafter by their companion, both walking together towards the center-left portion of the parking lot while maintaining close proximity throughout their movement path until exiting out-of-frame entirely. The surrounding environment features multiple brick buildings situated beyond dense greenery lining the perimeter of this expansive asphalt-covered space which contains numerous empty spaces designated for vehicles but currently occupied solely by those few mentioned above alongside some scattered signage indicating directions such as "No Parking" zones around specific areas including entrances/exits leading off onto adjacent roads flanked by sidewalks bordered further still by additional trees providing shade over parts of these structures under cloudy skies overhead suggesting either early morning light conditions or approaching dusk hours depending upon exact timing relative to day-night cycle progression observed during recording period indicated here via consistent

**Full-frame 2 fps**

A silver sedan drives into view from the bottom right corner of the frame, stops at an empty parking space near the center-right side of the lot, and remains stationary for several moments before being exited by two individuals who walk away toward the lower edge of the screen. A white pickup truck then enters from the far right, travels along the road adjacent to the parking area without stopping, and exits out of sight on the left side of the frame after passing behind some trees. Another vehicle briefly appears moving down the street towards the upper right portion of the image but quickly disappears beyond the foliage line. Throughout these occurrences, other cars remain parked throughout various sections within this large open-air asphalt expanse surrounded by brick buildings and greenery under overcast skies.

**Motion ROI**

A silver sedan drives onto the parking lot entrance ramp, turns left into a designated spot near a grassy median with signs, parks, and remains stationary while other vehicles pass on the adjacent road. A person exits the parked car, walks around to the driver’s side door, opens it, retrieves items including a backpack, closes the door, then proceeds walking along the sidewalk toward the right edge of the frame before exiting view.

#### 30b

**Full-frame**

A silver sedan drives into an empty parking lot from the right side of the frame at 0:09 and parks near the center-right edge; after remaining stationary for several seconds, two people exit the vehicle around 0:45 and walk toward the bottom-left corner of the screen before leaving the view. A black car remains parked on the left side of the lot throughout the clip. Another white car enters the frame from the top-right background at 0:18 but quickly exits out of sight. Several other vehicles pass through the road adjacent to the parking area along the lower portion of the image during various points between 0:00 and 0:20.

**Full-frame 2 fps**

A silver sedan enters from the bottom right corner of the frame at 0:00, drives along the road bordering the parking lot, makes a left turn into an entrance on the lower-left side around 0:06, and parks near the center-right edge of the lot between 0:10 and 0:12; after remaining parked for several seconds, two people exit the vehicle starting at approximately 0:22, walk towards the sidewalk area adjacent to the building on the far right, then continue walking toward the foreground until exiting the view by 1:04. A dark-colored car remains stationary in the upper-left section of the parking lot throughout the entire clip. Another light-colored car moves slowly through the background portion of the parking lot from right to left during the first few seconds (around 0:00–0:03). At least three other vehicles—a white pickup truck, another black SUV, and a red van—pass quickly through the roadway on the right side of the screen between 0:17 and 0:21 before leaving the frame. The overall environment consists of large multi-story buildings flanking both sides of the expansive paved parking lot under overcast skies.

**Motion ROI**

A silver sedan enters the frame from the bottom right, turns left onto the road adjacent to the parking lot, crosses the intersection, and parks diagonally across two spaces near the entrance. A dark-colored car passes through the intersection behind the silver sedan. After parking, a person exits the driver's side of the parked sedan, walks along the curb away from the vehicle toward the lower part of the frame, then stops briefly before continuing out of view. Another white pickup truck drives past on the main road after the first car has stopped.

#### 30b-fp8

**Full-frame**

A silver sedan drives into an empty parking lot from the right side of the frame at 0:09 and parks near the center-right edge; two people then exit the vehicle around 0:45 and walk toward the bottom-left corner of the screen before leaving the view. A black car remains parked on the left side throughout the clip while another white car moves along the road outside the parking area, passing through the lower portion of the frame multiple times between 0:00 and 0:20. Another dark-colored SUV appears briefly entering the parking lot from the top-center direction after 0:10 but quickly exits back out onto the main road without stopping. The rest of the large asphalt parking lot stays mostly vacant for the duration of the footage under overcast skies.

**Full-frame 2 fps**

A silver sedan enters from the bottom right corner of the frame at 0:00, drives along the road toward the left side of the parking lot entrance, turns into an available space near the center-right edge of the lot around 0:08, parks facing away from the street, and remains stationary for the rest of the clip; during this period, two people walk on the sidewalk adjacent to the parked car starting at approximately 0:46, moving towards the lower-left portion of the screen before exiting the view. A dark-colored SUV passes through the intersection at the top left of the image between 0:00 and 0:04, heading out of the frame. Another white pickup truck appears briefly at the bottom right at 0:18, driving past the main area. The remaining vehicles—a black sedan on the far left and another light-colored vehicle further back—remain mostly still throughout the entire duration.

**Motion ROI**

A silver sedan enters the frame from the right on a road adjacent to a large parking lot, makes a left turn onto the access road leading to the lot entrance, and parks diagonally across two spaces near the grassy median. A dark-colored car passes behind it on the main road before exiting the view. After the silver sedan stops, a person exits the driver's side door, walks along the curb away from the vehicle toward the bottom edge of the frame, then turns back towards the parked car while holding a white object. Another individual briefly appears standing next to the open passenger-side door before also walking down the sidewalk out of sight. The silver sedan remains stationary throughout these interactions.

### VIRAT_S_010001_03_000537_000563

#### 8b

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual sits at an outdoor table under a red umbrella near the center foreground. In the background, several people move around the building entrance area, including individuals ascending and descending the staircase and walking along pathways adjacent to the grassy slope. A few more people can be seen seated at tables further back under umbrellas, mostly remaining still throughout the clip. The overall setting remains calm with minimal movement apart from these occasional pedestrians navigating the plaza space.

**Full-frame 2 fps**

People walk up and down the stairs near the building entrance while others sit under red umbrellas at tables on the plaza. A few individuals move around the area, including someone walking past the left side of the frame and another person standing near the bottom right corner before moving out of view. The background features green grassy slopes bordered by trees and buildings with large windows, set against distant mountains under clear skies.

**Motion ROI**

In the foreground, a person sits under a red umbrella at a table on a paved plaza while others walk up and down nearby stairs leading to a building entrance. A second individual walks across a grassy lawn toward another set of tables shaded by umbrellas near a brick wall with large windows. The sky above remains clear blue throughout the scenes, indicating bright daylight conditions.

#### 8b-int8

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual sits at an outdoor table under a red umbrella near the center foreground. In the background, people move around various areas including walking up and down steps leading into buildings and sitting at tables further back under umbrellas. The setting remains consistent throughout the clip, showing a sunny day with clear skies over modern architecture surrounded by greenery and distant mountains.

**Full-frame 2 fps**

People walk up and down the stairs near the building while others sit under red umbrellas at tables on the plaza.

**Motion ROI**

In the foreground, a person sits under a red umbrella at a table, engrossed with their phone while others walk up and down the stairs behind them near a modern building entrance. Meanwhile, another individual walks across a grassy lawn toward a brick planter bed adjacent to a glass-walled building before exiting frame on the right side. The sky above remains clear blue throughout these activities.

#### 8b-fp8

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual sits at an outdoor table under a red umbrella near the center foreground. In the background, several people move around the building entrance area, including individuals ascending and descending the staircase leading into the structure. The surrounding environment remains largely unchanged throughout the clip, featuring multiple tables shaded by umbrellas arranged across a paved plaza adjacent to a grassy slope lined with trees and modern architecture against a backdrop of distant mountains under clear skies.

**Full-frame 2 fps**

People walk up and down the stairs leading into the building while others sit under red umbrellas at tables on the plaza below. A few individuals move around near the grassy area adjacent to the building’s entrance, but most remain seated or stationary throughout the duration of the footage. The background remains consistent with no significant changes in lighting or activity beyond these minor movements within the outdoor space.

**Motion ROI**

People walk down the stairs on the left side while another person sits under an umbrella near the center foreground looking at their phone. A woman walks across the grassy area toward the building entrance on the right side. The sky remains clear above the distant mountains throughout the scene.

#### 30b

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual remains seated at a table under an umbrella in the foreground; people can be seen walking near the building entrance in the background throughout the clip.

**Full-frame 2 fps**

People walk down the stairs on the left side of the frame while others sit at tables under red umbrellas in an outdoor seating area; some individuals move around near the building entrance in the background.

**Motion ROI**

On the outdoor patio with red umbrellas, two people walk down the stairs on the left side of the frame while another person sits alone under an umbrella, looking at their phone; later, someone exits through glass doors near the building entrance on the right side of the courtyard.

#### 30b-fp8

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual remains seated at an outdoor table under a red umbrella in the foreground; people can be seen walking near the building entrance in the background throughout the clip.

**Full-frame 2 fps**

People walk down the stairs on the left side of the frame while others sit at tables under red umbrellas in an outdoor seating area; some individuals move around near the building entrance in the background.

**Motion ROI**

Two people walk down a set of stairs on the left side of the courtyard while another person sits under a red umbrella near the center, looking at their phone; later, two individuals exit through glass doors on the right building entrance and proceed across the patio area.

### VIRAT_S_010003_03_000219_000259

#### 8b

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual sits at an outdoor table under a red umbrella near the bottom-left corner. In the background, people move around various areas including walking up steps and passing through doorways within the building complex situated behind a grassy hill adorned with trees and more tables shaded by umbrellas. A few individuals can be seen seated or standing further back among the patio furniture scattered throughout the open-air courtyard area surrounded by modern architecture against a backdrop of distant mountains under a partly cloudy sky.

**Full-frame 2 fps**

A person walks down stairs on the left side while another individual ascends them nearby; simultaneously, people move around tables under red umbrellas in an outdoor courtyard area near buildings surrounded by greenery and mountains in the background.

**Motion ROI**

People walk up and down stairs on the left side of the courtyard while others stroll across the grassy lawn near outdoor tables with red umbrellas under bright sunlight. Two individuals sit together beneath a large umbrella at a table, engaged in conversation, occasionally leaning forward during their interaction. A person walks past them carrying something white before disappearing out of view. Another individual approaches the seated pair briefly before walking away toward the right edge of the frame. The overall atmosphere remains calm throughout these activities within this open-air campus setting surrounded by buildings and distant mountains.

#### 8b-int8

**Full-frame**

A person walks down stairs near a building while another individual sits at an outdoor table under a red umbrella; people move around on pathways adjacent to grassy areas surrounded by buildings and trees against a backdrop of mountains.

**Full-frame 2 fps**

People walk up and down stairs near buildings while others sit under red umbrellas on tables in an outdoor courtyard surrounded by green grass and trees against a backdrop of mountains.

**Motion ROI**

People walk up and down stairs on the left side of the courtyard while others stroll across the grassy lawn near outdoor tables with red umbrellas under sunny skies. A person walks past seated individuals beneath a large umbrella on the lower patio level before disappearing out of frame. Two people sit together at a table shaded by an orange umbrella; they lean forward slightly during their conversation then return to sitting upright after briefly adjusting something on the ground beside them. The view remains fixed throughout these activities capturing movement within this open-air campus setting surrounded by buildings and distant mountains.

#### 8b-fp8

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual sits at an outdoor table under a red umbrella near the bottom-left corner. In the background, people move around various areas including walking up and down steps leading into buildings and strolling along pathways adjacent to the grassy lawn. A few individuals remain seated throughout the duration, mostly stationary beneath their respective umbrellas scattered across the courtyard space. The overall setting remains consistent with no significant changes in activity or environment observed during this period.

**Full-frame 2 fps**

People walk up and down stairs near tables under red umbrellas while others sit at those tables on a sunny day surrounded by green grass and buildings with mountains in the background.

**Motion ROI**

People walk up and down stairs on the left side of the courtyard while others stroll across the grassy lawn toward the building entrance under red umbrellas. Two individuals sit together beneath a large umbrella near the foreground tables, engaged in conversation with occasional gestures. A person walks past them carrying something white before disappearing behind another table nearby. The camera remains fixed throughout these activities capturing movement within this outdoor campus setting surrounded by buildings and distant mountains.

#### 30b

**Full-frame**

A person walks down the stairs on the left side of the courtyard while another individual ascends them; later, two people walk together from right to left near the building entrance at the back of the grassy area.

**Full-frame 2 fps**

People walk up and down stairs on the left side of the courtyard while others sit under red umbrellas at tables; individuals move around the grassy area and near buildings in the background throughout the clip.

**Motion ROI**

In the courtyard, people walk across the grassy lawn near the building entrance while others sit under red umbrellas on the paved seating area; two individuals are seen sitting together at a table with an umbrella providing shade, engaging in conversation throughout the clip.

#### 30b-fp8

**Full-frame**

A person walks down the stairs on the left side of the courtyard while another individual stands at the top of the staircase; two people sit under an umbrella near the bottom left corner of the frame throughout the clip.

**Full-frame 2 fps**

People walk up and down stairs on the left side of the courtyard while others sit under umbrellas at tables; some individuals move around near the building entrance on the right side of the frame throughout the clip.

**Motion ROI**

In the courtyard, people walk across the grassy lawn near the building entrance while others sit under red umbrellas on the paved seating area; two individuals are seen sitting together at a table with an umbrella, engaged in conversation throughout the clip.

### VIRAT_S_010003_10_000901_000934

#### 8b

**Full-frame**

A person walks down stairs toward an entrance while another individual ascends them; simultaneously, someone else moves from left to right near tables under red umbrellas on a paved area adjacent to a grassy lawn bordered by buildings and palm trees against a backdrop of mountains.

**Full-frame 2 fps**

A person walks from left to right across the foreground patio area beneath red umbrellas while others ascend stairs leading into a building behind them. Another individual descends those stairs after entering through an open doorway near the top landing. A few people remain seated at tables on the upper level under more umbrellas, occasionally shifting positions slightly but mostly staying still throughout the duration of the footage. The background features green grassy areas bordered by low stone walls, palm trees scattered around, and distant mountains partially obscured by clouds against a bright blue sky above modern architectural structures made primarily of brick and glass panels.

**Motion ROI**

People walk across the paved courtyard under red umbrellas while others ascend the stairs leading up to the building’s entrance. A person walks along the grassy pathway near the outdoor seating area adjacent to the brick facade with large windows. Another individual approaches the glass doors on the right side of the frame before walking away toward the left edge of the view.

#### 8b-int8

**Full-frame**

People walk through an outdoor plaza featuring numerous tables shaded by red umbrellas, while others ascend stairs leading into a building under a clear sky with mountains in the background.

**Full-frame 2 fps**

People walk through an outdoor plaza featuring numerous red umbrellas over tables, ascending stairs toward a building entrance while others stroll nearby under clear skies with mountains in the background.

**Motion ROI**

People walk across the paved courtyard under red umbrellas near stairs leading up to a building with large windows, while others ascend the staircase toward the entrance. A person walks past tables on the left side of the frame before moving out of view. Another individual descends the steps after reaching the top, joining those already walking down. In another section, people stroll along a grassy path adjacent to outdoor seating areas beneath closed umbrellas, passing between planters and heading towards the building’s glass doors. One person exits through the doorway, followed shortly afterward by two more individuals entering the space. The movement continues as additional figures pass behind them, eventually exiting the frame entirely.

#### 8b-fp8

**Full-frame**

People walk up and down stairs leading into a building while others stroll through an outdoor area furnished with tables under red umbrellas. A few individuals sit at tables near a grassy lawn bordered by trees and buildings, with mountains visible in the background under a partly cloudy sky.

**Full-frame 2 fps**

People walk through an outdoor plaza featuring numerous tables shaded by red umbrellas, ascending stairs toward a building entrance while others stroll along pathways near landscaped greenery under a bright sky.

**Motion ROI**

People walk across the courtyard under red umbrellas while others ascend the stairs leading to the building entrance. A person walks along the paved path near the grassy lawn adjacent to the outdoor seating area with tables and closed umbrellas. Two individuals are seen walking toward the building’s glass doors on the right side of the frame. Another individual exits through the doorway after entering earlier. Someone approaches the blue trash can located beside the main staircase before moving away. An individual wearing green clothing passes behind the planter boxes situated between the patio and the grassy section. People continue their movement around the plaza, some heading towards the upper level via the staircases while others stroll leisurely past the dining areas. The overall atmosphere remains calm throughout these activities within this campus setting.

#### 30b

**Full-frame**

People walk down the stairs on the left side of the courtyard while others move around the outdoor seating area under red umbrellas; some individuals enter and exit through doors at the building entrance in the background.

**Full-frame 2 fps**

People walk up and down stairs on the left side of the courtyard while others pass through the foreground area under red umbrellas; some individuals move toward the building entrance at the top center of the frame.

**Motion ROI**

People walk up and down a set of stairs near outdoor seating areas with red umbrellas on a sunny day; later, individuals are seen walking across a grassy lawn adjacent to another building section also featuring tables under red umbrellas.

#### 30b-fp8

**Full-frame**

People walk down the stairs on the left side of the courtyard while others move around the area under red umbrellas; some individuals enter and exit through doors at the building entrance in the background.

**Full-frame 2 fps**

A person walks from left to right through an outdoor seating area with red umbrellas while others ascend stairs leading into a building on the left side of the frame; further back, individuals walk near tables under additional red umbrellas on a grassy lawn bordered by trees and mountains in the distance.

**Motion ROI**

People walk up and down the stairs near the outdoor seating area with red umbrellas on a sunny day; later, individuals are seen walking across the grassy lawn adjacent to another section of tables under similar umbrellas.

### VIRAT_S_010003_06_000526_000560

#### 8b

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual sits at an outdoor table under a red umbrella near the bottom-left corner. A few people move around in the background near the building entrance and along pathways, including someone walking from right to left past tables toward the center-right area. Another person enters the frame from the lower-left edge and moves towards the central seating area before exiting out of view. The overall setting remains consistent throughout, featuring multiple red umbrellas over black tables arranged across a paved courtyard adjacent to a grassy slope leading up to modern buildings with large windows, surrounded by trees and mountains in the distance under clear skies.

**Full-frame 2 fps**

A person walks down stairs on the left side of the frame while another individual sits at an outdoor table under a red umbrella near the bottom-left corner. A few people walk through various parts of the courtyard area throughout the clip, including someone walking from right to left along the upper level behind the grassy hill and others moving around tables scattered across the paved space beneath numerous open umbrellas. The background remains consistent with buildings flanking both sides of the central lawn, trees providing shade, and mountains rising prominently against a clear sky above.

**Motion ROI**

In the foreground courtyard with red umbrellas, two individuals are seated at a table engaged in conversation while people walk past them on the paved ground near the stairs leading up to the building entrance. Meanwhile, another person walks across the patio adjacent to the glass doors under more umbrellas before exiting through the doorway. A third individual briefly appears walking along the edge of the grassy lawn beside large planters that line the pathway toward the main building’s entrance.

#### 8b-int8

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual sits at an outdoor table under a red umbrella near the bottom-left corner. A few people walk through various parts of the courtyard area throughout the clip, including someone walking from right to left along the upper level behind the grassy hill. The overall setting remains consistent with no major changes in activity or focus within the scene.

**Full-frame 2 fps**

A person walks down the stairs on the left side of the frame while another individual moves from right to left near the building entrance at the far end of the courtyard. A few people sit under umbrellas scattered throughout the paved area, remaining mostly stationary during the duration of the clip. The background features buildings surrounded by greenery and mountains under clear skies, maintaining consistent lighting conditions without any noticeable changes in weather or environment.

**Motion ROI**

In the foreground courtyard with red umbrellas, two individuals are seated at a table engaged in conversation while pedestrians walk past them on the paved ground near stairs leading up to a building entrance. Meanwhile, another person walks across the patio toward the right side of the frame before exiting view, passing under several open umbrellas that cast shadows onto the pavement below. Additionally, someone else briefly appears walking along the edge of the grassy lawn adjacent to the outdoor seating area, moving towards the left side of the screen. The background features architectural elements including brick walls, large windows, staircases flanked by railings, potted plants within raised planters, and distant mountains under clear skies providing ample sunlight throughout these scenes.

#### 8b-fp8

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual sits at an outdoor table under a red umbrella near the bottom-left corner. A few people move around the courtyard area, including someone walking from right to left along the upper level behind the grassy hill. Another person enters the frame from the lower-left edge and moves toward the center before exiting out of view. The overall setting remains consistent throughout, featuring multiple tables with umbrellas arranged across a paved patio adjacent to a green lawn bordered by buildings and trees against a backdrop of mountains.

**Full-frame 2 fps**

A person walks down stairs on the left side of the frame while another individual sits at an outdoor table under a red umbrella near the bottom-left corner. A few people walk through various parts of the courtyard area throughout the clip, including someone walking from right to left along the upper level behind the grassy hill and others moving around tables scattered across the paved space beneath numerous open umbrellas. The background remains consistent with buildings featuring large windows and staircases leading up to entrances, surrounded by greenery such as palm trees and other plants against a backdrop of distant mountains under clear skies.

**Motion ROI**

In the foreground courtyard with red umbrellas, two individuals are seated at a table engaged in conversation while people walk past them on the paved ground near the stairs leading up to the building entrance. Meanwhile, another person walks across the patio toward the right side of the frame, passing under several open umbrellas before exiting view. A third individual appears briefly walking down the steps towards the lower level, moving away from the main seating area. In the background, someone enters through glass doors located beneath large windows along the brick facade of the building, then exits shortly after. Finally, a fourth figure can be seen descending the staircase adjacent to the outdoor dining space, heading downward out of sight behind some planters positioned between tables.

#### 30b

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual briefly appears at the bottom right corner; people remain seated under umbrellas throughout the outdoor courtyard area surrounded by buildings and mountains in the background.

**Full-frame 2 fps**

A person walks down the stairs on the left side of the courtyard while another individual briefly appears at the bottom right corner; meanwhile, people remain seated under umbrellas throughout the outdoor seating area surrounded by buildings and mountains in the background.

**Motion ROI**

In the courtyard with red umbrellas and tables, two people sit under an umbrella on the left while another person walks down the stairs behind them; later, more individuals walk through various parts of the outdoor seating area near the building entrance.

#### 30b-fp8

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual briefly appears at the bottom edge; people remain seated under umbrellas throughout the courtyard area.

**Full-frame 2 fps**

A person walks down the stairs on the left side of the courtyard while another individual briefly appears at the bottom right corner; people remain seated under umbrellas throughout the scene.

**Motion ROI**

In the courtyard area with red umbrellas and tables, two people are seated under an umbrella on the left side while another person walks down the stairs behind them; later, this individual exits the frame and others walk through the lower part of the view. On the right side near the building entrance, several individuals enter and exit the doorway throughout the clip. In the foreground, a group remains seated beneath their respective umbrellas, occasionally shifting positions slightly but largely remaining stationary. The background features green lawns bordered by steps leading up to buildings, with mountains visible beyond.

### VIRAT_S_010106_02_000656_000692

#### 8b

**Full-frame**

Two individuals walk together through an empty parking lot toward the right side before separating; one continues walking while the other stops near parked vehicles. In the background, several people can be seen moving along pathways adjacent to buildings on the far end of the lot. A few cars drive past at intervals within the frame’s perimeter without stopping inside the main area shown. The overall setting remains consistent throughout, featuring multiple parked automobiles under bright daylight conditions against urban architecture including multi-story structures flanked by trees lining streets beyond the immediate vicinity of the parking space.

**Full-frame 2 fps**

A person walks from right to left near parked cars in an outdoor parking lot while other individuals move around in the background near buildings and trees.

**Motion ROI**

A person walks across the parking lot near several parked vans while another individual rides a bicycle along the sidewalk adjacent to a fenced sports field. Simultaneously, two cyclists ride past a street corner with traffic cones nearby, moving away from the intersection. In parallel, a cyclist continues riding on the path beside the fence enclosing the athletic field.

#### 8b-int8

**Full-frame**

Two individuals walk together through an empty parking lot toward the right side before separating; one continues walking while the other stops near parked vehicles on the far right edge.

**Full-frame 2 fps**

A person walks slowly from right to left near parked cars on the far side of the parking lot while another individual briefly appears walking toward them before disappearing out of frame. In the background, people can be seen moving along pathways adjacent to buildings outside the main focus area. A few vehicles remain stationary throughout the entire duration within their respective spaces without any noticeable movement or interaction among other objects or individuals present in this wide-angle view.

**Motion ROI**

In the parking lot, a person walks across the open space near several parked vehicles including white vans and a box truck before stopping to look around. Meanwhile, two individuals ride bicycles along a path adjacent to the field on the far side of the parking lot. Another cyclist rides past them further down that same path. Additionally, another individual can be seen riding a bicycle closer to the foreground edge of the frame, moving away from the viewer's perspective.

#### 8b-fp8

**Full-frame**

Two individuals walk together through an empty parking lot toward the right side before separating; one continues walking while the other stops near parked vehicles. A person walks alone from left to right along the far edge of the parking lot adjacent to buildings. Another individual exits a vehicle on the right side of the frame and begins walking away from the car towards the center-right area of the lot. In the background, several people move around near the entrance of a building located behind the parking lot. Throughout the entire duration, multiple cars remain stationary within designated spaces throughout the expansive asphalt-covered parking lot surrounded by trees and multi-story structures under clear daylight conditions.

**Full-frame 2 fps**

A person walks from right to left near parked cars on the far side of the parking lot while another individual moves toward them briefly before walking away again. In the background, people can be seen moving along pathways adjacent to buildings at various points throughout the footage without engaging directly with others nearby. A few vehicles remain stationary within designated spaces under clear daylight conditions during this observation period.

**Motion ROI**

A person walks across the parking lot near several parked vans while another individual rides a bicycle along the sidewalk adjacent to a fenced sports field. In the background, two cyclists ride past on a street next to a building with large windows, followed by someone riding a bike down a path beside trees.

#### 30b

**Full-frame**

Two people walk from right to left through the parking lot near the center-right area; another person walks into view on the far right side of the frame and stands still for several moments before walking out of sight. A blue car remains parked in the middle section of the lot throughout the clip while other vehicles remain stationary around the perimeter. The shadows cast by buildings and trees shift slightly over time indicating movement of the sun but no significant changes occur within the environment beyond these minor movements.

**Full-frame 2 fps**

A person walks through the parking lot from right to left near the center of the frame while several cars remain parked throughout the scene; no other significant movements occur.

**Motion ROI**

In the parking lot near a building with trees, a person walks across the paved area while another individual rides a bicycle along a path adjacent to a fenced sports field; further down the road, two people are seen walking together on a sidewalk beside parked cars.

#### 30b-fp8

**Full-frame**

A blue car drives through the parking lot from right to left while two people walk on the far side near white vans; other cars remain parked throughout the scene under bright sunlight casting long shadows.

**Full-frame 2 fps**

A person walks from right to left through the parking lot near the center-right area while several cars remain parked throughout the scene; no other significant movement or activity occurs.

**Motion ROI**

In the parking lot near a building with trees, a person walks across the paved area while another individual stands still holding their head; simultaneously, two people ride bicycles along a path adjacent to a fenced sports field, followed later by additional cyclists passing on the road nearby.

### VIRAT_S_010113_05_000776_000805

#### 8b

**Full-frame**

A person rides a bicycle from right to left along the edge of the parking lot near some parked vehicles while another individual walks briskly through the lower-right section of the frame before exiting out of view.

**Full-frame 2 fps**

A person walks from right to left across the lower-right portion of the parking lot while several cars remain stationary throughout the entire duration of the footage.

**Motion ROI**

In the parking lot near the fence line, two individuals walk across the asphalt toward the building entrance while several parked cars remain stationary nearby. Simultaneously, another person rides a bicycle through the open space between vehicles on the opposite side of the lot before stopping beside a black SUV partially obscured by trees. In the background, people are seen walking along the sidewalk adjacent to the multi-story office buildings under bright sunlight casting long shadows across the pavement.

#### 8b-int8

**Full-frame**

A person rides a bicycle from right to left along the edge of the parking lot near some parked vehicles while another individual walks toward them before continuing their path away from the camera's view.

**Full-frame 2 fps**

A person walks from right to left across the lower portion of the parking lot while several cars remain parked throughout the scene.

**Motion ROI**

In the parking lot near the fence line, two individuals walk side-by-side along the sidewalk before separating and walking away in opposite directions. Meanwhile, another person rides a bicycle across the open space toward the left edge of the frame while passing between parked vehicles. In the background, several people are seen standing together on the far right side of the lot near some trees, appearing to converse casually. A white van drives slowly through the central part of the parking lot past other stationary cars. On the upper level, three pedestrians stroll leisurely down the paved path adjacent to the building’s exterior wall under bright sunlight casting long shadows across the asphalt surface.

#### 8b-fp8

**Full-frame**

A person rides a bicycle from right to left along the edge of the parking lot near the building on the far side. Another individual walks briskly through the lower-right section of the frame, moving toward the center before exiting out of view. A third pedestrian enters from behind some trees at the bottom right corner, walking diagonally upward into the middle area of the parking lot. In the background, several people can be seen standing around near the entrance of the buildings beyond the parking lot.

**Full-frame 2 fps**

A person walks from right to left near parked cars on the far side of the lot while another individual rides a bicycle along the edge of the parking area toward the center before stopping briefly.

**Motion ROI**

In the parking lot near the fence line, two individuals walk side-by-side along the sidewalk before separating and walking away in opposite directions. In another section of the lot, three people ride bicycles across the open space toward the right edge of the frame while passing parked vehicles. Near the building’s facade, several pedestrians are seen moving around the perimeter, with some entering and exiting through doors located on the ground floor. A person walks past a white van parked beside a large truck, heading towards the left side of the lot. Another individual rides a bicycle down the middle aisle between rows of cars, eventually turning off-screen after reaching the bottom corner of the view. Throughout these activities, various vehicles remain stationary throughout the duration observed within this environment.

#### 30b

**Full-frame**

A person rides a bicycle from right to left across the lower portion of the parking lot while another individual walks through the shadowed area near the center-right; several cars remain parked throughout the scene, including a white truck on the far right which remains stationary until briefly moving forward at 0:07 before stopping again.

**Full-frame 2 fps**

A person walks through the parking lot from right to left near the bottom edge of the frame while another individual rides a bicycle along the far side of the parking area close to the buildings on the upper part of the image; several cars remain parked throughout the clip.

**Motion ROI**

In the parking lot near a building with greenery on the left side, two people walk along the sidewalk adjacent to parked cars; later, another person rides a bicycle across the lower part of the frame while several vehicles remain stationary throughout the scenes.

#### 30b-fp8

**Full-frame**

A person rides a bicycle from right to left across the parking lot near the bottom edge of the frame, while another individual walks through the shadowed area on the lower right side of the lot. Several cars remain parked throughout the clip, including a silver sedan positioned centrally in an empty space, and others scattered around the perimeter. A white truck remains stationary at the far end of the lot near the buildings. The shadows cast by surrounding trees shift slightly over time due to changing sunlight conditions.

**Full-frame 2 fps**

A person walks through the parking lot from right to left near the bottom edge of the frame while another individual rides a bicycle along the far side of the lot, moving from right to left before exiting the view; several cars remain parked throughout the clip.

**Motion ROI**

In the parking lot near a building with greenery on the left side, two people walk along the sidewalk adjacent to parked cars; later, another person rides a bicycle across the lower part of the frame while several vehicles remain stationary throughout the scenes.

### VIRAT_S_040000_01_000042_000099

#### 8b

**Full-frame**

A red car drives through an intersection near a parking lot while other vehicles remain stationary nearby under clear skies.

**Full-frame 2 fps**

A red car drives through an intersection near a parking lot while other vehicles remain stationary nearby under clear skies.

**Motion ROI**

A red convertible drives through a parking lot intersection near parked vehicles before exiting the frame. A person walks along a sidewalk adjacent to a low wall with vegetation nearby. The top portion of tree branches moves slightly against a clear sky background. Another view shows similar movement of tree branches against the sky.

#### 8b-int8

**Full-frame**

A red car drives through an intersection near a parking lot surrounded by green grass, trees, and buildings under clear skies.

**Full-frame 2 fps**

A red car drives through an intersection near a parking lot while other vehicles remain stationary nearby under clear skies.

**Motion ROI**

A red convertible car enters the parking lot, drives past parked vehicles, makes a turn around a central island, and exits toward another section of the lot while other cars remain stationary nearby.

#### 8b-fp8

**Full-frame**

A red car drives through an intersection near a parking lot surrounded by green grass, trees, and buildings under clear skies.

**Full-frame 2 fps**

A red car drives through an intersection near a parking lot while other vehicles remain stationary nearby under clear skies.

**Motion ROI**

A red convertible drives through a parking lot intersection near parked vehicles before exiting the frame. A person walks along a sidewalk adjacent to a low wall with vegetation nearby. The top portion of tree branches moves slightly against a clear sky background. Another section of tree foliage sways gently under similar conditions above another part of the landscape.

#### 30b

**Full-frame**

A red car drives through an intersection from left to right on a sunny day at a campus-like setting with buildings, trees, grassy areas, and parked cars; no other significant movements occur throughout the clip.

**Full-frame 2 fps**

A red car drives from left to right through an intersection at the center-left of the frame while other cars remain parked on both sides of the road; no significant changes occur elsewhere in the scene throughout the clip.

**Motion ROI**

In the parking lot on the left side of the frame, a red sedan enters the view from the bottom-left corner, drives across the paved area towards the right, makes a turn near a central landscaped island, continues driving past parked vehicles including a black SUV and green sedans, then exits the frame toward the top-right. In another part of the scene, a person walks along a sidewalk in the background behind some trees, moving from left to right before exiting the field of view. The sky remains clear throughout with tree tops occasionally appearing against it.

#### 30b-fp8

**Full-frame**

A red car drives from left to right through an intersection near a parking lot on a sunny day; no other vehicles move during the clip.

**Full-frame 2 fps**

A red car drives from left to right through an intersection at the top-left portion of the frame while other cars remain parked on both sides of the road; no significant changes occur throughout the rest of the scene.

**Motion ROI**

In the parking lot on the left side of the frame, a red sedan enters the view from behind other parked cars, drives around a circular island, makes a turn, and exits the frame back toward the top-left corner; this action repeats multiple times throughout the clip with slight variations in timing but consistent movement pattern. In another part of the scene near the bottom right edge, a person briefly appears walking along a path before moving out of sight. The upper portion of the image shows tree tops against the sky, which remain mostly stationary except for minor swaying due to wind. A small bird can be seen flying across the lower section of the frame above some greenery.

### VIRAT_S_040005_08_001225_001276

#### 8b

**Full-frame**

A black car drives through an intersection near a parking lot surrounded by greenery and buildings under clear skies.

**Full-frame 2 fps**

A black car drives through an intersection near a parking lot surrounded by greenery and buildings under clear skies.

**Motion ROI**

A person walks toward a parked green car near other vehicles before stepping away as the car drives off through a landscaped roundabout. Meanwhile, shadows cast by nearby structures move across a grassy field bordered by fencing under clear skies. In another part of the view, trees sway gently against a backdrop of brick buildings beneath a bright blue sky.

#### 8b-int8

**Full-frame**

A black car drives through an intersection near a parking lot surrounded by greenery and buildings under clear skies.

**Full-frame 2 fps**

A black car drives through an intersection near a grassy area surrounded by trees and buildings under clear skies.

**Motion ROI**

A person walks toward parked cars near a landscaped median before exiting the frame. A dark sedan drives through the intersection after making a turn, then proceeds down the road away from the parking lot entrance. The view shifts upward to focus on tree branches swaying gently against a clear blue sky with part of a brick building visible behind them.

#### 8b-fp8

**Full-frame**

A black car drives through an intersection near a parking lot surrounded by greenery and buildings under clear skies.

**Full-frame 2 fps**

A black car drives through an intersection near a parking lot surrounded by greenery and buildings under clear skies.

**Motion ROI**

A person walks toward a parked green car near other vehicles before getting inside and driving away through a landscaped intersection. Meanwhile, another individual runs across a grassy field adjacent to a fence while trees sway gently against a clear sky above buildings.

#### 30b

**Full-frame**

A black car drives from left to right through an intersection on a paved road within a campus-like setting featuring green lawns, trees, streetlights, and buildings under clear skies; other cars remain parked along the roadside throughout the clip.

**Full-frame 2 fps**

A black car drives from left to right through an intersection on a paved road near a grassy area; another dark-colored vehicle enters the frame from the bottom-left corner at 0:34 and proceeds toward the center before exiting the view.

**Motion ROI**

In the parking lot on the left side of the frame, a person wearing red walks past parked cars before getting into a dark green sedan; the car then drives out of the parking space, makes a right turn onto the road, and continues driving away down the street. On the grassy lawn near the fence line on the lower right portion of the image, shadows are cast that shift slightly due to movement outside the view. In the background above the trees, birds can be seen flying across the sky.

#### 30b-fp8

**Full-frame**

A black car drives from left to right through an intersection on a paved road within a campus-like setting featuring green lawns, trees, and buildings under clear skies.

**Full-frame 2 fps**

A black car drives from left to right through an intersection on a paved road within a campus-like setting featuring green lawns, trees, and buildings under clear skies.

**Motion ROI**

In the parking lot on the left side of the frame, a person wearing red walks past parked cars before getting into a dark green sedan; the car then drives out of the parking space, makes a right turn onto the road, and continues driving away down the street. On the grassy lawn near the fence line on the lower right side of the image, shadows are cast that shift slightly due to movement outside the view. In the background behind the trees, birds can be seen flying across the sky above the building.
