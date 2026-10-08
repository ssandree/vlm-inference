# Full-frame vs Motion Multi-ROI

같은 generation setting으로 생성한 결과를 video별로 나란히 기록한다. 어느 condition이 더 좋은지는 판정하지 않는다.

## Setup

- Prompts:
  - Full-frame: `prompts/frames_full.txt`
  - Motion ROI: `prompts/frames_roi_per_tube.txt`
- Sampling: full-frame uniform 1.0 fps; ROI sequence는 저장된 Stage3 tube의 1 fps crop.
- Motion ROI packing: each tube is `[Global]` once at the ROI start, then `[ROI#]` and that ROI video. No anchor.
- Models: 8b-fp16

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
| VIRAT_S_000001 | 8b-fp16 | Full-frame | 691 | 15570 | 1432857600 | 90.622 | ok |
| VIRAT_S_000001 | 8b-fp16 | Motion ROI | 649 | 45972 | 1174522880 | 165.514 | ok |
| VIRAT_S_000201_04_000682_000822 | 8b-fp16 | Full-frame | 141 | 17040 | 129945600 | 63.398 | ok |
| VIRAT_S_000201_04_000682_000822 | 8b-fp16 | Motion ROI | 186 | 15920 | 65302528 | 66.440 | ok |
| VIRAT_S_000205_02_000409_000566 | 8b-fp16 | Full-frame | 158 | 18960 | 145612800 | 61.066 | ok |
| VIRAT_S_000205_02_000409_000566 | 8b-fp16 | Motion ROI | 172 | 14412 | 78868480 | 54.121 | ok |
| VIRAT_S_000205_05_001092_001124 | 8b-fp16 | Full-frame | 33 | 4080 | 30412800 | 31.130 | ok |
| VIRAT_S_000205_05_001092_001124 | 8b-fp16 | Motion ROI | 39 | 4376 | 13125632 | 29.282 | ok |
| VIRAT_S_010000_07_000827_000860 | 8b-fp16 | Full-frame | 27 | 3360 | 24883200 | 9.284 | ok |
| VIRAT_S_010000_07_000827_000860 | 8b-fp16 | Motion ROI | 33 | 3872 | 12369920 | 30.724 | ok |
| VIRAT_S_010001_02_000195_000498 | 8b-fp16 | Full-frame | 301 | 20385 | 277401600 | 39.395 | ok |
| VIRAT_S_010001_02_000195_000498 | 8b-fp16 | Motion ROI | 301 | 23004 | 222564352 | 77.589 | ok |
| VIRAT_S_010001_03_000537_000563 | 8b-fp16 | Full-frame | 22 | 2640 | 20275200 | 21.116 | ok |
| VIRAT_S_010001_03_000537_000563 | 8b-fp16 | Motion ROI | 52 | 3660 | 9117696 | 20.953 | ok |
| VIRAT_S_010001_05_000649_000684 | 8b-fp16 | Full-frame | 28 | 3360 | 25804800 | 31.504 | ok |
| VIRAT_S_010001_05_000649_000684 | 8b-fp16 | Motion ROI | 48 | 3600 | 8183808 | 26.571 | ok |
| VIRAT_S_010001_08_000826_000893 | 8b-fp16 | Full-frame | 60 | 7200 | 55296000 | 14.822 | ok |
| VIRAT_S_010001_08_000826_000893 | 8b-fp16 | Motion ROI | 55 | 1572 | 4395008 | 10.259 | ok |
| VIRAT_S_010001_09_000921_000952 | 8b-fp16 | Full-frame | 25 | 3120 | 23040000 | 11.486 | ok |
| VIRAT_S_010001_09_000921_000952 | 8b-fp16 | Motion ROI | 36 | 1392 | 4472832 | 12.718 | ok |
| VIRAT_S_010001_10_000962_001005 | 8b-fp16 | Full-frame | 36 | 4320 | 33177600 | 12.215 | ok |
| VIRAT_S_010001_10_000962_001005 | 8b-fp16 | Motion ROI | 67 | 3444 | 8327168 | 18.774 | ok |
| VIRAT_S_010002_02_000174_000204 | 8b-fp16 | Full-frame | 28 | 3360 | 25804800 | 12.234 | ok |
| VIRAT_S_010002_02_000174_000204 | 8b-fp16 | Motion ROI | 43 | 3580 | 8714240 | 34.640 | ok |
| VIRAT_S_010002_03_000236_000261 | 8b-fp16 | Full-frame | 23 | 2880 | 21196800 | 9.947 | ok |
| VIRAT_S_010002_03_000236_000261 | 8b-fp16 | Motion ROI | 53 | 3036 | 8163328 | 19.559 | ok |
| VIRAT_S_010002_06_000441_000467 | 8b-fp16 | Full-frame | 20 | 2400 | 18432000 | 51.767 | ok |
| VIRAT_S_010002_06_000441_000467 | 8b-fp16 | Motion ROI | 38 | 3960 | 9052160 | 15.142 | ok |
| VIRAT_S_010002_07_000522_000547 | 8b-fp16 | Full-frame | 17 | 2160 | 15667200 | 29.307 | ok |
| VIRAT_S_010002_07_000522_000547 | 8b-fp16 | Motion ROI | 22 | 864 | 2580480 | 15.830 | ok |
| VIRAT_S_010003_02_000165_000202 | 8b-fp16 | Full-frame | 37 | 4560 | 34099200 | 15.422 | ok |
| VIRAT_S_010003_02_000165_000202 | 8b-fp16 | Motion ROI | 73 | 5184 | 16768000 | 31.605 | ok |
| VIRAT_S_010003_03_000219_000259 | 8b-fp16 | Full-frame | 41 | 5040 | 37785600 | 27.148 | ok |
| VIRAT_S_010003_03_000219_000259 | 8b-fp16 | Motion ROI | 110 | 5860 | 17246208 | 26.263 | ok |
| VIRAT_S_010003_06_000526_000560 | 8b-fp16 | Full-frame | 30 | 3600 | 27648000 | 31.229 | ok |
| VIRAT_S_010003_06_000526_000560 | 8b-fp16 | Motion ROI | 88 | 4832 | 11069440 | 23.887 | ok |
| VIRAT_S_010003_07_000608_000636 | 8b-fp16 | Full-frame | 22 | 2640 | 20275200 | 17.108 | ok |
| VIRAT_S_010003_07_000608_000636 | 8b-fp16 | Motion ROI | 54 | 2956 | 7278592 | 24.949 | ok |
| VIRAT_S_010003_10_000901_000934 | 8b-fp16 | Full-frame | 28 | 3360 | 25804800 | 12.550 | ok |
| VIRAT_S_010003_10_000901_000934 | 8b-fp16 | Motion ROI | 58 | 4872 | 13828096 | 39.839 | ok |
| VIRAT_S_010005_03_000215_000270 | 8b-fp16 | Full-frame | 52 | 6240 | 47923200 | 13.967 | ok |
| VIRAT_S_010005_03_000215_000270 | 8b-fp16 | Motion ROI | 79 | 7752 | 25200640 | 22.471 | ok |
| VIRAT_S_010005_05_000397_000430 | 8b-fp16 | Full-frame | 25 | 3120 | 23040000 | 9.009 | ok |
| VIRAT_S_010005_05_000397_000430 | 8b-fp16 | Motion ROI | 51 | 3600 | 9000960 | 24.912 | ok |
| VIRAT_S_010106_00_000098_000315 | 8b-fp16 | Full-frame | 212 | 18020 | 195379200 | 60.941 | ok |
| VIRAT_S_010106_00_000098_000315 | 8b-fp16 | Motion ROI | 276 | 26432 | 89324544 | 87.961 | ok |
| VIRAT_S_010106_02_000656_000692 | 8b-fp16 | Full-frame | 36 | 4320 | 33177600 | 30.157 | ok |
| VIRAT_S_010106_02_000656_000692 | 8b-fp16 | Motion ROI | 73 | 4800 | 11411456 | 20.082 | ok |
| VIRAT_S_010106_05_000954_000996 | 8b-fp16 | Full-frame | 29 | 3600 | 26726400 | 11.444 | ok |
| VIRAT_S_010106_05_000954_000996 | 8b-fp16 | Motion ROI | 53 | 3260 | 7331840 | 22.858 | ok |
| VIRAT_S_010107_00_000019_000057 | 8b-fp16 | Full-frame | 38 | 4560 | 35020800 | 12.009 | ok |
| VIRAT_S_010107_00_000019_000057 | 8b-fp16 | Motion ROI | 56 | 3640 | 8527872 | 22.100 | ok |
| VIRAT_S_010107_02_000282_000312 | 8b-fp16 | Full-frame | 21 | 2640 | 19353600 | 9.069 | ok |
| VIRAT_S_010107_02_000282_000312 | 8b-fp16 | Motion ROI | 44 | 5124 | 11046912 | 17.915 | ok |
| VIRAT_S_010109_07_000876_000910 | 8b-fp16 | Full-frame | 35 | 4320 | 32256000 | 25.792 | ok |
| VIRAT_S_010109_07_000876_000910 | 8b-fp16 | Motion ROI | 46 | 1780 | 4849664 | 11.640 | ok |
| VIRAT_S_010110_00_000000_000021 | 8b-fp16 | Full-frame | 22 | 2640 | 20275200 | 8.409 | ok |
| VIRAT_S_010110_00_000000_000021 | 8b-fp16 | Motion ROI | 66 | 3964 | 9388032 | 16.892 | ok |
| VIRAT_S_010110_02_000513_000566 | 8b-fp16 | Full-frame | 46 | 5520 | 42393600 | 11.748 | ok |
| VIRAT_S_010110_02_000513_000566 | 8b-fp16 | Motion ROI | 70 | 5372 | 17367040 | 18.763 | ok |
| VIRAT_S_010110_05_000899_000935 | 8b-fp16 | Full-frame | 29 | 3600 | 26726400 | 9.921 | ok |
| VIRAT_S_010110_05_000899_000935 | 8b-fp16 | Motion ROI | 56 | 2336 | 6356992 | 17.358 | ok |
| VIRAT_S_010111_04_000718_000760 | 8b-fp16 | Full-frame | 42 | 5040 | 38707200 | 13.197 | ok |
| VIRAT_S_010111_04_000718_000760 | 8b-fp16 | Motion ROI | 99 | 5276 | 12152832 | 22.997 | ok |
| VIRAT_S_010111_05_000762_000799 | 8b-fp16 | Full-frame | 29 | 3600 | 26726400 | 10.829 | ok |
| VIRAT_S_010111_05_000762_000799 | 8b-fp16 | Motion ROI | 50 | 4568 | 14376960 | 18.466 | ok |
| VIRAT_S_010111_06_000820_000860 | 8b-fp16 | Full-frame | 38 | 4560 | 35020800 | 11.032 | ok |
| VIRAT_S_010111_06_000820_000860 | 8b-fp16 | Motion ROI | 43 | 4472 | 23093248 | 21.835 | ok |
| VIRAT_S_010111_07_000872_000909 | 8b-fp16 | Full-frame | 34 | 4080 | 31334400 | 27.158 | ok |
| VIRAT_S_010111_07_000872_000909 | 8b-fp16 | Motion ROI | 39 | 4328 | 18585600 | 22.453 | ok |
| VIRAT_S_010111_08_000920_000954 | 8b-fp16 | Full-frame | 27 | 3360 | 24883200 | 10.830 | ok |
| VIRAT_S_010111_08_000920_000954 | 8b-fp16 | Motion ROI | 72 | 4196 | 9666560 | 26.344 | ok |
| VIRAT_S_010111_09_000981_001014 | 8b-fp16 | Full-frame | 15 | 1920 | 13824000 | 16.817 | ok |
| VIRAT_S_010111_09_000981_001014 | 8b-fp16 | Motion ROI | 21 | 2360 | 5414912 | 15.358 | ok |
| VIRAT_S_010113_02_000434_000479 | 8b-fp16 | Full-frame | 42 | 5040 | 38707200 | 12.158 | ok |
| VIRAT_S_010113_02_000434_000479 | 8b-fp16 | Motion ROI | 57 | 4648 | 13780992 | 17.651 | ok |
| VIRAT_S_010113_05_000776_000805 | 8b-fp16 | Full-frame | 24 | 2880 | 22118400 | 11.581 | ok |
| VIRAT_S_010113_05_000776_000805 | 8b-fp16 | Motion ROI | 57 | 3956 | 10761216 | 23.131 | ok |
| VIRAT_S_010113_07_000965_001013 | 8b-fp16 | Full-frame | 39 | 4800 | 35942400 | 14.067 | ok |
| VIRAT_S_010113_07_000965_001013 | 8b-fp16 | Motion ROI | 52 | 5560 | 22183936 | 35.848 | ok |
| VIRAT_S_010114_02_000765_000802 | 8b-fp16 | Full-frame | 32 | 3840 | 29491200 | 12.610 | ok |
| VIRAT_S_010114_02_000765_000802 | 8b-fp16 | Motion ROI | 95 | 6396 | 14069760 | 24.184 | ok |
| VIRAT_S_010116_00_000061_000121 | 8b-fp16 | Full-frame | 60 | 7200 | 55296000 | 15.685 | ok |
| VIRAT_S_010116_00_000061_000121 | 8b-fp16 | Motion ROI | 61 | 7008 | 14917632 | 41.655 | ok |
| VIRAT_S_010200_02_000349_000398 | 8b-fp16 | Full-frame | 45 | 5520 | 41472000 | 15.057 | ok |
| VIRAT_S_010200_02_000349_000398 | 8b-fp16 | Motion ROI | 36 | 1900 | 5984256 | 21.589 | ok |
| VIRAT_S_010201_02_000167_000197 | 8b-fp16 | Full-frame | 21 | 2640 | 19353600 | 9.601 | ok |
| VIRAT_S_010201_02_000167_000197 | 8b-fp16 | Motion ROI | 35 | 1704 | 4239360 | 10.183 | ok |
| VIRAT_S_010203_09_001010_001036 | 8b-fp16 | Full-frame | 26 | 3120 | 23961600 | 10.596 | ok |
| VIRAT_S_010203_09_001010_001036 | 8b-fp16 | Motion ROI | 36 | 1912 | 4595712 | 14.251 | ok |
| VIRAT_S_010204_03_000606_000632 | 8b-fp16 | Full-frame | 18 | 2160 | 16588800 | 23.267 | ok |
| VIRAT_S_010204_03_000606_000632 | 8b-fp16 | Motion ROI | 38 | 3528 | 8681472 | 28.706 | ok |
| VIRAT_S_010204_10_001372_001395 | 8b-fp16 | Full-frame | 20 | 2400 | 18432000 | 9.173 | ok |
| VIRAT_S_010204_10_001372_001395 | 8b-fp16 | Motion ROI | 29 | 1988 | 5156864 | 13.178 | ok |
| VIRAT_S_010205_03_000370_000395 | 8b-fp16 | Full-frame | 21 | 2640 | 19353600 | 28.808 | ok |
| VIRAT_S_010205_03_000370_000395 | 8b-fp16 | Motion ROI | 27 | 1848 | 4472832 | 20.038 | ok |
| VIRAT_S_010205_04_000545_000576 | 8b-fp16 | Full-frame | 25 | 3120 | 23040000 | 24.725 | ok |
| VIRAT_S_010205_04_000545_000576 | 8b-fp16 | Motion ROI | 50 | 3528 | 8183808 | 15.261 | ok |
| VIRAT_S_010207_02_000790_000816 | 8b-fp16 | Full-frame | 25 | 3120 | 23040000 | 27.293 | ok |
| VIRAT_S_010207_02_000790_000816 | 8b-fp16 | Motion ROI | 35 | 2472 | 5750784 | 17.334 | ok |
| VIRAT_S_010207_09_001484_001510 | 8b-fp16 | Full-frame | 20 | 2400 | 18432000 | 9.398 | ok |
| VIRAT_S_010207_09_001484_001510 | 8b-fp16 | Motion ROI | 26 | 1200 | 3174400 | 8.921 | ok |
| VIRAT_S_010207_10_001549_001596 | 8b-fp16 | Full-frame | 43 | 5280 | 39628800 | 16.467 | ok |
| VIRAT_S_010207_10_001549_001596 | 8b-fp16 | Motion ROI | 54 | 3688 | 8101888 | 16.251 | ok |
| VIRAT_S_010208_09_000857_000886 | 8b-fp16 | Full-frame | 22 | 2640 | 20275200 | 30.986 | ok |
| VIRAT_S_010208_09_000857_000886 | 8b-fp16 | Motion ROI | 30 | 1784 | 4833280 | 16.513 | ok |
| VIRAT_S_040000_00_000063_000085 | 8b-fp16 | Full-frame | 23 | 2880 | 47692800 | 8.299 | ok |
| VIRAT_S_040000_00_000063_000085 | 8b-fp16 | Motion ROI | 30 | 1992 | 7055360 | 11.093 | ok |
| VIRAT_S_040000_01_000042_000099 | 8b-fp16 | Full-frame | 58 | 6960 | 120268800 | 13.684 | ok |
| VIRAT_S_040000_01_000042_000099 | 8b-fp16 | Motion ROI | 87 | 5856 | 18206720 | 20.002 | ok |
| VIRAT_S_040000_02_000111_000145 | 8b-fp16 | Full-frame | 35 | 4320 | 72576000 | 14.326 | ok |
| VIRAT_S_040000_02_000111_000145 | 8b-fp16 | Motion ROI | 39 | 1096 | 5326848 | 8.489 | ok |
| VIRAT_S_040000_04_000532_000622 | 8b-fp16 | Full-frame | 91 | 11040 | 188697600 | 22.137 | ok |
| VIRAT_S_040000_04_000532_000622 | 8b-fp16 | Motion ROI | 107 | 10920 | 25249792 | 19.426 | ok |
| VIRAT_S_040005_00_000003_000058 | 8b-fp16 | Full-frame | 56 | 6720 | 116121600 | 13.411 | ok |
| VIRAT_S_040005_00_000003_000058 | 8b-fp16 | Motion ROI | 68 | 7068 | 22809600 | 16.836 | ok |
| VIRAT_S_040005_08_001225_001276 | 8b-fp16 | Full-frame | 52 | 6240 | 107827200 | 12.030 | ok |
| VIRAT_S_040005_08_001225_001276 | 8b-fp16 | Motion ROI | 99 | 7300 | 19412480 | 23.697 | ok |

## Descriptions

### VIRAT_S_000001

#### 8b-fp16

**Full-frame**

A white car remains parked near a brick building throughout most of the footage until two individuals approach from the left side of the frame; shortly after, another vehicle drives past on the road adjacent to the parking lot while pedestrians walk along sidewalks nearby. The first group walks toward the center before dispersing into smaller groups around the area, including some walking towards an exit at the far end of the lot. A second individual enters later and stands alone for several moments beside the initial pair before moving away again. Another person appears briefly standing next to the parked car before leaving too. Later, more people enter through various entrances: three arrive together followed by others individually, eventually forming clusters scattered across multiple sections within viewable range. One man exits his own vehicle which pulls up close to the original spot occupied earlier but then departs quickly without interacting further. Meanwhile, additional figures continue entering and exiting via pathways leading off-screen beyond the immediate field-of-view boundaries. Throughout these movements, no significant changes occur regarding background elements such as buildings, trees, signage, or lighting conditions indicating any major environmental shift during this period observed under consistent daylight illumination.

**Motion ROI**

The video begins with a stationary silver car parked near a brick building adjacent to an empty parking lot marked with white lines. Shortly after, two individuals appear on the sidewalk beside the car, engaging in conversation while standing still before beginning to walk away together toward the right side of the frame. As they move along, another person enters the scene walking towards them from further down the path, eventually joining their group as they continue moving forward across the pavement. Meanwhile, additional pedestrians can be seen entering and exiting various parts of the vicinity; some approach the vehicle briefly before proceeding past it, while others remain engaged in casual interactions nearby. The overall activity remains calm throughout this period, characterized primarily by these movements around the central location involving both vehicles and people navigating through the open space under clear daylight conditions. Eventually, more groups emerge onto the sidewalks, including pairs that stroll leisurely alongside each other, contributing to the dynamic yet tranquil atmosphere observed within the setting. Throughout the duration, no significant changes occur beyond routine pedestrian traffic patterns interacting naturally with the environment established initially during the opening moments of observation.

### VIRAT_S_000201_04_000682_000822

#### 8b-fp16

**Full-frame**

A silver sedan drives into the parking lot from the bottom left corner and parks near the center-left section. A person exits the vehicle and walks toward the right side of the frame while another individual approaches them from the opposite direction along the road adjacent to the parking area. Two pedestrians cross the street at an intersection on the far right edge of the view before walking away towards the upper-right portion of the screen. Several cars pass through the roadway running alongside the parking space throughout the duration of the footage, including black sedans, white SUVs, and other vehicles moving both directions. In addition, multiple individuals can be seen entering and exiting their parked automobiles within various sections of the lot during this period. The overall environment remains consistent with no significant changes observed beyond these movements over time.

**Motion ROI**

A silver sedan drives through the parking lot entrance while pedestrians walk along the sidewalk adjacent to parked cars near a road with moving vehicles. A person wearing high-visibility clothing walks across the street toward the right side of the frame before continuing their path alongside other individuals walking nearby. In another part of the scene, two people are seen standing together on the far end of the parking lot beside several parked vehicles, appearing engaged in conversation. Meanwhile, inside a building located behind trees, someone moves around within an open doorway that overlooks the parking area below. Another individual exits this same building shortly after entering, stepping out onto the pavement outside. Throughout these moments, various automobiles pass by both lanes of traffic flowing past the edge of the parking facility, including sedans and SUVs traveling in opposite directions. The overall activity continues steadily as more drivers enter and exit the premises, maintaining consistent movement throughout the environment.

### VIRAT_S_000205_02_000409_000566

#### 8b-fp16

**Full-frame**

A black pickup truck drives past on the road adjacent to the parking lot while pedestrians walk along the sidewalk near parked cars. A silver sedan enters from the bottom left corner of the frame and proceeds toward the right side before exiting out of view. Two individuals cross the street diagonally through an intersection marked with white lines, moving from the lower-left area towards the upper-right section of the screen. Another person walks alone down the sidewalk at the far end of the parking lot, heading away from the viewer's perspective. Several vehicles pass by continuously on the main roadway running alongside the parking facility throughout the duration of the footage. The overall environment remains consistent with no significant changes observed within this timeframe.

**Motion ROI**

A black pickup truck drives past on the road adjacent to the parking lot while pedestrians walk along the sidewalk near parked cars. A person walks across the street toward the entrance of the building with several vehicles passing by them. Two individuals cross the intersection together after walking through the parking lot towards the right side of the frame. Another pedestrian enters the view from the left edge of the screen and proceeds down the sidewalk beside the roadway. Multiple cars continue moving along the road throughout this period including a silver sedan that passes close to the curb before exiting out-of-frame. The traffic flow remains steady with various automobiles traveling both directions along the main thoroughfare bordering the large open-air parking facility filled with numerous parked vehicles under clear skies.

### VIRAT_S_000205_05_001092_001124

#### 8b-fp16

**Full-frame**

A black car drives past on the road adjacent to the parking lot while several other vehicles remain parked within the lot. A silver minivan then enters from the right side of the frame and proceeds down the road before exiting out of view. Shortly after, another vehicle follows along the same path, continuing through the intersection without stopping. Throughout these movements, pedestrians can be seen walking near the entrance area at the far end of the parking lot but do not interact directly with any cars passing by. The surrounding environment remains unchanged throughout this period, featuring brick buildings flanking both sides of the parking structure surrounded by greenery under an overcast sky.

**Motion ROI**

A black car drives along the road adjacent to the parking lot, passing near parked vehicles and moving toward the right side of the frame. Two individuals walk across the parking lot’s open space, heading towards the far end while remaining within view of the brick buildings behind them. A silver minivan enters the roadway from the bottom left corner, driving straight ahead before exiting out of sight on the opposite side. Another vehicle follows closely after, maintaining steady movement through the intersection with no apparent delay. The traffic continues flowing smoothly past the stationary cars lined up neatly beside the curb under clear daylight conditions.

### VIRAT_S_010000_07_000827_000860

#### 8b-fp16

**Full-frame**

People ascend and descend the stairs leading into the building while others walk along pathways near tables under red umbrellas on a sunny day.

**Motion ROI**

People ascend and descend the wide staircase leading up to the building entrance while others stroll along the upper walkway near the glass doors. Several individuals move through the outdoor seating area beneath red umbrellas on the paved plaza below, with some walking past tables toward the grassy slope behind them. A person walks across the lower patio carrying something before disappearing out of view among the umbrellas. Another individual briefly appears ascending the stairs on the left side of the frame before exiting upward towards the second level balcony overlooking the courtyard. The palm tree situated centrally between two sets of steps remains stationary throughout these movements against the backdrop of distant mountains under clear skies.

### VIRAT_S_010001_02_000195_000498

#### 8b-fp16

**Full-frame**

A person sits at an outdoor table under a red umbrella while others walk around the plaza near stairs leading up to buildings, some ascending and descending, against a backdrop of green grassy slopes and distant mountains.

**Motion ROI**

The video opens with a wide view of an outdoor plaza featuring numerous red umbrellas shading tables on a paved surface, bordered by green grassy areas and modern buildings under clear skies; people are seen walking up stairs leading to entrances while others sit quietly beneath the umbrellas. A person walks across the foreground near some seated individuals before moving out of frame. In another section, someone exits through glass doors onto the patio, followed shortly after by two more individuals entering the building entrance nearby. Meanwhile, further along the walkway, several pedestrians pass between planters and seating arrangements, continuing their paths toward various parts of the campus grounds. Finally, attention shifts briefly to a close-up shot focusing on one individual sitting alone under an umbrella, seemingly engrossed in reading material placed upon the table beside them.

### VIRAT_S_010001_03_000537_000563

#### 8b-fp16

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual sits at an outdoor table under a red umbrella near the center foreground. In the background, several people move around the building entrance area, including individuals ascending and descending the staircase and walking along pathways adjacent to the grassy slope. The surrounding environment remains largely unchanged throughout the clip, featuring multiple tables shaded by umbrellas, palm trees, and distant mountains under clear skies.

**Motion ROI**

Underneath red umbrellas on a sunny plaza, a person sits alone at a table engrossed with their phone while others walk up and down nearby stairs leading to a modern building set against distant mountains. A woman walks across a grassy lawn toward a brick patio lined with tables under more umbrellas before exiting frame near a glass door entrance. The sky above remains clear blue throughout the tranquil outdoor setting.

### VIRAT_S_010001_05_000649_000684

#### 8b-fp16

**Full-frame**

A person walks from left to right near the entrance of a modern building while another individual stands nearby under an umbrella on the patio area. Two people walk together toward the building’s entrance before stopping briefly at the doorway. A third person enters the frame walking towards the stairs leading up to the main entrance. Another individual exits through the glass doors into the courtyard space. Throughout these movements, several other individuals remain seated beneath red umbrellas scattered around the paved plaza, some engaged in conversation or looking down at their phones. The background features greenery including palm trees and shrubs along stone retaining walls surrounding grassy slopes adjacent to the buildings. In the distance, mountains rise against a clear blue sky above the campus setting.

**Motion ROI**

A person walks down a set of outdoor stairs while another individual ascends them nearby, both moving past railings with greenery behind them under bright sunlight. In the courtyard below, several people stroll across paved areas near tables shaded by red umbrellas, some pausing briefly before continuing their paths around landscaped planters and trees adjacent to modern buildings. A few individuals enter through glass doors on the right side of the building facade, while others exit toward the left, blending seamlessly within the open-air environment surrounded by hills in the distance.

### VIRAT_S_010001_08_000826_000893

#### 8b-fp16

**Full-frame**

A serene outdoor plaza featuring numerous red umbrellas shading tables remains undisturbed throughout the video, set against modern buildings and lush greenery under clear skies.

**Motion ROI**

A person walks down a set of outdoor stairs near red umbrellas while another individual exits a building entrance nearby, followed by someone walking across a grassy lawn adjacent to patio tables under similar umbrellas.

### VIRAT_S_010001_09_000921_000952

#### 8b-fp16

**Full-frame**

A person walks down the stairs on the left side of the frame near the building entrance while another individual sits at an outdoor table under an umbrella further back toward the right side of the courtyard.

**Motion ROI**

A person wearing a pink shirt walks across the patio near potted plants while another individual exits through glass doors nearby. A pedestrian ascends the staircase leading up to the elevated walkway above the courtyard. The camera focuses on red umbrellas arranged neatly around tables under bright sunlight.

### VIRAT_S_010001_10_000962_001005

#### 8b-fp16

**Full-frame**

A person walks from left to right across the foreground plaza beneath red umbrellas while others move near the building entrance steps and around tables on the grassy slope behind them.

**Motion ROI**

In the courtyard with red umbrellas and green lawns, people walk around near tables and buildings while others ascend stairs leading to entrances under bright sunlight. A person wearing blue walks across the paved area toward steps beside outdoor seating before exiting frame. Another individual runs through the open plaza past empty tables beneath large umbrellas, moving swiftly across the space.

### VIRAT_S_010002_02_000174_000204

#### 8b-fp16

**Full-frame**

A wide outdoor plaza featuring numerous red umbrellas over tables remains largely empty under clear skies, with people walking through the area near the building entrance and along pathways adjacent to green lawns and trees.

**Motion ROI**

The video opens with a wide view of an outdoor plaza featuring numerous red umbrellas shading tables on a paved surface, bordered by a grassy slope leading up to modern buildings under clear skies; people are seen walking along pathways near the building entrances and across the courtyard. A person walks briskly past several tables beneath open umbrellas while another individual strolls leisurely through the shaded seating areas nearby. Two individuals walk together toward the left side of the frame, passing between rows of empty tables before exiting the immediate foreground. In the background, someone enters the building’s main entrance after briefly pausing outside, followed shortly thereafter by others entering and leaving the facility's doors. The overall atmosphere remains calm throughout, capturing casual movement within this sunny campus setting.

### VIRAT_S_010002_03_000236_000261

#### 8b-fp16

**Full-frame**

A few people walk up and down the stairs leading into the building on the left side of the frame while others stroll along the pathway near the grassy hill behind them.

**Motion ROI**

People walk toward and enter the building entrance under the covered awning near two trees on the grassy lawn. A person walks across the paved patio area with tables and red umbrellas while another individual ascends the wide staircase leading up to the main building’s entryway. Someone strolls along the brick-paved pathway adjacent to outdoor seating areas beside the building's windows.

### VIRAT_S_010002_06_000441_000467

#### 8b-fp16

**Full-frame**

A person walks down the stairs leading from the building into the courtyard area below. Another individual enters the frame walking toward the left side of the screen near the bottom edge while passing through an open space between tables under red umbrellas. A third person appears briefly at the top right corner moving towards the entrance of the adjacent structure before disappearing out of view behind some trees. Throughout these movements, several other people can be seen seated around various table setups scattered throughout the outdoor plaza; however, their specific activities remain largely unchanged during this period. The background features include modern architectural elements such as glass windows on buildings flanked by staircases adorned with railings, alongside landscaped greenery including grassy slopes bordered by low stone walls and potted plants placed strategically along pathways. In addition, multiple tall lampposts equipped with circular light fixtures stand evenly spaced among clusters of dining furniture beneath large orange parasols providing shade over black metal chairs paired with matching rectangular tabletops arranged for patrons enjoying meals outdoors. Distant mountains rise majestically against a clear blue sky above the entire setting, adding depth and natural beauty to the urban environment depicted within the shot.

**Motion ROI**

In the foreground courtyard with red umbrellas and tables, people walk across the paved areas near the building entrances while others ascend and descend the wide staircase leading up to the upper level seating area surrounded by greenery and palm trees under bright sunlight.

### VIRAT_S_010002_07_000522_000547

#### 8b-fp16

**Full-frame**

A person walks from left to right along the pathway near the building entrance while another individual moves toward them before continuing their path. A few people walk through the outdoor seating area under red umbrellas, moving between tables without stopping. One person exits the frame on the far right side after walking past several tables. Another individual enters from behind some trees at the upper level and proceeds towards the center-right portion of the courtyard. Two more individuals appear briefly near the top-left corner, standing close together for a moment before departing out of view. Throughout these movements, no significant changes occur within the surrounding environment such as lighting conditions or background elements like buildings, grassy hills, or distant mountains.

**Motion ROI**

In the foreground courtyard with red umbrellas and tables, two individuals walk across the paved area near the grassy slope before stopping beside a table under an umbrella. Meanwhile, another person walks past on the far side of the lawn toward the building entrance while carrying something blue. Above them, tree branches sway gently against the clear sky above the roofline of the modern architecture.

### VIRAT_S_010003_02_000165_000202

#### 8b-fp16

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual stands near the building entrance further up the steps; meanwhile, two people sit at an outdoor table under a red umbrella in the foreground, remaining seated throughout the clip.

**Motion ROI**

People walk up and down the stairs on the left side of the courtyard near the building entrance while others stroll across the grassy lawn toward the right side under the shade of trees. A person sits beneath an orange umbrella at a table in the foreground, remaining still throughout the clip. Another individual walks past tables with umbrellas along the lower patio level before disappearing out of frame. In the background, people move around various seating areas adjacent to large windows overlooking the central green space. The entire scene remains calm and quiet except for these subtle movements within this sunny outdoor setting surrounded by mountains in the distance.

### VIRAT_S_010003_03_000219_000259

#### 8b-fp16

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual sits at an outdoor table under a red umbrella near the bottom-left corner. In the background, people move around various areas including walking up steps and passing through doorways within the building complex situated behind a grassy lawn bordered by trees and more tables shaded by umbrellas. A few individuals can be seen seated at other tables scattered throughout the courtyard area, enjoying their surroundings amidst the sunny weather and distant mountains providing a scenic backdrop.

**Motion ROI**

People walk up and down stairs on the left side while others stroll across the courtyard near tables with red umbrellas under bright sunlight. Two individuals sit together beneath a large umbrella, engaged in conversation beside a table positioned close to the staircase. A person walks past seated patrons nearby before moving toward another part of the outdoor seating area. The view remains consistent throughout, capturing movement within this open-air campus setting surrounded by buildings and greenery against distant mountains.

### VIRAT_S_010003_06_000526_000560

#### 8b-fp16

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual sits at an outdoor table under a red umbrella near the bottom-left corner. A few people move around in the background near the building entrance and along pathways adjacent to the grassy area, including someone walking from right to left past tables toward the center-right portion of the courtyard. Another person enters the frame from the lower-left edge and moves towards the central seating area before exiting out of view. The overall setting remains consistent throughout, featuring multiple red umbrellas over black tables scattered across a paved patio surrounding a green lawn bordered by trees and buildings, with mountains visible in the distance behind them.

**Motion ROI**

In the foreground courtyard with red umbrellas, two individuals are seated at a table engaged in conversation while people walk past them on the paved ground near the stairs leading up to the building entrance. Meanwhile, another person walks across the patio adjacent to the glass doors under more umbrellas before descending steps toward the right side of the frame. Additionally, someone briefly appears walking along the lower edge of the view towards the bottom left corner.

### VIRAT_S_010003_07_000608_000636

#### 8b-fp16

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual sits at an outdoor table under a red umbrella near the bottom-left corner. A third person enters from the far right edge of the frame walking toward the center area beneath several umbrellas. The background features buildings surrounded by greenery and mountains under a partly cloudy sky.

**Motion ROI**

In the courtyard with red umbrellas and outdoor seating, two individuals are seated under an umbrella engaged in conversation while others walk past on the stairs leading up to the building entrance. A person walks down the steps toward the lower level near the patio tables before exiting the frame. Another individual briefly appears walking across the foreground near the bottom left corner of the view. In the background, someone exits through a doorway located beside the staircase adjacent to the main building facade. The overall setting remains calm throughout these movements within this sunny campus environment.

### VIRAT_S_010003_10_000901_000934

#### 8b-fp16

**Full-frame**

A person walks down stairs toward an outdoor seating area under red umbrellas while another individual ascends the steps nearby, both moving within the frame’s foreground near modern buildings and green lawns set against distant mountains.

**Motion ROI**

People walk across the paved courtyard under red umbrellas near stairs leading up to a building with large windows, while others ascend the staircase toward the entrance. A person walks past tables on the left side of the frame before moving out of view. Another individual descends the steps after reaching the top, followed closely by someone else walking down behind them. In another section, people stroll along a pathway beside outdoor seating areas beneath closed umbrellas; two individuals pass through this zone—one wearing green clothing—and then exit towards the right edge of the screen. Meanwhile, more pedestrians move around various parts of the campus grounds including grassy lawns bordered by planters and buildings featuring brick facades and glass doors. The overall atmosphere remains calm throughout these scenes set against a backdrop of mountains visible beyond trees lining the horizon.

### VIRAT_S_010005_03_000215_000270

#### 8b-fp16

**Full-frame**

People walk up and down stairs near buildings while others stroll along pathways under red umbrellas on a sunny day surrounded by green grass and mountains in the background.

**Motion ROI**

People walk across the courtyard near the building entrance while others ascend and descend the stairs on the left side of the frame. A person sits under an umbrella in the foreground patio area, remaining still throughout the clip. Another individual walks past tables with umbrellas toward the right edge of the scene before exiting view.

### VIRAT_S_010005_05_000397_000430

#### 8b-fp16

**Full-frame**

People ascend and descend the stairs leading into the building while others walk along the pathways near the outdoor seating area under red umbrellas.

**Motion ROI**

People are seen ascending and descending wide outdoor staircases adjacent to a modern building with brick walls and large windows, while others walk along pathways nearby under red umbrellas that shade tables on a paved plaza below. A person walks across the foreground near some patio furniture before moving out of view. Another individual strolls past planter boxes situated beside the stairs leading up toward the entrance of the building. The ground surface transitions between smooth pavement areas marked with white lines and sections covered in reddish-brown material.

### VIRAT_S_010106_00_000098_000315

#### 8b-fp16

**Full-frame**

A black sedan drives through a parking lot filled with various parked cars, moving from left to right before exiting the frame on the far side. A red car enters the frame from the top center, driving toward the bottom right corner while passing behind several stationary vehicles. Two individuals walk together along the lower-right edge of the parking lot near some trees, heading towards the right side of the screen. Another individual walks alone down the middle section of the parking lot, moving from the upper-left area downward toward the central part of the space. Multiple small white vans appear sequentially at the top-center portion of the parking lot, entering from off-screen and stopping among other parked vehicles without any further movement observed after their arrival.

**Motion ROI**

Two individuals walk across the lower left corner of the parking lot near some greenery before exiting the frame. A black SUV parked on the right side of the lot has two people walking past it toward the center of the lot, moving away from the camera's viewpoint. In another section of the lot, several cars are seen driving along the road adjacent to the parking spaces; vehicles move both directions with occasional stops and starts. Near the upper right portion of the lot, pedestrians can be observed crossing between rows of parked cars while others remain stationary nearby. On the far end of the lot, close to the building entrance, multiple people gather around a white van that appears to have just pulled up or stopped momentarily. Finally, towards the bottom edge of the view, someone walks through the shadowed area beside a tree-lined path leading out of sight behind foliage partially obscuring their movement.

### VIRAT_S_010106_02_000656_000692

#### 8b-fp16

**Full-frame**

Two individuals walk together through an empty parking lot toward the right side before separating; one continues walking while the other stops near parked vehicles. In the background, several cars remain stationary throughout the footage, including a blue car positioned centrally within the frame, surrounded by various models such as sedans and SUVs scattered across marked spaces under bright daylight conditions. A few people can be seen moving around on sidewalks adjacent to buildings at intervals but do not interact directly with anyone else present in the main area shown. The overall environment remains calm without any noticeable changes except for minor movements from pedestrians outside the primary focus zone.

**Motion ROI**

A person walks across the parking lot near several parked vans while another individual rides a bicycle along the sidewalk adjacent to a fenced sports field. Simultaneously, two cyclists ride past a street corner with traffic cones nearby, moving away from the intersection. In parallel, a cyclist continues riding on the path beside the fence enclosing the athletic field.

### VIRAT_S_010106_05_000954_000996

#### 8b-fp16

**Full-frame**

A person walks toward a parked black car in an open parking lot, pauses briefly near the driver’s side door, then steps back away from the vehicle while remaining within view.

**Motion ROI**

In a large parking lot with several parked cars, a person approaches a dark sedan, retrieves items from the trunk, places them on the ground beside the vehicle, then walks away toward the left side of the frame near a white container while leaving the car's rear door open. Meanwhile, across the lot, two individuals are seen riding bicycles along a paved path adjacent to a fenced sports field, moving steadily from right to left before disappearing out of view behind some trees.

### VIRAT_S_010107_00_000019_000057

#### 8b-fp16

**Full-frame**

A person rides a bicycle through an empty parking lot while another individual walks nearby, both moving from left to right under bright sunlight casting long shadows on the pavement.

**Motion ROI**

In the parking lot, a person pushes a cart across the asphalt toward the left side while several cars are parked around them near a fence line with trees behind. In another part of the frame, people walk along a sidewalk adjacent to buildings under bright sunlight casting long shadows on the ground. A tree partially obscures some vehicles nearby, including a black SUV, as someone walks past underneath it heading towards the right edge of the view.

### VIRAT_S_010107_02_000282_000312

#### 8b-fp16

**Full-frame**

A person rides a bicycle from right to left across the parking lot while two individuals walk together toward the lower-left corner near a white container unit.

**Motion ROI**

In the parking lot, a person rides a bicycle across the open space near parked cars and trees before exiting the frame. Meanwhile, two individuals walk together through another section of the lot, pushing a cart with items toward the lower left corner while passing by several vehicles under bright sunlight.

### VIRAT_S_010109_07_000876_000910

#### 8b-fp16

**Full-frame**

A person walks along the sidewalk on the left side of the parking lot while another individual stands near a parked car further down the row. A white van drives through an intersection at the far end of the lot before stopping next to other vehicles. The sun casts long shadows over the asphalt, highlighting stationary cars arranged neatly within marked spaces under clear skies. Trees line parts of the perimeter around buildings situated behind the expansive parking area. No significant movement or activity beyond these observations takes place throughout the duration shown.

**Motion ROI**

In the parking lot adjacent to a large building with green windows, several cars are parked while two individuals walk along the sidewalk near a fenced sports field; shortly after, another person rides a bicycle past them on the same path.

### VIRAT_S_010110_00_000000_000021

#### 8b-fp16

**Full-frame**

A person walks from right to left near parked cars at the bottom-right corner of the parking lot while vehicles remain stationary throughout the footage.

**Motion ROI**

A silver sedan parked near a white dumpster has a person walking past it carrying bags before exiting the frame. A gray SUV drives through the parking lot, moving from right to left across multiple rows of parked cars. The upper portion of the building with trees in front remains stationary throughout the footage.

### VIRAT_S_010110_02_000513_000566

#### 8b-fp16

**Full-frame**

A large parking lot filled with various parked cars remains stationary throughout the video, surrounded by buildings and trees under clear daylight conditions.

**Motion ROI**

In the parking lot adjacent to modern buildings with large windows, several people are seen riding bicycles along designated paths near parked cars and trees on a sunny day. Meanwhile, two individuals approach a white box placed beside some rocks and engage briefly before walking away together toward the right side of the frame.

### VIRAT_S_010110_05_000899_000935

#### 8b-fp16

**Full-frame**

A person on a scooter moves from left to right across the lower-left portion of the parking lot while cars remain stationary throughout the scene.

**Motion ROI**

In the parking lot near the fenced sports field, two individuals are seen walking across parked cars toward the right side of the frame. Meanwhile, another person walks briskly through a shadowed section of the parking lot heading towards the bottom left corner. A tall tree stands prominently against the backdrop of a large building on the far right edge of the scene.

### VIRAT_S_010111_04_000718_000760

#### 8b-fp16

**Full-frame**

A person walks from right to left near parked cars at the bottom-right corner of the parking lot while another individual briefly appears walking on the far-left side before disappearing behind vehicles.

**Motion ROI**

A person walks across a parking lot near parked cars, moving away from the frame after passing a silver sedan. Meanwhile, another individual rides a bicycle along a path adjacent to a fenced sports field, continuing their journey past trees and toward the edge of the viewable area. In the background, tall buildings with green-tinted windows stand behind several large trees that sway gently in the breeze.

### VIRAT_S_010111_05_000762_000799

#### 8b-fp16

**Full-frame**

A person walks from the bottom right corner toward the center of the parking lot while another individual moves along the far side near parked vehicles under bright daylight conditions.

**Motion ROI**

In the parking lot near the beige building with green-tinted windows, two individuals walk across the asphalt toward the right side of the frame while cars remain parked around them. A person carrying a yellow object walks past a white dumpster located beside a silver sedan on the lower-right portion of the view before exiting the shot.

### VIRAT_S_010111_06_000820_000860

#### 8b-fp16

**Full-frame**

A person walks from left to right across the parking lot near the center while another individual moves toward them before both continue walking away into the distance.

**Motion ROI**

Two individuals walk across the parking lot near parked cars on the left side, moving toward the center before exiting the frame. A person walks along the edge of the parking lot on the far right, passing by several vehicles including vans and sedans. Another individual enters the frame from the bottom-left corner, walking diagonally towards the upper-right section of the parking lot while carrying something dark-colored.

### VIRAT_S_010111_07_000872_000909

#### 8b-fp16

**Full-frame**

A white van drives through the parking lot from left to right near the center while several cars remain parked around it. A silver sedan exits on the far right side of the frame moving toward the bottom edge before disappearing out of view. Two pedestrians walk along the sidewalk at the top-left corner heading towards the middle section of the parking area. Another pedestrian walks alone down the path adjacent to the building located behind the upper-right portion of the parking space. The sun casts long shadows over parts of the asphalt surface throughout the entire duration of this footage.

**Motion ROI**

A white van drives through the parking lot entrance near the building on the right side of the frame. A person walks across the foreground toward the left edge of the view. Another individual crosses the parking lot diagonally from the upper-left corner towards the lower-right section. A silver sedan exits the parking space located near the bottom center of the image and moves out of sight along the road adjacent to the parking lot.

### VIRAT_S_010111_08_000920_000954

#### 8b-fp16

**Full-frame**

A person walks from left to right near parked cars on the far side of the lot while another individual moves toward them before continuing their path along the edge of the parking area.

**Motion ROI**

In the parking lot near the beige building with green windows, two individuals walk across the asphalt toward parked vehicles while another person walks nearby on the opposite side of the road. A man approaches a silver SUV parked beside other cars under tall trees before walking away after interacting briefly with someone inside the vehicle. Meanwhile, further down the sidewalk adjacent to a fenced sports field, two people stroll past parked cars heading towards the left edge of the frame, eventually stopping near some blue containers before continuing their path along the pavement.

### VIRAT_S_010111_09_000981_001014

#### 8b-fp16

**Full-frame**

A silver sedan drives into the parking lot from the right side and parks near other vehicles. A white van exits the parking area on the left side while another vehicle enters from the top center lane. Several cars move slowly within designated spaces throughout the frame without causing significant disruption to traffic flow. The overall activity remains calm with minimal movement observed among parked automobiles under clear daylight conditions.

**Motion ROI**

In the parking lot adjacent to a large building with light-colored walls and green-tinted windows, several cars are seen moving through designated lanes while others remain parked; on the left side near a fenced sports field, vehicles enter and exit the frame along a road bordered by trees and lampposts.

### VIRAT_S_010113_02_000434_000479

#### 8b-fp16

**Full-frame**

A red car drives slowly through the parking lot from left to right near the center while other vehicles remain stationary throughout the entire duration of the footage.

**Motion ROI**

A red car drives through the parking lot toward the building entrance while other vehicles remain stationary nearby. A person walks away from the edge of the frame near some trees on the left side of the scene. The view briefly darkens before returning to normal lighting conditions with no significant movement observed.

### VIRAT_S_010113_05_000776_000805

#### 8b-fp16

**Full-frame**

A person rides a bicycle from right to left along the edge of the parking lot near some parked vehicles while another individual walks briskly through the lower-right section of the frame before exiting out of view.

**Motion ROI**

In the parking lot near the fence line, two individuals walk across the asphalt toward the building entrance while several parked cars remain stationary nearby. Simultaneously, another person rides a bicycle through the open space between vehicles on the opposite side of the lot before stopping beside a black SUV partially obscured by trees. In the background, people are seen walking along the sidewalk adjacent to the multi-story office buildings under bright sunlight casting long shadows across the pavement.

### VIRAT_S_010113_07_000965_001013

#### 8b-fp16

**Full-frame**

A person walks from left to right near parked cars in the middle section of the parking lot while another individual exits a vehicle on the far right side before walking away toward the bottom edge of the frame.

**Motion ROI**

In the parking lot near the building entrance, two individuals walk across the asphalt toward parked cars while another person exits a vehicle nearby. A silver sedan drives through the central lanes before stopping beside other vehicles. Near the right side of the frame, a man walks past a white dumpster under sunlight casting shadows on the pavement. In the lower left corner, someone pushes a cart with items along the edge of the shadowed area next to trees. Another individual approaches the dumpster carrying boxes, then turns away after placing them down. Throughout these movements, several people are seen walking around various sections of the parking lot, including areas close to sports fields and buildings, indicating routine activity within this urban environment.

### VIRAT_S_010114_02_000765_000802

#### 8b-fp16

**Full-frame**

A person walks steadily through an empty parking lot filled with parked cars under bright sunlight, moving from right to left near the bottom edge of the frame while buildings and trees remain stationary in the background.

**Motion ROI**

A person walks across the parking lot near parked cars on the right side while another individual rides a bicycle along the edge of the field adjacent to the building. A white van drives past several other vehicles parked nearby under bright sunlight with trees casting shadows around them. The red car remains stationary next to the pickup truck positioned close to the fence bordering the sports field throughout this period.

### VIRAT_S_010116_00_000061_000121

#### 8b-fp16

**Full-frame**

A person rides a bicycle from left to right along the road adjacent to the parking lot while several cars remain stationary within the marked spaces under bright sunlight casting long shadows on the pavement.

**Motion ROI**

In the parking lot near the fenced sports field, two individuals are seen walking along the sidewalk toward the building entrance while several parked vehicles remain stationary around them. A cyclist rides across the open space between rows of cars moving from right to left before exiting the frame on the far side. Near the center-right section, another person walks past a cluster of white vans and trucks heading towards the edge of the lot. In the background adjacent to the buildings, people can be observed standing or sitting casually under trees beside the walkway. Throughout these moments, no significant changes occur within any specific areas except for minor movements like pedestrians passing through designated pathways or cyclists traversing empty spaces among parked automobiles. The overall environment remains calm with consistent lighting indicating daytime conditions throughout this period.

### VIRAT_S_010200_02_000349_000398

#### 8b-fp16

**Full-frame**

A person walks past the Student Services building on a sunny day, moving from right to left along the paved walkway while other individuals can be seen further down the path near parked cars and greenery.

**Motion ROI**

A person wearing a green shirt walks across the paved courtyard near a white newspaper box with advertisements on its side before exiting the frame. In the background, another individual can be seen walking along a grassy path adjacent to a fence line under bright daylight conditions. Meanwhile, further down the pathway toward the building entrance, someone else moves past while carrying something that appears dark-colored. The overall setting remains consistent throughout these movements within this campus environment featuring modern architecture and landscaped areas.

### VIRAT_S_010201_02_000167_000197

#### 8b-fp16

**Full-frame**

People walk along a paved pathway near a building labeled "STUDENT SERVICES," passing by trash bins, lampposts, and greenery under bright sunlight.

**Motion ROI**

Students walk along paved pathways near green lawns with flowering bushes under bright sunlight, while others move toward buildings labeled "Student Services" and "Chicano," indicating campus activity within a university setting.

### VIRAT_S_010203_09_001010_001036

#### 8b-fp16

**Full-frame**

A person walks along a paved pathway toward a building labeled "STUDENT SERVICES," passing near trash bins and lampposts while other individuals move about in the background.

**Motion ROI**

People walk along a paved pathway near a building labeled "Student Services," with some entering through glass doors while others continue walking away on the sidewalk adjacent to greenery and lampposts. A bird flies across the pavement close to the ground, moving steadily forward before disappearing out of frame.

### VIRAT_S_010204_03_000606_000632

#### 8b-fp16

**Full-frame**

A woman walks from right to left past a building labeled "STUDENT SERVICES," while another person exits the building's entrance nearby. In the background, several individuals walk along a pathway toward the building’s main entrance. A third individual approaches the building on the far side before turning away. Throughout the clip, people continue moving through the area around the student services center, including walking up stairs near an information kiosk marked “Go Thanks.” The setting remains consistent throughout, showing outdoor pathways adjacent to buildings under daylight conditions.

**Motion ROI**

People walk along a paved pathway near a building with "Student Services" signage, while others move across the foreground toward a set of stairs adjacent to a recycling bin and bulletin boards displaying posters. A woman walks past a circular manhole cover on the pavement before exiting the frame, leaving behind a quiet campus courtyard surrounded by greenery and lampposts under daylight conditions. In another section, individuals stroll down a sidewalk beside a brick wall leading towards glass doors; some pass through the entrance while others continue walking away, including two people jogging briskly alongside the path bordered by flowering bushes and trees.

### VIRAT_S_010204_10_001372_001395

#### 8b-fp16

**Full-frame**

A person walks along a paved pathway toward a building labeled "STUDENT SERVICES," passing by trash bins and lampposts under an overcast sky.

**Motion ROI**

A person walks along a paved pathway beside a grassy area with trees toward the entrance of a building labeled "Student Services." The view then shifts downward briefly before returning to focus on the main scene, which remains largely unchanged except for minor movements within the frame.

### VIRAT_S_010205_03_000370_000395

#### 8b-fp16

**Full-frame**

A person walks along a paved pathway toward a building labeled "STUDENT SERVICES," passing near a lamppost and moving past an entrance marked “CHICAGO” before exiting the frame on the left side. Another individual can be seen walking further down the path in the background, heading away from the main subject’s direction. A third figure briefly appears at a distance, also walking along the sidewalk but remains out of focus and moves slowly through the far end of the walkway. The surrounding area includes landscaped greenery, trash receptacles, informational signage, and architectural elements such as brick walls and staircases leading upward beside the building facade.

**Motion ROI**

Two individuals walk along a paved pathway near a brick building with "STUDENT SERVICES" signage, passing potted plants and moving toward the right side of the frame before exiting view. Another person descends a set of stairs adjacent to a red wall featuring directional signs and posters, then steps off-screen after reaching the bottom. A third individual walks away from the camera down a covered corridor on the left side of the scene, disappearing around a corner.

### VIRAT_S_010205_04_000545_000576

#### 8b-fp16

**Full-frame**

A person walks from right to left near the bottom edge of the frame while carrying bags, passing close to a trash can on the sidewalk outside a building labeled "STUDENT SERVICES." In the background, several individuals walk along a pathway leading toward the entrance of the building, moving at varying speeds and directions. A few people continue walking away from the camera down the path into the distance, maintaining their pace without stopping or interacting visibly. The overall setting remains consistent throughout, showing an outdoor campus environment under overcast lighting conditions.

**Motion ROI**

In front of a building with directional signage pointing toward campus centers, a person runs past a beige kiosk before exiting the frame. Meanwhile, on another part of the pathway near greenery and trees, several individuals walk along the sidewalk towards the entrance of a brick building.

### VIRAT_S_010207_02_000790_000816

#### 8b-fp16

**Full-frame**

A person walks away from the building entrance toward the right side of the frame while another individual approaches from the left carrying a backpack. Two people walk together along the sidewalk near the student services area before separating; one continues walking towards the stairs on the right while the other heads further down the path. A third person enters the frame from the bottom-right corner wearing shorts and sandals, moving diagonally upward toward the center-left portion of the screen until exiting out-of-frame at the top edge. The remaining individuals remain stationary throughout this period within their respective areas around campus buildings and pathways.

**Motion ROI**

Three individuals walk along a paved pathway near a building entrance with glass doors, moving away from the camera toward the right side of the frame before exiting view on the left. A person wearing shorts and carrying a backpack walks past a beige utility box located beside stairs leading up to a red wall marked with directional signage, heading towards the left edge of the frame.

### VIRAT_S_010207_09_001484_001510

#### 8b-fp16

**Full-frame**

A person walks along a paved pathway toward a building labeled "STUDENT SERVICES," while another individual ascends stairs adjacent to a wall featuring directional signage and posters.

**Motion ROI**

A person walks along a paved pathway near a grassy area with white flowers while another individual ascends a set of stairs adjacent to a brick building entrance marked "STUDENT SERVICES."

### VIRAT_S_010207_10_001549_001596

#### 8b-fp16

**Full-frame**

People walk along a paved pathway near a building labeled "STUDENT SERVICES," passing by various signs and trash receptacles while others stand nearby conversing. A woman walks down stairs adjacent to the building's entrance before continuing her path on the sidewalk.

**Motion ROI**

Students walk across the campus courtyard near the Student Services building while others stand conversing nearby under covered pathways. A woman walks past a white mailbox situated beside stairs leading up to a maroon wall with directional signage indicating various student centers including Asian Pacific American Programs and Native American Hillen.

### VIRAT_S_010208_09_000857_000886

#### 8b-fp16

**Full-frame**

A person walks toward the entrance of a building labeled "STUDENT SERVICES," enters through the glass doors under an awning marked “CHICANO,” then exits shortly after walking away from the building along a paved pathway. Another individual appears briefly on the right side of the frame near some stairs before moving out of view. A third person passes behind the first individual entering the building but remains partially obscured until exiting into the background. The surrounding area includes trash receptacles, potted plants, signage indicating directions for various services such as “CAREER CENTER” and “FACULTY OFFICE,” and a lamppost positioned centrally within the foreground throughout the entire duration of the footage.

**Motion ROI**

A person walks away from the entrance marked "CHICANO" toward the left side of the frame along a paved pathway near potted plants. Another individual exits through the same doorway carrying bags and proceeds down the walkway before turning right out of view. A third person wearing a red cap briefly appears walking past recycling bins on the lower-right corner of the scene.

### VIRAT_S_040000_00_000063_000085

#### 8b-fp16

**Full-frame**

A red car drives through an intersection near a parking lot surrounded by green grass, trees, and buildings under clear skies.

**Motion ROI**

In a parking lot surrounded by greenery and buildings, two red cars drive past parked vehicles near a landscaped median before exiting the frame. Meanwhile, above, tree branches sway gently against a clear blue sky.

### VIRAT_S_040000_01_000042_000099

#### 8b-fp16

**Full-frame**

A red car drives through an intersection near a parking lot surrounded by green grass, trees, and buildings under clear skies.

**Motion ROI**

A red convertible drives through a parking lot intersection near parked vehicles before exiting the frame. A person walks along a sidewalk adjacent to a low wall with foliage nearby. The top portion of tree branches moves slightly against a clear sky background. Another view shows similar movement of tree branches against the bright blue sky above.

### VIRAT_S_040000_02_000111_000145

#### 8b-fp16

**Full-frame**

A stationary aerial view captures a campus-like setting with paved roads, landscaped green areas, trees, buildings including a structure with satellite dishes on top, and parked cars remaining motionless throughout the duration of the footage.

**Motion ROI**

A person wearing red walks past a parked green car near a curved sidewalk corner before exiting the frame. A tree's top remains stationary against the clear sky throughout the footage.

### VIRAT_S_040000_04_000532_000622

#### 8b-fp16

**Full-frame**

A blue pickup truck remains parked on the left side of the frame while two individuals walk from right to left along a paved path near the parking area throughout the duration of the video.

**Motion ROI**

A person walks across a paved parking lot near parked vehicles including a blue pickup truck, moving toward the foreground while carrying a bag.

### VIRAT_S_040005_00_000003_000058

#### 8b-fp16

**Full-frame**

A blue car drives through an intersection within a campus-like setting surrounded by green lawns, trees, and buildings under clear skies.

**Motion ROI**

A blue pickup truck stops near parked cars, two individuals exit the vehicle, and then drive away through a landscaped roundabout while pedestrians walk nearby on sidewalks under clear skies with trees lining the streets.

### VIRAT_S_040005_08_001225_001276

#### 8b-fp16

**Full-frame**

A black car drives through an intersection near a parking lot surrounded by greenery and buildings under clear skies.

**Motion ROI**

A person walks toward a parked green car near other vehicles before stepping away as the car drives off through a landscaped roundabout. Meanwhile, shadows cast by nearby structures move across a grassy field bordered by fencing under clear skies. In another part of the view, trees sway gently against a backdrop of brick buildings beneath a bright blue sky.
