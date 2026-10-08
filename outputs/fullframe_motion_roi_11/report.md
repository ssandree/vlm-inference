# Full-frame vs Motion Multi-ROI

같은 generation setting으로 생성한 결과를 video별로 나란히 기록한다. 어느 condition이 더 좋은지는 판정하지 않는다.

## Setup

- Prompts:
  - Full-frame: `prompts/frames_full.txt`
  - Motion ROI: `prompts/frames_roi_per_tube.txt`
- Sampling: full-frame uniform 1.0 fps; ROI sequence는 저장된 Stage3 tube의 1 fps crop.
- Motion ROI packing: each tube is `[Global]` once at the ROI start, then `[ROI#]` and that ROI video. No anchor.
- Models: 8b, 30b

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
| VIRAT_S_010114_01_000235_000311 | 8b | Full-frame | 73 | 8880 | 67276800 | 18.879 | ok |
| VIRAT_S_010114_01_000235_000311 | 8b | Motion ROI | 68 | 8956 | 28364800 | 14.959 | ok |
| VIRAT_S_010207_04_000929_000954 | 8b | Full-frame | 17 | 2160 | 15667200 | 5.187 | ok |
| VIRAT_S_010207_04_000929_000954 | 8b | Motion ROI | 11 | 332 | 1085440 | 3.439 | ok |
| VIRAT_S_000200_00_000100_000171 | 8b | Full-frame | 72 | 8640 | 66355200 | 25.101 | ok |
| VIRAT_S_000200_00_000100_000171 | 8b | Motion ROI | 73 | 9468 | 22228992 | 19.871 | ok |
| VIRAT_S_010001_03_000537_000563 | 8b | Full-frame | 22 | 2640 | 20275200 | 9.807 | ok |
| VIRAT_S_010001_03_000537_000563 | 8b | Motion ROI | 52 | 3660 | 9117696 | 10.013 | ok |
| VIRAT_S_010003_03_000219_000259 | 8b | Full-frame | 41 | 5040 | 37785600 | 12.401 | ok |
| VIRAT_S_010003_03_000219_000259 | 8b | Motion ROI | 110 | 5860 | 17246208 | 17.116 | ok |
| VIRAT_S_010003_10_000901_000934 | 8b | Full-frame | 28 | 3360 | 25804800 | 12.014 | ok |
| VIRAT_S_010003_10_000901_000934 | 8b | Motion ROI | 58 | 4872 | 13828096 | 8.839 | ok |
| VIRAT_S_010003_06_000526_000560 | 8b | Full-frame | 30 | 3600 | 27648000 | 14.737 | ok |
| VIRAT_S_010003_06_000526_000560 | 8b | Motion ROI | 88 | 4832 | 11069440 | 11.588 | ok |
| VIRAT_S_010106_02_000656_000692 | 8b | Full-frame | 36 | 4320 | 33177600 | 14.637 | ok |
| VIRAT_S_010106_02_000656_000692 | 8b | Motion ROI | 73 | 4800 | 11411456 | 9.979 | ok |
| VIRAT_S_010113_05_000776_000805 | 8b | Full-frame | 24 | 2880 | 22118400 | 5.354 | ok |
| VIRAT_S_010113_05_000776_000805 | 8b | Motion ROI | 57 | 3956 | 10761216 | 13.868 | ok |
| VIRAT_S_040000_01_000042_000099 | 8b | Full-frame | 58 | 6960 | 120268800 | 6.085 | ok |
| VIRAT_S_040000_01_000042_000099 | 8b | Motion ROI | 87 | 5856 | 18206720 | 8.901 | ok |
| VIRAT_S_040005_08_001225_001276 | 8b | Full-frame | 52 | 6240 | 107827200 | 5.386 | ok |
| VIRAT_S_040005_08_001225_001276 | 8b | Motion ROI | 99 | 7300 | 19412480 | 8.831 | ok |
| VIRAT_S_010114_01_000235_000311 | 30b | Full-frame | 73 | 8880 | 67276800 | 9.065 | ok |
| VIRAT_S_010114_01_000235_000311 | 30b | Motion ROI | 68 | 8956 | 28364800 | 14.750 | ok |
| VIRAT_S_010207_04_000929_000954 | 30b | Full-frame | 17 | 2160 | 15667200 | 6.316 | ok |
| VIRAT_S_010207_04_000929_000954 | 30b | Motion ROI | 11 | 332 | 1085440 | 3.858 | ok |
| VIRAT_S_000200_00_000100_000171 | 30b | Full-frame | 72 | 8640 | 66355200 | 15.497 | ok |
| VIRAT_S_000200_00_000100_000171 | 30b | Motion ROI | 73 | 9468 | 22228992 | 13.596 | ok |
| VIRAT_S_010001_03_000537_000563 | 30b | Full-frame | 22 | 2640 | 20275200 | 4.838 | ok |
| VIRAT_S_010001_03_000537_000563 | 30b | Motion ROI | 52 | 3660 | 9117696 | 6.294 | ok |
| VIRAT_S_010003_03_000219_000259 | 30b | Full-frame | 41 | 5040 | 37785600 | 6.088 | ok |
| VIRAT_S_010003_03_000219_000259 | 30b | Motion ROI | 110 | 5860 | 17246208 | 7.159 | ok |
| VIRAT_S_010003_10_000901_000934 | 30b | Full-frame | 28 | 3360 | 25804800 | 4.957 | ok |
| VIRAT_S_010003_10_000901_000934 | 30b | Motion ROI | 58 | 4872 | 13828096 | 6.213 | ok |
| VIRAT_S_010003_06_000526_000560 | 30b | Full-frame | 30 | 3600 | 27648000 | 5.340 | ok |
| VIRAT_S_010003_06_000526_000560 | 30b | Motion ROI | 88 | 4832 | 11069440 | 6.415 | ok |
| VIRAT_S_010106_02_000656_000692 | 30b | Full-frame | 36 | 4320 | 33177600 | 9.564 | ok |
| VIRAT_S_010106_02_000656_000692 | 30b | Motion ROI | 73 | 4800 | 11411456 | 6.631 | ok |
| VIRAT_S_010113_05_000776_000805 | 30b | Full-frame | 24 | 2880 | 22118400 | 6.447 | ok |
| VIRAT_S_010113_05_000776_000805 | 30b | Motion ROI | 57 | 3956 | 10761216 | 6.045 | ok |
| VIRAT_S_040000_01_000042_000099 | 30b | Full-frame | 58 | 6960 | 120268800 | 7.192 | ok |
| VIRAT_S_040000_01_000042_000099 | 30b | Motion ROI | 87 | 5856 | 18206720 | 11.848 | ok |
| VIRAT_S_040005_08_001225_001276 | 30b | Full-frame | 52 | 6240 | 107827200 | 7.054 | ok |
| VIRAT_S_040005_08_001225_001276 | 30b | Motion ROI | 99 | 7300 | 19412480 | 11.853 | ok |

## Descriptions

### VIRAT_S_010114_01_000235_000311

#### 8b

**Full-frame**

A white sedan drives into the parking lot from the left side, parks near other vehicles, then exits on the right after maneuvering around parked cars. A person walks along the sidewalk adjacent to the building at the far end of the lot while another individual approaches a vehicle nearby before walking away. Two individuals walk together toward the center-right area of the parking lot, passing between rows of parked cars. Another pedestrian enters the frame from the bottom-left corner, moving diagonally upward through an open space within the parking lot towards the upper-middle section. The sun casts long shadows over parts of the asphalt surface throughout the entire duration of the footage.

**Motion ROI**

Two cyclists ride across the parking lot near the center, moving from right to left past parked cars. A silver sedan drives through the lower-left section of the lot, exiting toward the bottom edge. Another car enters from the upper-right side, driving along the perimeter road before turning away. Two pedestrians walk together on the far right sidewalk adjacent to the building’s wall. A white van pulls out slowly from behind other vehicles near the middle-right portion of the frame, then merges back into traffic.

#### 30b

**Full-frame**

A red car drives through the parking lot from left to right while several other cars remain parked throughout the clip; people walk along the sidewalk at the far end of the lot near the buildings.

**Motion ROI**

In the parking lot near a building with large windows and trees along the perimeter, several cars are parked while others move through the space; on the left side, vehicles enter and exit the frame, including a red car that drives across the lot, and further back, people walk along the sidewalk adjacent to the green field enclosed by a fence. On the right side, white vans remain stationary throughout, but two individuals appear walking together toward the bottom-right corner of the frame after briefly appearing near the center background. A person riding a bicycle moves horizontally across the middle section of the parking lot from left to right around mid-sequence.

### VIRAT_S_010207_04_000929_000954

#### 8b

**Full-frame**

A person walks along a paved pathway toward a building labeled "STUDENT SERVICES," passing under an awning marked “CHICANO” while other individuals remain stationary near trash receptacles and informational posters on adjacent walls.

**Motion ROI**

A person walks toward the entrance of a building with glass doors on the left side of the frame, passing under a green awning labeled "CHICAGO" before entering.

#### 30b

**Full-frame**

A person walks along a pathway on the left side of the frame near the "Student Services" building while another individual briefly appears at the entrance under the green awning labeled "Chicano." A white van drives past in the background from right to left behind the buildings. The rest of the scene remains largely unchanged throughout the clip.

**Motion ROI**

A person walks across the paved courtyard toward the entrance of the building marked "STUDENT SERVICES," passing under a covered walkway on the left side of the frame before entering through glass doors near the center-right portion of the image.

### VIRAT_S_000200_00_000100_000171

#### 8b

**Full-frame**

A silver sedan drives into view from the bottom right corner of the frame, stops at an intersection near a grassy area marked with signs, and remains stationary for several moments before beginning to move again toward the left side of the screen. A white pickup truck follows behind the sedan along the road, passing through the intersection without stopping. Shortly after, another vehicle appears on the far-right edge of the frame moving away from the viewer's perspective down the street. In parallel, two individuals emerge from within the parking lot’s vicinity; one walks towards the center while carrying something over their shoulder, then turns around and heads back out onto the sidewalk adjacent to the parked cars. The second person exits further ahead but soon returns closer to the first individual, walking alongside them briefly before continuing independently past the curb line leading off-screen. Throughout these movements, other vehicles remain either parked or traveling slowly throughout various parts of the expansive asphalt expanse surrounding buildings constructed primarily of brick material under cloudy skies overhead.

**Motion ROI**

A silver sedan drives onto the parking lot entrance ramp, turns left into a designated spot near a grassy median with signs, parks parallel to other spaces, and remains stationary while traffic continues on the adjacent road. A person exits the vehicle, walks around to the driver’s side door, opens it, retrieves their belongings including a backpack, closes the car door, and proceeds toward the sidewalk along the edge of the parking lot before walking away out of view. Another individual approaches the parked car carrying items, interacts briefly with someone inside, then steps back outside after placing something down beside them. The second person eventually leaves the vicinity entirely, leaving behind no further activity within this specific section of the parking facility.

#### 30b

**Full-frame**

A silver sedan drives into an empty parking lot from the right side of the frame at 0:09 and parks near the center-right edge; after remaining stationary for several seconds, two people exit the vehicle around 0:45 and walk toward the bottom-left corner of the screen before leaving the view. A black car remains parked on the left side of the lot throughout the clip. Another white car enters the frame from the top-right background at 0:18 but quickly exits out of sight. Several other vehicles pass through the road adjacent to the parking area along the lower portion of the image during various points between 0:00 and 0:20.

**Motion ROI**

A silver sedan enters the frame from the bottom right, turns left onto the road adjacent to the parking lot, crosses the intersection, and parks diagonally across two spaces near the entrance. A dark-colored car passes through the intersection behind the silver sedan. After parking, a person exits the driver's side of the parked sedan, walks along the curb away from the vehicle toward the lower part of the frame, then stops briefly before continuing out of view. Another white pickup truck drives past on the main road after the first car has stopped.

### VIRAT_S_010001_03_000537_000563

#### 8b

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual sits at an outdoor table under a red umbrella near the center foreground. In the background, several people move around the building entrance area, including individuals ascending and descending the staircase and walking along pathways adjacent to the grassy slope. The surrounding environment remains largely unchanged throughout the clip, featuring multiple tables shaded by umbrellas, palm trees, and distant mountains under clear skies.

**Motion ROI**

Underneath red umbrellas on a sunny plaza, a person sits alone at a table engrossed with their phone while others walk up and down nearby stairs leading to a modern building. In another part of the courtyard near brick planters, individuals stroll past outdoor tables under similar umbrellas adjacent to glass doors of a brick-walled structure. Above, clear blue skies stretch across distant mountains that frame the campus setting.

#### 30b

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual remains seated at a table under an umbrella in the foreground; people can be seen walking near the building entrance in the background throughout the clip.

**Motion ROI**

On the outdoor patio with red umbrellas, two people walk down the stairs on the left side of the frame while another person sits alone under an umbrella, looking at their phone; later, someone exits through glass doors near the building entrance on the right side of the courtyard.

### VIRAT_S_010003_03_000219_000259

#### 8b

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual sits at an outdoor table under a red umbrella near the bottom-left corner. In the background, people move around various areas including walking up steps and passing through doorways within the building complex situated behind a grassy lawn bordered by trees and more tables shaded by umbrellas. A few individuals can be seen seated at other tables scattered throughout the courtyard area, enjoying their surroundings amidst the sunny weather and distant mountainous backdrop.

**Motion ROI**

People walk up and down stairs on the left side of the courtyard while others stroll across the grassy lawn near outdoor tables with red umbrellas under bright sunlight. Two individuals sit together beneath a large umbrella at a table, engaged in conversation; later, they lean closer, appearing to share something personal before resuming their chat. A person walks past them carrying a white bag, moving toward the staircase leading away from the seating area. Another individual approaches from behind, walking along the paved path adjacent to the seated pair, eventually passing out of view towards the right edge of the frame. The overall atmosphere remains calm throughout these interactions within this open-air campus setting surrounded by buildings and distant mountains.

#### 30b

**Full-frame**

A person walks down the stairs on the left side of the courtyard while another individual ascends them; later, two people walk together from right to left near the building entrance at the back of the grassy area.

**Motion ROI**

In the courtyard, people walk across the grassy lawn near the building entrance while others sit under red umbrellas on the paved seating area; two individuals are seen sitting together at a table with an umbrella providing shade, engaging in conversation throughout the clip.

### VIRAT_S_010003_10_000901_000934

#### 8b

**Full-frame**

A person walks down stairs toward an outdoor seating area under red umbrellas while another individual ascends nearby steps leading into a building entrance. A woman enters from the left side of the frame walking past tables before heading towards the upper level near the main structure’s doorway. Several people move around within this open-air plaza space including someone descending further along the staircase path adjacent to greenery and palm trees lining the walkway beside buildings featuring large windows and modern architectural design elements against a backdrop of distant mountains partially covered by clouds on a bright day.

**Motion ROI**

People walk across the paved courtyard under red umbrellas while others ascend the stairs leading up to the building’s entrance, with mountains visible in the background. A person walks along the grassy pathway near the outdoor seating area adjacent to the brick building, passing by potted plants and closed umbrellas before exiting the frame.

#### 30b

**Full-frame**

People walk down the stairs on the left side of the courtyard while others move around the outdoor seating area under red umbrellas; some individuals enter and exit through doors at the building entrance in the background.

**Motion ROI**

People walk up and down a set of stairs near outdoor seating areas with red umbrellas on a sunny day; later, individuals are seen walking across a grassy lawn adjacent to another building section also featuring tables under red umbrellas.

### VIRAT_S_010003_06_000526_000560

#### 8b

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual sits at an outdoor table under a red umbrella near the bottom-left corner. A few people move around in the background near the building entrance and along pathways surrounding the grassy area, including someone walking from right to left past tables toward the center-right section of the courtyard. Another person enters the frame from the lower-left edge and moves towards the central seating area before exiting out of view. The overall setting remains consistent throughout, featuring multiple red umbrellas over black tables arranged across a paved patio adjacent to a green lawn bordered by trees and modern buildings against a backdrop of distant mountains under clear skies.

**Motion ROI**

In the foreground courtyard with red umbrellas, two individuals are seated at a table engaged in conversation while people walk past them on the paved ground near the stairs leading up to the building entrance. Meanwhile, another person walks across the patio adjacent to the glass doors under more umbrellas before exiting through the doorway. A third individual briefly appears walking along the edge of the grassy lawn beside large planters that line the pathway toward the main building’s entrance.

#### 30b

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual briefly appears at the bottom right corner; people remain seated under umbrellas throughout the outdoor courtyard area surrounded by buildings and mountains in the background.

**Motion ROI**

In the courtyard with red umbrellas and tables, two people sit under an umbrella on the left while another person walks down the stairs behind them; later, more individuals walk through various parts of the outdoor seating area near the building entrance.

### VIRAT_S_010106_02_000656_000692

#### 8b

**Full-frame**

Two individuals walk together through an empty parking lot toward the right side before separating; one continues walking while the other stops near parked vehicles. In the background, several people can be seen moving along sidewalks adjacent to buildings under bright sunlight casting long shadows over the asphalt surface dotted with various cars including sedans, SUVs, vans, and trucks. A white van drives past on a road behind the parking area, followed shortly after by another vehicle exiting from further down the street into view. The overall environment remains calm throughout without any significant changes except for minor movements within this urban setting featuring trees lining parts of the perimeter alongside multi-story structures nearby.

**Motion ROI**

A person walks across the parking lot near several parked vans while another individual rides a bicycle along the sidewalk adjacent to a fenced sports field. In the background, two cyclists ride past on a street lined with buildings and trees, passing close to a parked car. Meanwhile, further down the road, someone else cycles toward the right side of the frame alongside greenery and fencing.

#### 30b

**Full-frame**

Two people walk from right to left through the parking lot near the center-right area; another person walks into view on the far right side of the frame and stands still for several moments before walking out of sight. A blue car remains parked in the middle section of the lot throughout the clip while other vehicles remain stationary around the perimeter. The shadows cast by buildings and trees shift slightly over time indicating movement of the sun but no significant changes occur within the environment beyond these minor movements.

**Motion ROI**

In the parking lot near a building with trees, a person walks across the paved area while another individual rides a bicycle along a path adjacent to a fenced sports field; further down the road, two people are seen walking together on a sidewalk beside parked cars.

### VIRAT_S_010113_05_000776_000805

#### 8b

**Full-frame**

A person rides a bicycle from right to left along the edge of the parking lot near some parked vehicles while another individual walks briskly through the lower-right section of the frame before exiting out of view.

**Motion ROI**

In the parking lot near the fence line, two individuals walk side-by-side along the sidewalk before separating and walking away in opposite directions. Meanwhile, another person rides a bicycle across the open space toward the left edge of the frame while passing parked vehicles including a white van and a black sedan. In the background, people are seen moving around near the building entrance with some standing still and others walking past them. A cyclist enters the view on the right-hand side of the parking lot, riding through empty spaces between cars until they exit out of sight behind trees located towards the bottom-right corner of the scene.

#### 30b

**Full-frame**

A person rides a bicycle from right to left across the lower portion of the parking lot while another individual walks through the shadowed area near the center-right; several cars remain parked throughout the scene, including a white truck on the far right which remains stationary until briefly moving forward at 0:07 before stopping again.

**Motion ROI**

In the parking lot near a building with greenery on the left side, two people walk along the sidewalk adjacent to parked cars; later, another person rides a bicycle across the lower part of the frame while several vehicles remain stationary throughout the scenes.

### VIRAT_S_040000_01_000042_000099

#### 8b

**Full-frame**

A red car drives through an intersection near a parking lot surrounded by green grass, trees, and buildings under clear skies.

**Motion ROI**

A red convertible drives through a parking lot intersection near parked vehicles before exiting the frame. A person walks along a sidewalk adjacent to a low wall with trees nearby. The top portion of tree branches moves slightly against a clear sky. Another view shows similar movement of tree branches against the sky.

#### 30b

**Full-frame**

A red car drives through an intersection from left to right on a sunny day at a campus-like setting with buildings, trees, grassy areas, and parked cars; no other significant movements occur throughout the clip.

**Motion ROI**

In the parking lot on the left side of the frame, a red sedan enters the view from the bottom-left corner, drives across the paved area towards the right, makes a turn near a central landscaped island, continues driving past parked vehicles including a black SUV and green sedans, then exits the frame toward the top-right. In another part of the scene, a person walks along a sidewalk in the background behind some trees, moving from left to right before exiting the field of view. The sky remains clear throughout with tree tops occasionally appearing against it.

### VIRAT_S_040005_08_001225_001276

#### 8b

**Full-frame**

A black car drives through an intersection near a parking lot surrounded by greenery and buildings under clear skies.

**Motion ROI**

A person walks toward a parked green car near other vehicles before stepping away as the car drives off through a landscaped roundabout. Meanwhile, shadows cast on grassy terrain indicate movement nearby while trees sway gently against a clear sky above buildings.

#### 30b

**Full-frame**

A black car drives from left to right through an intersection on a paved road within a campus-like setting featuring green lawns, trees, streetlights, and buildings under clear skies; other cars remain parked along the roadside throughout the clip.

**Motion ROI**

In the parking lot on the left side of the frame, a person wearing red walks past parked cars before getting into a dark green sedan; the car then drives out of the parking space, makes a right turn onto the road, and continues driving away down the street. On the grassy lawn near the fence line on the lower right portion of the image, shadows are cast that shift slightly due to movement outside the view. In the background above the trees, birds can be seen flying across the sky.
