# Full-frame vs Motion Multi-ROI

같은 generation setting으로 생성한 결과를 video별로 나란히 기록한다. 어느 condition이 더 좋은지는 판정하지 않는다.

## Setup

- Prompts:
  - Full-frame: `prompts/frames_full.txt`
  - Motion Multi-ROI: `prompts/frames_roi_per_tube.txt`
- Sampling: full-frame uniform 1.0 fps; Motion Multi-ROI는 저장된 Stage3 tube의 1 fps crop과 start-only Global.
- Motion packing: sparse Global images, then one video per ROI tube.
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
| VIRAT_S_010001_03_000537_000563 | 8b | Full-frame | 22 | 2640 | 20275200 | 4.936 | ok |
| VIRAT_S_010001_03_000537_000563 | 8b | Motion Multi-ROI | 50 | 3156 | 7274496 | 9.732 | ok |
| VIRAT_S_010003_03_000219_000259 | 8b | Full-frame | 41 | 5040 | 37785600 | 8.932 | ok |
| VIRAT_S_010003_03_000219_000259 | 8b | Motion Multi-ROI | 108 | 5356 | 15403008 | 21.874 | ok |
| VIRAT_S_010003_10_000901_000934 | 8b | Full-frame | 28 | 3360 | 25804800 | 5.366 | ok |
| VIRAT_S_010003_10_000901_000934 | 8b | Motion Multi-ROI | 57 | 4620 | 12906496 | 10.004 | ok |
| VIRAT_S_010003_06_000526_000560 | 8b | Full-frame | 30 | 3600 | 27648000 | 12.542 | ok |
| VIRAT_S_010003_06_000526_000560 | 8b | Motion Multi-ROI | 87 | 4580 | 10147840 | 12.592 | ok |
| VIRAT_S_010106_02_000656_000692 | 8b | Full-frame | 36 | 4320 | 33177600 | 4.915 | ok |
| VIRAT_S_010106_02_000656_000692 | 8b | Motion Multi-ROI | 72 | 4548 | 10489856 | 11.643 | ok |
| VIRAT_S_010113_05_000776_000805 | 8b | Full-frame | 24 | 2880 | 22118400 | 9.843 | ok |
| VIRAT_S_010113_05_000776_000805 | 8b | Motion Multi-ROI | 57 | 3956 | 10761216 | 13.803 | ok |
| VIRAT_S_040000_01_000042_000099 | 8b | Full-frame | 58 | 6960 | 120268800 | 5.743 | ok |
| VIRAT_S_040000_01_000042_000099 | 8b | Motion Multi-ROI | 87 | 5856 | 18206720 | 12.607 | ok |
| VIRAT_S_040005_08_001225_001276 | 8b | Full-frame | 52 | 6240 | 107827200 | 5.330 | ok |
| VIRAT_S_040005_08_001225_001276 | 8b | Motion Multi-ROI | 98 | 7048 | 17338880 | 9.673 | ok |
| VIRAT_S_010001_03_000537_000563 | 30b | Full-frame | 22 | 2640 | 20275200 | 6.460 | ok |
| VIRAT_S_010001_03_000537_000563 | 30b | Motion Multi-ROI | 50 | 3156 | 7274496 | 7.680 | ok |
| VIRAT_S_010003_03_000219_000259 | 30b | Full-frame | 41 | 5040 | 37785600 | 8.747 | ok |
| VIRAT_S_010003_03_000219_000259 | 30b | Motion Multi-ROI | 108 | 5356 | 15403008 | 6.524 | ok |
| VIRAT_S_010003_10_000901_000934 | 30b | Full-frame | 28 | 3360 | 25804800 | 5.178 | ok |
| VIRAT_S_010003_10_000901_000934 | 30b | Motion Multi-ROI | 57 | 4620 | 12906496 | 6.178 | ok |
| VIRAT_S_010003_06_000526_000560 | 30b | Full-frame | 30 | 3600 | 27648000 | 5.062 | ok |
| VIRAT_S_010003_06_000526_000560 | 30b | Motion Multi-ROI | 87 | 4580 | 10147840 | 11.655 | ok |
| VIRAT_S_010106_02_000656_000692 | 30b | Full-frame | 36 | 4320 | 33177600 | 9.533 | ok |
| VIRAT_S_010106_02_000656_000692 | 30b | Motion Multi-ROI | 72 | 4548 | 10489856 | 6.250 | ok |
| VIRAT_S_010113_05_000776_000805 | 30b | Full-frame | 24 | 2880 | 22118400 | 6.506 | ok |
| VIRAT_S_010113_05_000776_000805 | 30b | Motion Multi-ROI | 57 | 3956 | 10761216 | 5.565 | ok |
| VIRAT_S_040000_01_000042_000099 | 30b | Full-frame | 58 | 6960 | 120268800 | 7.062 | ok |
| VIRAT_S_040000_01_000042_000099 | 30b | Motion Multi-ROI | 87 | 5856 | 18206720 | 9.310 | ok |
| VIRAT_S_040005_08_001225_001276 | 30b | Full-frame | 52 | 6240 | 107827200 | 6.972 | ok |
| VIRAT_S_040005_08_001225_001276 | 30b | Motion Multi-ROI | 98 | 7048 | 17338880 | 10.787 | ok |

## Descriptions

### VIRAT_S_010001_03_000537_000563

#### 8b

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual sits alone under an umbrella near the center foreground throughout the sequence.

**Motion Multi-ROI**

Underneath red umbrellas on a sunlit plaza, a person sits alone at a table engrossed with their phone while students ascend wide stone stairs behind them. A woman walks briskly past outdoor tables near a brick building’s entrance before exiting frame. The view shifts upward briefly toward clear blue sky above distant mountains, then pans back down across another section of patio seating adjacent to green lawns bordered by planters.

#### 30b

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual remains seated under an umbrella in the foreground; people can be seen moving near the entrance of the building in the background throughout the sequence.

**Motion Multi-ROI**

On a sunny day on a college campus courtyard with red umbrellas and tables, two people walk down a set of stairs near a building entrance while another person sits alone under an umbrella looking at their phone; later, a man walks out of a nearby brick building onto a patio area before walking away across the grassy lawn towards the right side of the frame.

### VIRAT_S_010003_03_000219_000259

#### 8b

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual sits under an umbrella near the bottom-left corner. In the background, people move around various areas including walking up steps and standing near tables under umbrellas. A few individuals walk along pathways adjacent to buildings and grassy sections throughout the courtyard.

**Motion Multi-ROI**

People walk up and down the stairs on the left side of the courtyard while others stroll across the grassy lawn near the building entrance under red umbrellas. Two individuals sit together beneath a large orange umbrella, engaged in conversation, with their table positioned close to the staircase railing. A person walks past them carrying a white bag before moving out of frame. Another individual approaches the seated pair briefly before walking away toward the right edge of the patio. The two people remain seated throughout, occasionally shifting positions slightly but staying focused on their interaction under the shade provided by the umbrella. In another part of the seating area, someone stands nearby observing the surroundings before turning around and exiting towards the far-right corner of the paved space. Throughout these moments, various other patrons can be seen sitting or standing further back along the upper level of the outdoor dining zone, enjoying the sunny day amidst the greenery and architectural features like palm trees and low brick planters lining the pathways.

#### 30b

**Full-frame**

A person walks down the stairs on the left side of the courtyard while another individual ascends them; later, two people walk together from right to left along the upper level near the building entrance. A few individuals remain seated under umbrellas throughout the sequence, occasionally shifting slightly but mostly staying still. The overall environment remains calm with no significant changes apart from these minor movements.

**Motion Multi-ROI**

In the courtyard, people walk across the grassy lawn and down the stairs on the left side while others sit under red umbrellas at tables; two individuals seated together briefly stand up before sitting back down again.

### VIRAT_S_010003_10_000901_000934

#### 8b

**Full-frame**

People walk up and down stairs leading into a building while others stroll through an outdoor plaza furnished with tables under red umbrellas near a grassy area surrounded by trees and mountains in the background.

**Motion Multi-ROI**

People walk across a paved courtyard with red umbrellas and tables, ascending wide stairs toward a building entrance while others stroll past nearby planters under bright sunlight. A person walks along a grassy path beside a row of outdoor seating areas near large windows on a brick building facade. Two individuals approach and enter through glass doors adjacent to closed patio umbrellas situated within landscaped planter beds.

#### 30b

**Full-frame**

People walk down the stairs on the left side of the courtyard while others move around the area under red umbrellas; some individuals enter and exit the building through the main entrance near the top of the frame.

**Motion Multi-ROI**

In the courtyard with red umbrellas and tables, several people walk up a set of stairs leading to a building entrance; later, individuals move across a grassy lawn near another section of outdoor seating under similar umbrellas.

### VIRAT_S_010003_06_000526_000560

#### 8b

**Full-frame**

A person walks down the stairs on the left side of the frame while another individual sits under an umbrella near the bottom-left corner. A few people move around in the background near the building entrance and along the right edge of the courtyard. One person exits from behind some tables toward the lower-right area. Another person enters from the far right and moves towards the center before exiting out of view. The overall setting remains calm throughout, featuring multiple red umbrellas over tables arranged across a paved patio adjacent to a grassy lawn bordered by buildings and mountains in the distance.

**Motion Multi-ROI**

In the foreground courtyard with red umbrellas, two people sit together at a table while others walk past on the stairs behind them. A person walks down the wide staircase leading away from the building's entrance. Another individual strolls across the paved plaza near the tables before disappearing out of frame. In another section, someone walks along the patio adjacent to the glass doors of the brick building, passing between the outdoor seating areas. Finally, a figure briefly appears walking toward the right side of the screen near the steps.

#### 30b

**Full-frame**

A person walks down the stairs on the left side of the courtyard while another individual briefly appears near the bottom right before moving out of view; people remain seated under umbrellas throughout the sequence.

**Motion Multi-ROI**

In the courtyard with red umbrellas and tables, two people sit under an umbrella while another person walks down the stairs on the left side of the frame; later, this individual exits the view and then reappears walking across the lower right portion of the screen before exiting again. In front of the building's entrance, several individuals walk past the outdoor seating area, including someone entering through the doors around the middle of the clip and others passing along the sidewalk outside. A final brief shot shows a dark object moving quickly near the bottom edge of the frame against a light-colored background.

### VIRAT_S_010106_02_000656_000692

#### 8b

**Full-frame**

Two individuals walk together through an empty parking lot toward the right side before separating; one continues walking while the other stops near parked vehicles.

**Motion Multi-ROI**

In a large parking lot adjacent to buildings and greenery, a person walks across the asphalt near parked vehicles including vans and trucks before stopping to look up toward the sky while standing alone amidst several cars. Simultaneously, another individual rides a bicycle along a path beside a fenced sports field with grassy areas beyond. Additionally, two people on bicycles ride past a street corner marked by traffic signs and poles, moving away from the viewer’s perspective under clear daylight conditions.

#### 30b

**Full-frame**

Two people walk from right to left through the parking lot near the center-right area; another person walks into view from the far right edge of the frame and moves toward the center before stopping. A blue car remains parked on the middle-left side throughout the clip while other cars remain stationary or move slowly within their designated spaces. The shadows cast by buildings and trees shift slightly over time indicating movement of the sun but no significant changes occur in vehicle positions beyond minor adjustments.

**Motion Multi-ROI**

In the parking lot, a person walks across the frame while another individual stands still near parked vehicles; later, two people walk along a path adjacent to a sports field, followed by several cyclists riding on a road nearby.

### VIRAT_S_010113_05_000776_000805

#### 8b

**Full-frame**

A person rides a bicycle from right to left along the edge of the parking lot near some parked vehicles. Another individual walks briskly through the lower-right section of the frame, moving toward the center before exiting view. A third pedestrian enters from behind trees on the far right side, walking diagonally towards the middle area while carrying an object. In the background, several people can be seen strolling around outside buildings adjacent to the parking lot.

**Motion Multi-ROI**

In the parking lot near the building with light-colored walls, two individuals walk along the sidewalk adjacent to parked vehicles while another person rides a bicycle across the open space between cars. A cyclist enters the frame riding toward the bottom left corner before exiting out of view beneath the shadow cast by nearby trees. Meanwhile, further back on the right side of the lot, three people stand together beside a white van and truck; after some brief interaction, they disperse—one walks away towards the center-left direction—while others remain stationary near their vehicle. The overall setting remains calm throughout these movements under bright daylight conditions.

#### 30b

**Full-frame**

A person rides a bicycle from right to left across the lower portion of the parking lot while another individual walks through the shadowed area near the center-right; several cars remain parked throughout the sequence, including a white truck on the far right which remains stationary until briefly moving forward around frame 7 before stopping again.

**Motion Multi-ROI**

In the parking lot near a building with greenery, several cars remain parked while people walk along the sidewalk adjacent to the lot; later, two individuals ride bicycles across the paved area before exiting the frame.

### VIRAT_S_040000_01_000042_000099

#### 8b

**Full-frame**

A red car drives through an intersection near a parking lot while other vehicles remain stationary nearby under clear skies.

**Motion Multi-ROI**

In a parking lot surrounded by greenery and buildings, a red convertible drives past parked cars before stopping near another stationary red car. A person exits the second red vehicle and walks toward the first while carrying something small. The individual then approaches the open driver's side door of the convertible and enters the passenger seat. After settling inside, they close the door and remain seated briefly before driving away down the road. Another red sedan passes through the intersection nearby after completing this sequence.

#### 30b

**Full-frame**

A red car drives from left to right through an intersection on the far side of the frame, passing behind trees before exiting view; another red vehicle remains stationary near parked cars throughout the sequence.

**Motion Multi-ROI**

In the parking lot on the left side of the frame, a red convertible car drives around a circular island while another dark-colored sedan enters the frame from the bottom left and moves toward the center before stopping near the first vehicle; after this interaction, both cars remain stationary. In the background, above the trees, two birds fly across the sky from right to left.

### VIRAT_S_040005_08_001225_001276

#### 8b

**Full-frame**

A black car drives through an intersection near a parking lot surrounded by greenery and buildings under clear skies.

**Motion Multi-ROI**

A person walks toward a parked green car near other vehicles before stepping back away. The green sedan then drives out of the parking lot onto a road, making a turn while moving through the intersection. After completing the maneuver, the vehicle continues driving along the street, eventually exiting the frame entirely.

#### 30b

**Full-frame**

A black car drives from left to right through an intersection on a paved road near a grassy area; another dark-colored vehicle enters the frame from the bottom-left corner and moves toward the center before exiting the view.

**Motion Multi-ROI**

In the parking lot on the left side of the frame, a person wearing red walks past parked cars before getting into a dark green sedan; the car then drives out of the parking space, makes a right turn onto the road, and proceeds down the street away from the camera. The view briefly shifts to focus on a grassy lawn with shadows cast across it, followed by another shift showing tree tops against a clear blue sky above buildings.
