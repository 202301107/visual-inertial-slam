# Room and ground-truth measurements

All distances are measured with a tape measure. Marker positions are in `markers.csv` (metres).

## Coordinate frame (the "world" frame)

- **Origin (0, 0, 0):** the floor-level corner where the S wall and the E wall meet.
- **X axis:** along the E wall, pointing from the S wall toward the N wall.
- **Y axis:** along the S wall, pointing from the E wall toward the W wall.
- **Z axis:** straight up from the floor.

Wall planes (all other measurements follow from these):

| Wall | Plane | Marker faces |
|------|-------|--------------|
| S | X = 0 | +X |
| N | X = 3.50 m | -X |
| E | Y = 0 | +Y |
| W | Y = 4.40 m | -Y |
| Floor | Z = 0 | +Z (up) |

Room size: about 3.50 m (X) by 4.40 m (Y), taken from the wall marker positions.

## Markers

- Dictionary: DICT_4X4_50, IDs 0-9.
- Printed black-square side: 15 cm for every marker (`markerLength` = 0.15 m).
- Positions are the marker centre, measured from the origin.
- Markers 7, 8 and 9 are flat on the floor. They are left OUT of the map and used only as scoring points. All other markers (0-6) are in the map.

## Start tape (used for every run)

- Position: X = 2.9 m, Y = 3.25 m  (fill in after taping it down)

## Run 1 path (ground truth)

- Corner points of the taped path, in order, as (X, Y) in m: X=0.5 m, Y= 3.25 m
- Total walked length: 5.4 m
- End point: X = 0.5 m, Y = 0.25 m

## Run 5 hard condition

- Chosen condition: dim lights
