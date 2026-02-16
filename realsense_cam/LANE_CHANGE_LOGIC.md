# Lane Change Logic Documentation

## Overview

The obstacle avoidance system implements intelligent lane change logic designed for a **two-lane road** where the vehicle normally drives on the **right lane**. This document explains how the system decides when and where to change lanes.

## Key Questions Answered

### 1. How does the vehicle decide which lane to change to?

The system uses a **priority-based decision algorithm**:

#### Priority System:
1. **Preferred Lane (LEFT)**: Since we normally drive on the right lane, the left lane is typically empty
   - First choice: Change to left lane if clear

2. **Opposite Lane (RIGHT)**: If left lane is blocked
   - Second choice: Change to right lane if available

3. **STOP**: If both lanes are blocked
   - Cannot overtake safely → Stop and wait

#### Safety Checks:
The system scans for obstacles in adjacent lanes:
```python
def check_lane_change_safety(self, all_boxes, all_distances, obstacle_box, obstacle_distance):
    # Checks:
    # 1. Is left lane clear? (preferred)
    # 2. Is right lane clear? (fallback)
    # 3. Are both blocked? (stop)

    return (can_change, direction, reason)
    # direction: -1 = LEFT, +1 = RIGHT, 0 = STOP
```

### 2. How far does the vehicle travel during lane change?

Since we only have **one depth camera** and cannot detect when we've passed the obstacle, the system uses a **FIXED distance** approach:

#### Fixed Distance Strategy:
```python
LANE_CHANGE_PASS_DISTANCE = 2.0  # meters (configurable)
```

#### Three-Phase Maneuver:

**Phase 1: MOVING_OVER** (Change to adjacent lane)
- Duration: `lane_width / lateral_speed`
- Example: 0.4m lane / 0.15 m/s = ~2.7 seconds
- LKAS: **DISABLED** (avoidance controls steering)

**Phase 2: PASSING** (Go straight in new lane)
- Duration: `PASS_DISTANCE / vehicle_speed`
- Example: 2.0m / 0.25 m/s = 8.0 seconds
- LKAS: **ENABLED** (keeps vehicle centered in new lane)

**Phase 3: RETURNING** (Return to original lane)
- Duration: Same as Phase 1 (~2.7 seconds)
- LKAS: **DISABLED** (avoidance controls steering back)

**Total Time**: ~13-14 seconds for complete maneuver

### 3. When does LKAS turn on/off during lane change?

LKAS is **dynamically controlled** based on the maneuver phase:

#### LKAS Control States:

| Phase | LKAS State | Reason |
|-------|-----------|---------|
| Phase 1: MOVING_OVER | **OFF** | Avoidance system controls steering to change lanes |
| Phase 2: PASSING | **ON** | LKAS keeps vehicle centered **in new lane** |
| Phase 3: RETURNING | **OFF** | Avoidance system controls steering back to original lane |
| After Return | **ON** | LKAS keeps vehicle centered **in original lane** |

#### Why This Design?

**Phase 2 (Passing)** is critical:
- Vehicle is now in the **new lane** (left lane if we were on right)
- LKAS detects the **new lane boundaries** and centers the vehicle
- This keeps the vehicle stable while passing the obstacle
- We travel a **fixed 2.0m** in this lane

**Phase 3 (Returning)** ensures we go back:
- LKAS is turned **OFF** again
- Avoidance system steers **opposite direction** from Phase 1
- This brings us back to the **original right lane**
- LKAS turns **ON** when back in original lane

### 4. What if LKAS stays on in the new lane?

The system is designed to **prevent this problem**:

```python
# Phase 3: RETURNING ensures we return to original lane
return_steering = -phase1_steering  # Opposite direction!
enable_lkas = False  # Disable LKAS during return
```

The timer-based state machine **guarantees** return to original lane:
1. After Phase 2 timer expires → automatically enter Phase 3
2. Phase 3 applies **opposite steering** to return
3. Phase 3 timer ensures we complete the return
4. Only then does LKAS re-engage in original lane

## Configuration Parameters

### Distance & Timing
```python
# In obstacle_avoidance.py

LANE_CHANGE_PASS_DISTANCE = 2.0  # How far to travel in new lane (meters)
VEHICLE_SPEED = 0.25             # Vehicle speed at throttle 0.5 (m/s)
LATERAL_SPEED = 0.15             # Lateral movement speed (m/s)
```

### Lane Preference
```python
PREFERRED_LANE = 'right'  # Which lane we normally drive in
# When on right → prefer left lane for overtaking
# When on left → prefer right lane for overtaking
```

### Safety Margins
```python
LANE_CHANGE_MIN_CLEARANCE = 0.3  # Minimum clearance from obstacle (meters)
```

## Example Scenarios

### Scenario 1: Clear Left Lane (Normal Case)
```
Initial: Vehicle on RIGHT lane, obstacle ahead
Check: Left lane is clear ✓
Decision: Change to LEFT lane
Execute:
  Phase 1 (2.7s): Steer LEFT, LKAS OFF
  Phase 2 (8.0s): Go straight, LKAS ON (centered in LEFT lane)
  Phase 3 (2.7s): Steer RIGHT, LKAS OFF
  Result: Back in RIGHT lane, LKAS ON
```

### Scenario 2: Left Lane Blocked, Right Clear
```
Initial: Vehicle on RIGHT lane, obstacle ahead
Check: Left lane blocked ✗, Right lane clear ✓
Decision: Change to RIGHT lane (shoulder/emergency lane)
Execute:
  Phase 1 (2.7s): Steer RIGHT, LKAS OFF
  Phase 2 (8.0s): Go straight, LKAS ON (centered in RIGHT shoulder)
  Phase 3 (2.7s): Steer LEFT, LKAS OFF
  Result: Back in original RIGHT lane, LKAS ON
```

### Scenario 3: Both Lanes Blocked
```
Initial: Vehicle on RIGHT lane, obstacle ahead
Check: Left lane blocked ✗, Right lane blocked ✗
Decision: STOP - Cannot overtake safely
Execute:
  Steering: 0.0
  Throttle: 0.0
  LKAS: OFF
  Wait: Until obstacle clears (distance > 0.6m)
  Resume: Normal operation
```

### Scenario 4: Small Obstacle - Micro Adjust
```
Initial: Small obstacle in LEFT third of lane (coverage < 33%)
Decision: MICRO_ADJUST (stay in lane)
Execute:
  Steering: +0.15 (slight right bias)
  Throttle: 0.5 (continue)
  LKAS: OFF (avoidance controls steering)
  Behavior: Nudge slightly right within lane
```

## Tuning Guide

### If vehicle doesn't pass obstacle completely:
```python
# Increase passing distance
LANE_CHANGE_PASS_DISTANCE = 2.5  # or 3.0 meters
```

### If lane change is too aggressive:
```python
# Reduce steering magnitude
LANE_CHANGE_STEERING = 0.5  # default is 0.6
```

### If returning to lane is inaccurate:
```python
# Calibrate vehicle/lateral speeds based on actual measurements
VEHICLE_SPEED = 0.30  # Measure actual speed
LATERAL_SPEED = 0.18  # Measure actual lateral movement
```

### If wrong lane is chosen:
```python
# Change preferred lane
PREFERRED_LANE = 'left'  # if you normally drive on left
```

## Debugging Tips

### Check Lane Change Direction:
Look for console output:
```
  → Decision: LANE_CHANGE (large obstacle)
     Safety check: Lane change LEFT - preferred lane is clear
  📊 Overtaking plan (LEFT):
    - Phase 1 (move to LEFT lane): 2.7s
    - Phase 2 (pass straight): 8.0s
    - Phase 3 (return to original lane): 2.7s
```

### Monitor LKAS State:
```
  ✓ Phase 1 complete → Phase 2: Passing in new lane
  ℹ️  LKAS can now engage in new lane

  ✓ Phase 2 complete → Phase 3: Returning to original lane
  ℹ️  LKAS will be disabled during lane return

  ✓ Phase 3 complete → Back in original lane!
  ℹ️  LKAS re-enabled in original lane
```

### Visual Indicators:
- **Status Panel Color**:
  - `LANE_CHANGE_MOVING_OVER`: Red (LKAS off)
  - `LANE_CHANGE_PASSING`: Orange (LKAS on)
  - `LANE_CHANGE_RETURNING`: Red (LKAS off)
  - `NORMAL`: Green (LKAS on)

## Safety Features

1. **Hysteresis**: Prevents rapid decision changes
   - Enter threshold: 0.5m
   - Exit threshold: 0.6m

2. **Adjacent Lane Checking**: Scans for obstacles in target lane

3. **STOP Fallback**: If no safe lane available, vehicle stops

4. **Fixed Distance**: Predictable behavior, no reliance on sensor detection

5. **Automatic Return**: Timer ensures vehicle returns to original lane

6. **LKAS Phase Control**: Prevents staying in wrong lane

## Limitations & Future Improvements

### Current Limitations:
1. **Fixed distance**: May over/undershoot depending on obstacle size
2. **Single camera**: Cannot confirm when obstacle is cleared
3. **No moving obstacles**: Designed for stationary obstacles only
4. **Two-lane assumption**: Not designed for multi-lane highways

### Potential Improvements:
1. **Adaptive distance**: Calculate based on obstacle size from bounding box
2. **Rear camera**: Detect when obstacle is behind vehicle
3. **Side cameras**: Better adjacent lane monitoring
4. **Moving obstacle tracking**: Adjust maneuver based on obstacle velocity
5. **GPS/IMU**: More accurate return to original lane
6. **Multi-lane support**: Handle 3+ lane roads

## Code References

- **Lane change decision**: [obstacle_avoidance.py:330](obstacle_avoidance.py#L330)
- **Safety check**: [obstacle_avoidance.py:177](obstacle_avoidance.py#L177)
- **Timer calculation**: [obstacle_avoidance.py:254](obstacle_avoidance.py#L254)
- **State machine**: [obstacle_avoidance.py:396](obstacle_avoidance.py#L396)
- **LKAS integration**: [yolo_depth_avoidance.py:213](yolo_depth_avoidance.py#L213)
