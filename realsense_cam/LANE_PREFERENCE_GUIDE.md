# Lane Change Preference System

## Overview

The obstacle avoidance system uses a **smart preference system** for lane changes, not hard requirements. This allows flexible overtaking while preferring the left lane (when driving on the right).

## Preference vs. Hard Requirement

### ❌ **Old Hard Requirement Approach**
```
If obstacle detected:
  Always go LEFT
  ↓
  LEFT blocked? → FAIL (stuck!)
```

### ✅ **New Smart Preference Approach**
```
If obstacle detected:
  1. Try LEFT (preferred) ✓
  2. If LEFT blocked, try RIGHT (fallback) ✓
  3. Consider obstacle position for smarter choice ✓
  4. If both blocked, STOP ✓
```

## Decision Logic Flow

```
┌─────────────────────────────────────┐
│ Obstacle Detected in Lane           │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ Check Adjacent Lanes                │
│ - Left lane: CLEAR/BLOCKED          │
│ - Right lane: CLEAR/BLOCKED         │
│ - Obstacle position: LEFT/RIGHT     │
└─────────────┬───────────────────────┘
              │
              ▼
       ┌──────┴──────┐
       │             │
       ▼             ▼
  LEFT CLEAR    LEFT BLOCKED
       │             │
       │             ▼
       │      ┌──────┴──────┐
       │      │             │
       │      ▼             ▼
       │  RIGHT CLEAR   BOTH BLOCKED
       │      │             │
       ▼      ▼             ▼
    GO LEFT  GO RIGHT    STOP
    (preferred) (fallback) (safe)
```

## Example Scenarios

### Scenario 1: Normal Case (Left Clear)
```
Setup:
  - Driving on RIGHT lane
  - Obstacle ahead in center
  - Left lane: CLEAR ✓
  - Right lane: doesn't matter

Decision:
  → GO LEFT (preferred lane)

Reason: "Overtaking LEFT - preferred lane clear"
```

### Scenario 2: Left Blocked, Right Clear
```
Setup:
  - Driving on RIGHT lane
  - Obstacle ahead in center
  - Left lane: BLOCKED (another car) ✗
  - Right lane: CLEAR ✓

Decision:
  → GO RIGHT (fallback)

Reason: "Overtaking RIGHT - preferred blocked but opposite clear"
```

### Scenario 3: Smart Choice Based on Obstacle Position
```
Setup:
  - Driving on RIGHT lane
  - Obstacle ahead on LEFT side of lane
  - Left lane: BLOCKED ✗
  - Right lane: CLEAR ✓

Decision:
  → GO RIGHT (makes more sense anyway)

Reason: "Overtaking RIGHT - obstacle on left, right lane clear"

Logic: Since obstacle is already on the left side,
       going right is more efficient even though
       left is our preferred direction normally.
```

### Scenario 4: Both Lanes Blocked
```
Setup:
  - Driving on RIGHT lane
  - Obstacle ahead
  - Left lane: BLOCKED (car) ✗
  - Right lane: BLOCKED (barrier) ✗

Decision:
  → STOP

Reason: "STOP - Both lanes blocked, cannot overtake safely"

Action:
  - Steering: 0.0
  - Throttle: 0.0
  - Wait until obstacle at distance > 0.6m
```

## Configuration

### Preferred Lane Setting
```python
# In obstacle_avoidance.py

PREFERRED_LANE = 'right'  # Your normal driving lane
```

**If you normally drive on RIGHT lane:**
- `PREFERRED_LANE = 'right'`
- Preferred overtaking direction: LEFT
- Fallback direction: RIGHT

**If you normally drive on LEFT lane:**
- `PREFERRED_LANE = 'left'`
- Preferred overtaking direction: RIGHT
- Fallback direction: LEFT

### Lane Detection Margins
```python
# Detection margin for adjacent lanes (pixels)
LEFT_MARGIN = 50   # Obstacle must be 50px left of LEFT_LANE_X to count as "in left lane"
RIGHT_MARGIN = 50  # Obstacle must be 50px right of RIGHT_LANE_X to count as "in right lane"
```

These margins prevent edge cases where obstacles barely touch the lane boundary from blocking overtaking.

### Detection Range
```python
# How far ahead to check for obstacles in adjacent lanes
ADJACENT_LANE_CHECK_DISTANCE = obstacle_distance + 1.5  # meters
```

Only obstacles within 1.5m ahead are considered when checking adjacent lanes.

## Console Output Example

```bash
[DECISION] Obstacle at 0.48m - Making decision...
  → Decision: LANE_CHANGE (large obstacle)
     Zone: center_third | Coverage: 45%
  🔍 Lane Safety Check:
     Preferred: LEFT lane
     Left lane: CLEAR ✓
     Right lane: BLOCKED (1 obstacles)
     Obstacle position: RIGHT side of lane
     Safety check: Overtaking LEFT - preferred lane clear
  📊 Overtaking plan (LEFT):
    - Lane width: 0.40m
    - Fixed passing distance: 2.00m
    - Phase 1 (move to LEFT lane): 2.7s
    - Phase 2 (pass straight): 8.0s
    - Phase 3 (return to original lane): 2.7s
    - Total time: 13.4s
[LANE_CHANGE] Starting timer-based overtaking maneuver
```

## Visual Indicators

### Status Panel
- **Direction Arrow**: Shows which way vehicle is changing lanes
  - `→ LEFT`: Going left (cyan color)
  - `RIGHT →`: Going right (magenta color)

### Action Colors
- `LANE_CHANGE_MOVING_OVER`: Red (changing lanes)
- `LANE_CHANGE_PASSING`: Orange (straight in new lane, LKAS active)
- `LANE_CHANGE_RETURNING`: Red (returning to original lane)

## Key Benefits

### 1. **Flexibility**
- Not stuck if preferred lane is blocked
- Adapts to real-world conditions

### 2. **Safety**
- Checks both lanes before deciding
- STOP fallback if no safe option

### 3. **Intelligence**
- Considers obstacle position
- Makes context-aware decisions

### 4. **Predictability**
- Clear preference order
- Consistent behavior

## Testing Scenarios

### Test 1: Verify Left Preference
```
Place obstacle in center of right lane
Ensure left lane is clear
Expected: Vehicle goes LEFT
```

### Test 2: Verify Right Fallback
```
Place obstacle in center of right lane
Place object in left adjacent lane (block it)
Expected: Vehicle goes RIGHT
```

### Test 3: Verify STOP Safety
```
Place obstacle in center of right lane
Block both left and right lanes
Expected: Vehicle STOPS
```

### Test 4: Smart Position Choice
```
Place small obstacle on left side of right lane
Block left lane
Ensure right lane clear
Expected: Vehicle goes RIGHT (makes more sense)
```

## Tuning Tips

### If vehicle always goes one direction:
- Check `PREFERRED_LANE` setting
- Verify camera view (is "left" actually left?)
- Check lane boundary values

### If vehicle doesn't detect adjacent obstacles:
- Reduce margins: `LEFT_MARGIN = 30`, `RIGHT_MARGIN = 30`
- Increase check distance: `obstacle_distance + 2.0`
- Verify YOLO detects obstacles in adjacent areas

### If vehicle stops too often:
- Margins might be too small (obstacles counted as blocking when they're not)
- Increase margins: `LEFT_MARGIN = 70`, `RIGHT_MARGIN = 70`
- Check if obstacles outside lanes are being detected

## Code Reference

Main function: `check_lane_change_safety()` in [obstacle_avoidance.py:177](obstacle_avoidance.py#L177)

Key variables:
- `preferred_direction`: -1 (left) or 1 (right)
- `left_lane_blocked`: Boolean
- `right_lane_blocked`: Boolean
- `obstacle_on_left_side`: Boolean for smart choice

Returns:
- `can_change`: True/False
- `direction`: -1 (left), +1 (right), 0 (stop)
- `reason`: Explanation string
