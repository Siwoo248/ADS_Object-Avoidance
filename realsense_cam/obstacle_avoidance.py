"""
Obstacle Avoidance System for JetRacer
Integrates with YOLO depth detection to make avoidance decisions.
Uses zone-based classification and distance thresholds.
"""

import numpy as np
import cv2
import time
from collections import deque


class ObstacleAvoidanceSystem:
    """
    Obstacle avoidance decision-making system.
    Makes decisions based on obstacle position, size, and distance.
    """
    
    def __init__(self, frame_width=640, frame_height=480):
        """
        Initialize the avoidance system.
        
        Args:
            frame_width (int): Camera frame width in pixels
            frame_height (int): Camera frame height in pixels
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # ========== CALIBRATION PARAMETERS ==========
        # TODO: Calibrate these by driving and observing lane positions in pixels
        self.LEFT_LANE_X = 260      # Left lane boundary (pixels)
        self.RIGHT_LANE_X = 490     # Right lane boundary (pixels)
        self.LANE_WIDTH_PIXELS = self.RIGHT_LANE_X - self.LEFT_LANE_X

        # Distance thresholds (meters)
        self.CRITICAL_DISTANCE = 0.5    # Make decision at this distance

        # Hysteresis thresholds to prevent oscillation
        self.DECISION_ENTER = 0.5       # Trigger decision
        self.DECISION_EXIT = 0.6        # Clear decision (must be > DECISION_ENTER)

        # Coverage thresholds (what fraction of lane the obstacle blocks)
        self.MICRO_ADJUST_THRESHOLD = 0.33   # < 33% of lane in A or C → micro adjust
        # >= 33% OR in B zone → lane change or stop

        # Steering parameters
        self.MICRO_ADJUST_STEERING = 0.15    # Small steering bias (stay in lane)
        self.LANE_CHANGE_STEERING = 0.6      # Full lane change maneuver

        # Throttle parameters - FIXED at 0.5
        self.THROTTLE = 0.5  # Fixed throttle for all situations

        # Vehicle speed calibration (measured experimentally)
        self.VEHICLE_SPEED = 0.25  # m/s at throttle 0.5 (adjust after testing!)
        self.LATERAL_SPEED = 0.15  # m/s at steering 0.5

        # Lane preference for two-lane road (vehicle normally on right)
        self.PREFERRED_LANE = 'right'  # 'right' or 'left'
        # When on right lane, left lane is typically empty
        # So we prefer to change to left lane when avoiding obstacles

        # Fixed distance to travel during lane change (since we can't detect when past obstacle)
        self.LANE_CHANGE_PASS_DISTANCE = 0.5  # meters to travel forward while passing
        self.LANE_CHANGE_MIN_CLEARANCE = 0.3   # minimum clearance from obstacle to attempt lane change
        
        # ========== STATE VARIABLES ==========
        self.decision_made = False
        self.current_action = 'NORMAL'  # Start with NORMAL action instead of None
        self.current_steering = 0.0
        self.current_throttle = self.THROTTLE
        self.decision_timestamp = 0
        self.obstacle_zone = None
        self.obstacle_coverage = 0
        
        # Timer-based overtaking state machine
        self.overtaking_state = 'NORMAL'  # NORMAL, MOVING_OVER, PASSING, RETURNING
        self.overtaking_phase_start = 0
        self.overtaking_durations = None
        self.overtaking_initial_distance = 0
        self.lane_change_direction = 0  # -1 = left, +1 = right, 0 = none
        self.can_change_lane = True  # Whether lane change is possible
        
        # History for smoothing (simple moving average)
        self.distance_history = deque(maxlen=5)
        
        print("=" * 60)
        print("Obstacle Avoidance System Initialized")
        print("=" * 60)
        print(f"Frame size: {frame_width}x{frame_height}")
        print(f"Lane boundaries: Left={self.LEFT_LANE_X}px, Right={self.RIGHT_LANE_X}px")
        print(f"Lane width: {self.LANE_WIDTH_PIXELS}px")
        print(f"Decision threshold: {self.CRITICAL_DISTANCE}m")
        print(f"Hysteresis: Enter={self.DECISION_ENTER}m, Exit={self.DECISION_EXIT}m")
        print(f"Preferred lane: {self.PREFERRED_LANE.upper()}")
        print("=" * 60)
    
    def smooth_distance(self, distance):
        """
        Apply simple moving average to smooth distance measurements.
        
        Args:
            distance (float): Raw distance measurement
            
        Returns:
            float: Smoothed distance
        """
        if distance > 0:  # Only add valid distances
            self.distance_history.append(distance)
        
        if len(self.distance_history) == 0:
            return distance
        
        return np.mean(self.distance_history)
    
    def classify_obstacle_zone(self, box):
        """
        Classify which third of the lane the obstacle is in.
        Uses bounding box overlap with the lane, not just the center point.

        Args:
            box (tuple): Bounding box (x1, y1, x2, y2)

        Returns:
            str: 'left_third', 'center_third', or 'right_third'
        """
        x1, y1, x2, y2 = box

        # Check if any part of the box overlaps with the lane
        if x2 < self.LEFT_LANE_X:
            return 'left_outside'
        elif x1 > self.RIGHT_LANE_X:
            return 'right_outside'

        # Clamp box edges to lane boundaries to get the overlapping portion
        overlap_left = max(x1, self.LEFT_LANE_X)
        overlap_right = min(x2, self.RIGHT_LANE_X)
        overlap_center = (overlap_left + overlap_right) / 2

        # Determine which third based on the center of the overlapping portion
        lane_width = self.RIGHT_LANE_X - self.LEFT_LANE_X
        zone_width = lane_width / 3
        relative_pos = overlap_center - self.LEFT_LANE_X

        if relative_pos < zone_width:
            return 'left_third'
        elif relative_pos < 2 * zone_width:
            return 'center_third'
        else:
            return 'right_third'
    
    def calculate_obstacle_coverage(self, box):
        """
        Calculate what percentage of lane width the obstacle covers.
        Only counts the portion of the box that actually overlaps the lane.

        Args:
            box (tuple): Bounding box (x1, y1, x2, y2)

        Returns:
            float: Coverage ratio (0.0 to 1.0)
        """
        x1, y1, x2, y2 = box
        lane_width = self.RIGHT_LANE_X - self.LEFT_LANE_X

        # Only measure the overlap with the lane, not the full box width
        overlap_left = max(x1, self.LEFT_LANE_X)
        overlap_right = min(x2, self.RIGHT_LANE_X)
        overlap_width = max(0, overlap_right - overlap_left)

        coverage = overlap_width / lane_width
        return coverage
    
    def calculate_steering_bias(self, zone):
        """
        Calculate steering bias based on obstacle zone.
        Positive = steer right, Negative = steer left

        Args:
            zone (str): Obstacle zone

        Returns:
            float: Steering bias (-1.0 to 1.0)
        """
        if zone == 'left_third':
            return self.MICRO_ADJUST_STEERING  # Steer right
        elif zone == 'right_third':
            return -self.MICRO_ADJUST_STEERING  # Steer left
        elif zone == 'center_third':
            # Choose direction based on which side has more space
            # For now, default to right
            return self.MICRO_ADJUST_STEERING
        else:
            return 0.0

    def check_lane_change_safety(self, boxes, distances, obstacle_box, obstacle_distance):
        """
        Check if lane change is safe and determine which direction to change.

        For a two-lane road where we're normally on the right:
        - LEFT lane is PREFERRED (typically empty when we're on right)
        - But also considers obstacle position and adjacent lane availability
        - Smart fallback to right lane if left is blocked

        Decision Priority:
        1. Preferred lane (left) if clear
        2. Opposite lane (right) if preferred blocked
        3. Side with more space based on obstacle position
        4. STOP if all options blocked

        Args:
            boxes (list): All detected bounding boxes
            distances (list): Distances to all obstacles
            obstacle_box (tuple): The main obstacle we're avoiding
            obstacle_distance (float): Distance to main obstacle

        Returns:
            tuple: (can_change, direction, reason)
                   can_change (bool): Whether lane change is safe
                   direction (int): -1 for left, +1 for right, 0 for none
                   reason (str): Explanation of decision
        """
        # Determine preferred direction based on which lane we normally drive in
        if self.PREFERRED_LANE == 'right':
            preferred_direction = -1  # Go left (we're on right, left is empty)
            preferred_name = "LEFT"
            opposite_direction = 1
            opposite_name = "RIGHT"
        else:
            preferred_direction = 1  # Go right (we're on left, right is empty)
            preferred_name = "RIGHT"
            opposite_direction = -1
            opposite_name = "LEFT"

        # Calculate obstacle position within our lane
        x1, y1, x2, y2 = obstacle_box
        obstacle_center = (x1 + x2) / 2
        lane_center = (self.LEFT_LANE_X + self.RIGHT_LANE_X) / 2

        # Is obstacle more to the left or right of our lane center?
        obstacle_on_left_side = obstacle_center < lane_center

        # Check for obstacles in adjacent lanes
        left_lane_blocked = False
        right_lane_blocked = False
        left_lane_obstacle_count = 0
        right_lane_obstacle_count = 0

        for box, dist in zip(boxes, distances):
            if dist <= 0 or dist > obstacle_distance + 1.5:  # Check obstacles within 1.5m
                continue

            bx1, by1, bx2, by2 = box
            b_center = (bx1 + bx2) / 2

            # Check if obstacle is in left adjacent lane (left of our left boundary)
            if b_center < self.LEFT_LANE_X - 50:  # 50px margin to avoid edge detection
                left_lane_blocked = True
                left_lane_obstacle_count += 1

            # Check if obstacle is in right adjacent lane (right of our right boundary)
            if b_center > self.RIGHT_LANE_X + 50:  # 50px margin
                right_lane_blocked = True
                right_lane_obstacle_count += 1

        # Decision logic with multiple factors
        print(f"  🔍 Lane Safety Check:")
        print(f"     Preferred: {preferred_name} lane")
        print(f"     Left lane: {'BLOCKED ({} obstacles)'.format(left_lane_obstacle_count) if left_lane_blocked else 'CLEAR ✓'}")
        print(f"     Right lane: {'BLOCKED ({} obstacles)'.format(right_lane_obstacle_count) if right_lane_blocked else 'CLEAR ✓'}")
        print(f"     Obstacle position: {'LEFT side' if obstacle_on_left_side else 'RIGHT side'} of lane")

        # Strategy 1: Try preferred lane first (usually LEFT when on RIGHT lane)
        if preferred_direction == -1:  # Prefer LEFT
            if not left_lane_blocked:
                return True, -1, f"Overtaking {preferred_name} - preferred lane clear"
            elif not right_lane_blocked:
                # Preferred blocked, but opposite clear
                if obstacle_on_left_side:
                    # Obstacle on left, going right makes more sense anyway
                    return True, 1, f"Overtaking {opposite_name} - obstacle on left, right lane clear"
                else:
                    # Obstacle on right, but left blocked, so go right
                    return True, 1, f"Overtaking {opposite_name} - preferred blocked but opposite clear"
            else:
                # Both lanes blocked - STOP
                return False, 0, "STOP - Both lanes blocked, cannot overtake safely"

        else:  # Prefer RIGHT
            if not right_lane_blocked:
                return True, 1, f"Overtaking {preferred_name} - preferred lane clear"
            elif not left_lane_blocked:
                # Preferred blocked, but opposite clear
                if not obstacle_on_left_side:  # obstacle on right
                    # Obstacle on right, going left makes more sense anyway
                    return True, -1, f"Overtaking {opposite_name} - obstacle on right, left lane clear"
                else:
                    # Obstacle on left, but right blocked, so go left
                    return True, -1, f"Overtaking {opposite_name} - preferred blocked but opposite clear"
            else:
                # Both lanes blocked - STOP
                return False, 0, "STOP - Both lanes blocked, cannot overtake safely"
    
    
    def calculate_overtaking_durations(self, obstacle_distance, obstacle_width, direction):
        """
        Calculate time durations for each phase of overtaking maneuver.
        Uses FIXED distance for passing since we can't detect when we've passed the obstacle.

        Args:
            obstacle_distance (float): Distance to obstacle in meters
            obstacle_width (float): Width of obstacle in meters (estimated from box)
            direction (int): -1 for left, +1 for right

        Returns:
            dict: Duration for each phase in seconds
        """
        # Phase 1: Move laterally to change lane
        vehicle_width = 0.2  # JetRacer width (meters)
        safety_margin = 0.2  # 20cm safety clearance

        # Need to move one full lane width
        lane_width_meters = self.LANE_WIDTH_PIXELS * 0.003  # Rough conversion: 1px ≈ 3mm at 1m distance
        # Alternative: Use calibrated value (e.g., standard lane is ~3.5m, but JetRacer course might be smaller)
        lane_width_meters = 0.4  # Typical miniature course lane width (adjust based on your setup)

        lateral_clearance = lane_width_meters
        phase1_duration = lateral_clearance / self.LATERAL_SPEED

        # Phase 2: Pass the obstacle (go straight in new lane)
        # FIXED distance since we can't see when we've passed
        # This should be enough to clear a stationary obstacle
        passing_distance = self.LANE_CHANGE_PASS_DISTANCE  # 2.0 meters default
        phase2_duration = passing_distance / self.VEHICLE_SPEED

        # Phase 3: Return to original lane (mirror of phase 1)
        phase3_duration = phase1_duration

        total = phase1_duration + phase2_duration + phase3_duration

        direction_name = "LEFT" if direction < 0 else "RIGHT"
        print(f"  📊 Overtaking plan ({direction_name}):")
        print(f"    - Lane width: {lane_width_meters:.2f}m")
        print(f"    - Fixed passing distance: {passing_distance:.2f}m")
        print(f"    - Phase 1 (move to {direction_name} lane): {phase1_duration:.1f}s")
        print(f"    - Phase 2 (pass straight): {phase2_duration:.1f}s")
        print(f"    - Phase 3 (return to original lane): {phase3_duration:.1f}s")
        print(f"    - Total time: {total:.1f}s")

        return {
            'phase1': phase1_duration,
            'phase2': phase2_duration,
            'phase3': phase3_duration,
            'total': total,
            'direction': direction
        }
    
    def estimate_obstacle_width_from_box(self, box, distance):
        """
        Estimate real-world obstacle width from bounding box and distance.
        
        Args:
            box (tuple): Bounding box (x1, y1, x2, y2)
            distance (float): Distance to obstacle in meters
            
        Returns:
            float: Estimated width in meters
        """
        x1, y1, x2, y2 = box
        box_width_pixels = x2 - x1
        
        # Simple approximation: assume obstacle width proportional to box width and distance
        # This is a rough estimate - calibrate with real measurements
        # Typical: at 1m distance, 100 pixels ≈ 0.3m
        estimated_width = (box_width_pixels / 100.0) * 0.3 * (distance / 1.0)
        
        # Clamp to reasonable values
        estimated_width = max(0.1, min(0.5, estimated_width))
        
        return estimated_width
    
    def decide_avoidance_action(self, box, distance, boxes, distances):
        """
        Make avoidance decision based on obstacle position and distance.

        Three main strategies:
        1. MICRO_ADJUST: Small obstacle in A or C zone only → slight steering, keep lane
        2. LANE_CHANGE: Large obstacle → check if safe, then execute maneuver
        3. STOP: Cannot avoid safely → stop and wait

        Args:
            box (tuple): Bounding box (x1, y1, x2, y2)
            distance (float): Distance to obstacle in meters
            boxes (list): All detected obstacles
            distances (list): Distances to all obstacles

        Returns:
            tuple: (action_name, steering_value, throttle_value)
        """
        # Classify obstacle
        zone = self.classify_obstacle_zone(box)
        coverage = self.calculate_obstacle_coverage(box)

        # Store for visualization
        self.obstacle_zone = zone
        self.obstacle_coverage = coverage

        # If obstacle is outside our lane, ignore it
        if zone in ['left_outside', 'right_outside']:
            return 'IGNORE', 0.0, self.THROTTLE

        # Decision logic:
        # MICRO_ADJUST: Small (<33% coverage) AND in outer zone (A or C) only
        # LANE_CHANGE or STOP: Everything else (touches B or too large)

        if coverage < self.MICRO_ADJUST_THRESHOLD and zone in ['left_third', 'right_third']:
            # Case 1: Small obstacle in A or C zone → micro adjustment, keep lane
            action = 'MICRO_ADJUST'
            steering = self.calculate_steering_bias(zone)
            throttle = self.THROTTLE

            print(f"  → Decision: {action} (stay in lane)")
            print(f"     Zone: {zone} | Coverage: {coverage:.2%} | Steering: {steering:+.2f}")

        else:
            # Case 2: Obstacle in B zone OR large obstacle → check if lane change is safe
            reason = "touches center zone" if zone == 'center_third' else f"large obstacle ({coverage:.1%})"

            # Check if lane change is safe and which direction
            can_change, direction, safety_reason = self.check_lane_change_safety(
                boxes, distances, box, distance
            )

            if can_change:
                # Lane change is safe - execute maneuver
                action = 'LANE_CHANGE'
                self.can_change_lane = True
                self.lane_change_direction = direction

                print(f"  → Decision: {action} ({reason})")
                print(f"     Zone: {zone} | Coverage: {coverage:.2%}")
                print(f"     Safety check: {safety_reason}")

                # Start timer-based overtaking
                obstacle_width = self.estimate_obstacle_width_from_box(box, distance)
                self.overtaking_durations = self.calculate_overtaking_durations(
                    distance, obstacle_width, direction
                )
                self.overtaking_state = 'MOVING_OVER'
                self.overtaking_phase_start = time.time()
                self.overtaking_initial_distance = distance

                # Steering based on determined direction
                if direction < 0:  # Left
                    steering = -self.LANE_CHANGE_STEERING
                else:  # Right
                    steering = self.LANE_CHANGE_STEERING

                throttle = self.THROTTLE

            else:
                # Cannot change lane safely - STOP
                action = 'STOP'
                self.can_change_lane = False
                steering = 0.0
                throttle = 0.0  # STOP

                print(f"  → Decision: {action} ({reason})")
                print(f"     Zone: {zone} | Coverage: {coverage:.2%}")
                print(f"     Safety check: {safety_reason}")
                print(f"     ⚠️  STOPPING - Cannot overtake safely!")

        return action, steering, throttle
    
    def update_overtaking_state(self):
        """
        Update timer-based overtaking state machine.
        Returns updated steering and throttle based on current phase.

        Three phases:
        1. MOVING_OVER: Change to adjacent lane (steer left/right)
        2. PASSING: Go straight in new lane for fixed distance
        3. RETURNING: Return to original lane (steer opposite direction)

        LKAS should be DISABLED during phases 1 and 3, ENABLED during phase 2.

        Returns:
            tuple: (steering, throttle, state_name, enable_lkas)
        """
        if self.overtaking_state == 'NORMAL':
            return 0.0, self.THROTTLE, 'NORMAL', True  # LKAS enabled in normal state

        elapsed = time.time() - self.overtaking_phase_start
        direction = self.overtaking_durations.get('direction', self.lane_change_direction)

        if self.overtaking_state == 'MOVING_OVER':
            # Phase 1: Move laterally to adjacent lane
            if elapsed > self.overtaking_durations['phase1']:
                self.overtaking_state = 'PASSING'
                self.overtaking_phase_start = time.time()
                print("  ✓ Phase 1 complete → Phase 2: Passing in new lane")
                print("  ℹ️  LKAS can now engage in new lane")
                return 0.0, self.THROTTLE, 'PASSING', True  # Enable LKAS in new lane

            # Continue moving over - LKAS should be disabled
            steering = -self.LANE_CHANGE_STEERING if direction < 0 else self.LANE_CHANGE_STEERING
            return steering, self.THROTTLE, 'MOVING_OVER', False  # Disable LKAS while changing

        elif self.overtaking_state == 'PASSING':
            # Phase 2: Go straight in new lane - LKAS keeps us centered
            if elapsed > self.overtaking_durations['phase2']:
                self.overtaking_state = 'RETURNING'
                self.overtaking_phase_start = time.time()
                print("  ✓ Phase 2 complete → Phase 3: Returning to original lane")
                print("  ℹ️  LKAS will be disabled during lane return")
                # Return in OPPOSITE direction to get back to original lane
                return_steering = self.LANE_CHANGE_STEERING if direction < 0 else -self.LANE_CHANGE_STEERING
                return return_steering, self.THROTTLE, 'RETURNING', False  # Disable LKAS

            # Continue going straight - LKAS keeps us in new lane
            return 0.0, self.THROTTLE, 'PASSING', True  # LKAS enabled

        elif self.overtaking_state == 'RETURNING':
            # Phase 3: Return to original lane (opposite steering from phase 1)
            if elapsed > self.overtaking_durations['phase3']:
                self.overtaking_state = 'NORMAL'
                self.decision_made = False
                print("  ✓ Phase 3 complete → Back in original lane!")
                print("  ℹ️  LKAS re-enabled in original lane")
                return 0.0, self.THROTTLE, 'NORMAL', True  # LKAS enabled in original lane

            # Continue returning - steer opposite direction from phase 1
            return_steering = self.LANE_CHANGE_STEERING if direction < 0 else -self.LANE_CHANGE_STEERING
            return return_steering, self.THROTTLE, 'RETURNING', False  # Disable LKAS

        return 0.0, self.THROTTLE, 'NORMAL', True
    
    def process_obstacles(self, detections, distances, boxes):
        """
        Process all detected obstacles and make avoidance decisions.
        Uses hysteresis to prevent oscillation.
        Now includes timer-based overtaking for LANE_CHANGE maneuvers.

        Args:
            detections (list): List of detection results
            distances (list): List of distances for each detection
            boxes (list): List of bounding boxes

        Returns:
            tuple: (action, steering, throttle, enable_lkas)
                   enable_lkas: Whether LKAS should be enabled (False during lane changes)
        """
        # If currently in overtaking maneuver, continue with timer-based control
        if self.overtaking_state != 'NORMAL':
            steering, throttle, state, enable_lkas = self.update_overtaking_state()
            self.current_action = f'LANE_CHANGE_{state}'
            self.current_steering = steering
            self.current_throttle = throttle
            return self.current_action, steering, throttle, enable_lkas
        
        # If no detections, return to normal
        if len(detections) == 0:
            if self.decision_made:
                # Check if we should clear the decision
                self.decision_made = False
                self.current_action = 'NORMAL'
                self.current_steering = 0.0
                self.current_throttle = self.THROTTLE
                print("No obstacles detected - returning to normal operation")

            return self.current_action, self.current_steering, self.current_throttle, True  # LKAS enabled
        
        # Find closest valid obstacle
        closest_distance = float('inf')
        closest_box = None
        closest_valid = False

        # Store all obstacles for lane change safety check
        all_boxes = boxes
        all_distances = distances

        for distance, box in zip(distances, boxes):
            if distance > 0 and distance < closest_distance:
                closest_distance = distance
                closest_box = box
                closest_valid = True
        
        # If no valid obstacles, return to normal
        if not closest_valid:
            if self.decision_made:
                self.decision_made = False
                self.current_action = 'NORMAL'
                self.current_steering = 0.0
                self.current_throttle = self.THROTTLE
            return self.current_action, self.current_steering, self.current_throttle, True  # LKAS enabled
        
        # Smooth the distance
        smoothed_distance = self.smooth_distance(closest_distance)
        
        # ========== HYSTERESIS LOGIC ==========
        
        # EXITING decision zone - clear decision
        if smoothed_distance > self.DECISION_EXIT:
            if self.decision_made:
                print(f"[HYSTERESIS] Clearing decision at {smoothed_distance:.2f}m")
                self.decision_made = False
                self.current_action = 'MONITOR'
                self.current_steering = 0.0
                self.current_throttle = self.THROTTLE
                self.distance_history.clear()  # Clear history for next obstacle

        # ENTERING decision zone - make decision ONCE
        elif smoothed_distance < self.DECISION_ENTER and not self.decision_made:
            print(f"\n[DECISION] Obstacle at {smoothed_distance:.2f}m - Making decision...")
            action, steering, throttle = self.decide_avoidance_action(
                closest_box, smoothed_distance, all_boxes, all_distances
            )

            # Lock the decision (except for LANE_CHANGE which uses its own state machine)
            if action not in ['LANE_CHANGE', 'STOP']:
                self.decision_made = True
            elif action == 'STOP':
                # STOP is also locked until obstacle clears
                self.decision_made = True

            self.current_action = action
            self.current_steering = steering
            self.current_throttle = throttle
            self.decision_timestamp = time.time()

            if action == 'LANE_CHANGE':
                print(f"[LANE_CHANGE] Starting timer-based overtaking maneuver")
            elif action == 'STOP':
                print(f"[STOP] Vehicle will stop until obstacle clears")
            else:
                print(f"[HYSTERESIS] Decision locked: {action}")

        # IN decision zone - continue current action
        elif self.decision_made:
            # Decision already made, continue executing
            pass

        # Outside decision zone - just monitor
        else:
            self.current_action = 'MONITOR'
            self.current_steering = 0.0
            self.current_throttle = self.THROTTLE

        # Determine if LKAS should be enabled
        # Disable LKAS during MICRO_ADJUST (we're adding bias) and STOP
        # LKAS is managed by state machine during LANE_CHANGE
        if self.current_action in ['MICRO_ADJUST', 'STOP']:
            enable_lkas = False
        else:
            enable_lkas = True

        return self.current_action, self.current_steering, self.current_throttle, enable_lkas
    
    def draw_lane_overlay(self, frame):
        """
        Draw lane boundaries and zones on frame.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            np.ndarray: Frame with lane overlay
        """
        overlay = frame.copy()
        
        # Draw lane boundaries
        cv2.line(overlay, 
                 (self.LEFT_LANE_X, 0), 
                 (self.LEFT_LANE_X, self.frame_height),
                 (0, 255, 0), 2)
        cv2.line(overlay, 
                 (self.RIGHT_LANE_X, 0), 
                 (self.RIGHT_LANE_X, self.frame_height),
                 (0, 255, 0), 2)
        
        # Draw zone divisions
        zone_width = (self.RIGHT_LANE_X - self.LEFT_LANE_X) / 3
        zone1_x = int(self.LEFT_LANE_X + zone_width)
        zone2_x = int(self.LEFT_LANE_X + 2 * zone_width)
        
        cv2.line(overlay,
                 (zone1_x, 0),
                 (zone1_x, self.frame_height),
                 (255, 255, 0), 1, cv2.LINE_AA)
        cv2.line(overlay,
                 (zone2_x, 0),
                 (zone2_x, self.frame_height),
                 (255, 255, 0), 1, cv2.LINE_AA)
        
        # Add zone labels
        cv2.putText(overlay, "L", (self.LEFT_LANE_X + 30, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(overlay, "C", (zone1_x + 40, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(overlay, "R", (zone2_x + 30, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Blend overlay with original frame
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        return result
    
    def draw_status_panel(self, frame):
        """
        Draw status information panel on frame.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            np.ndarray: Frame with status panel
        """
        panel_height = 180
        panel = np.zeros((panel_height, self.frame_width, 3), dtype=np.uint8)
        
        # Action status with color coding
        action_colors = {
            'NORMAL': (0, 255, 0),      # Green
            'MONITOR': (0, 255, 255),   # Yellow
            'MICRO_ADJUST': (255, 255, 0), # Cyan - stay in lane
            'LANE_CHANGE': (0, 0, 255),    # Red - full lane change
            'LANE_CHANGE_MOVING_OVER': (0, 0, 255),  # Red
            'LANE_CHANGE_PASSING': (0, 165, 255),     # Orange - LKAS active in new lane
            'LANE_CHANGE_RETURNING': (0, 0, 255),    # Red
            'STOP': (0, 0, 255),           # Red - emergency stop
            'IGNORE': (128, 128, 128)      # Gray
        }
        
        action = self.current_action if self.current_action else 'NORMAL'
        color = action_colors.get(action, (255, 255, 255))
        
        # Draw action
        cv2.putText(panel, f"Action: {action}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw overtaking state if active
        if self.overtaking_state != 'NORMAL':
            elapsed = time.time() - self.overtaking_phase_start
            if self.overtaking_durations:
                current_phase = self.overtaking_durations.get(
                    'phase1' if self.overtaking_state == 'MOVING_OVER' else
                    'phase2' if self.overtaking_state == 'PASSING' else 'phase3'
                )
                cv2.putText(panel, f"Timer: {elapsed:.1f}s / {current_phase:.1f}s",
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw decision status
        decision_text = "LOCKED" if self.decision_made else "READY"
        if self.overtaking_state != 'NORMAL':
            decision_text = f"OVERTAKING-{self.overtaking_state}"
        decision_color = (0, 0, 255) if self.decision_made or self.overtaking_state != 'NORMAL' else (0, 255, 0)
        cv2.putText(panel, f"State: {decision_text}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, decision_color, 2)
        
        # Draw steering and throttle
        cv2.putText(panel, f"Steering: {self.current_steering:+.2f}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(panel, f"Throttle: {self.current_throttle:.2f} (FIXED)",
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw obstacle info if available
        if self.obstacle_zone:
            cv2.putText(panel, f"Zone: {self.obstacle_zone}",
                       (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(panel, f"Coverage: {self.obstacle_coverage:.1%}",
                       (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw lane change direction if active
        if self.lane_change_direction != 0:
            direction_text = "→ LEFT" if self.lane_change_direction < 0 else "RIGHT →"
            direction_color = (0, 255, 255) if self.lane_change_direction < 0 else (255, 0, 255)
            cv2.putText(panel, f"Direction: {direction_text}",
                       (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, direction_color, 2)
        
        # Combine with main frame
        result = np.vstack([frame, panel])
        
        return result
    
    def get_control_outputs(self):
        """
        Get current steering and throttle values for robot control.
        
        Returns:
            tuple: (steering, throttle)
        """
        return self.current_steering, self.current_throttle
    
    def reset(self):
        """Reset the avoidance system state."""
        self.decision_made = False
        self.current_action = None
        self.current_steering = 0.0
        self.current_throttle = self.THROTTLE
        self.obstacle_zone = None
        self.obstacle_coverage = 0
        self.distance_history.clear()
        
        # Reset overtaking state machine
        self.overtaking_state = 'NORMAL'
        self.overtaking_phase_start = 0
        self.overtaking_durations = None
        self.overtaking_initial_distance = 0
        
        print("Avoidance system reset")


# ========== INTEGRATION FUNCTIONS ==========

def integrate_with_lane_keeping(avoidance_system):
    """
    Placeholder for lane-keeping integration.
    
    TODO: Replace this with actual integration:
    1. Import your lane-keeping module
    2. Get lane boundaries from lane detection
    3. Enable/disable lane-keeping based on action
    4. Combine steering outputs
    
    Args:
        avoidance_system: ObstacleAvoidanceSystem instance
    """
    # Example integration:
    # from lane_keeping import LaneKeeper
    # lane_keeper = LaneKeeper()
    # 
    # # Get current lane boundaries
    # left_x, right_x = lane_keeper.get_lane_boundaries()
    # avoidance_system.LEFT_LANE_X = left_x
    # avoidance_system.RIGHT_LANE_X = right_x
    # 
    # # Get current action
    # action, avoid_steering, avoid_throttle = avoidance_system.get_control_outputs()
    # 
    # # Combine steering
    # if action == 'MICRO_ADJUST':
    #     # Add bias to lane-keeping
    #     lane_steering = lane_keeper.get_steering()
    #     final_steering = lane_steering + avoid_steering
    # elif action in ['HARD_NUDGE', 'LANE_CHANGE']:
    #     # Override lane-keeping
    #     lane_keeper.disable()
    #     final_steering = avoid_steering
    # else:
    #     # Normal lane-keeping
    #     final_steering = lane_keeper.get_steering()
    # 
    # return final_steering, avoid_throttle
    
    pass


if __name__ == "__main__":
    """
    Test the avoidance system with mock data.
    This demonstrates how the system makes decisions.
    """
    print("\n" + "=" * 60)
    print("OBSTACLE AVOIDANCE SYSTEM TEST")
    print("=" * 60 + "\n")
    
    # Create avoidance system
    avoidance = ObstacleAvoidanceSystem()
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Small obstacle in left third',
            'box': (200, 200, 250, 300),  # Small box in left zone
            'distance': 0.45,
            'expected': 'MICRO_ADJUST'
        },
        {
            'name': 'Medium obstacle in center',
            'box': (280, 200, 360, 300),  # Medium box in center
            'distance': 0.48,
            'expected': 'HARD_NUDGE'
        },
        {
            'name': 'Large obstacle blocking lane',
            'box': (220, 200, 420, 300),  # Large box
            'distance': 0.42,
            'expected': 'LANE_CHANGE'
        },
        {
            'name': 'Warning zone obstacle',
            'box': (280, 200, 360, 300),
            'distance': 0.8,
            'expected': 'SLOW_DOWN'
        },
        {
            'name': 'Safe zone obstacle',
            'box': (280, 200, 360, 300),
            'distance': 1.2,
            'expected': 'MONITOR'
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\nTest {i}: {scenario['name']}")
        print("-" * 60)
        
        # Reset for each test
        avoidance.reset()
        
        # Process obstacle
        action, steering, throttle = avoidance.process_obstacles(
            detections=[1],
            distances=[scenario['distance']],
            boxes=[scenario['box']]
        )
        
        # Check result
        print(f"Expected: {scenario['expected']}")
        print(f"Got: {action}")
        print(f"Match: {'✓' if action == scenario['expected'] else '✗'}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)