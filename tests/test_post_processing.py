#!/usr/bin/env python3
"""
tests/test_post_processing.py -- Unit Tests for Post-Processing Pipeline

Tests for:
- temporal_filter: Temporal consistency filtering
- confidence_stabilizer: EMA-based confidence smoothing  
- roi_monitor: ROI zone detection
- risk_scorer: Risk scoring logic
- alert_cooldown: Alert cooldown mechanism
- evidence_logger: Evidence logging (basic functionality)

Run with:
    python -m pytest tests/test_post_processing.py -v

Requirements:
    pip install pytest numpy opencv-python
"""

import sys
import pytest
import numpy as np
import os
import tempfile
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import post-processing modules
from post_processing.temporal_filter import TemporalConsistencyFilter
from post_processing.confidence_stabilizer import ConfidenceStabilizer
from post_processing.roi_monitor import ROIMonitor
from post_processing.risk_scorer import RiskScorer
from post_processing.alert_cooldown import AlertCooldown
from post_processing.evidence_logger import EvidenceLogger


class TestTemporalConsistencyFilter:
    """Test temporal consistency filtering."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.filter = TemporalConsistencyFilter(window_size=5, min_hits=3, min_confidence=0.30)
    
    def test_init(self):
        """Test initialization."""
        assert self.filter.window_size == 5
        assert self.filter.min_hits == 3
        assert self.filter.min_confidence == 0.30
    
    def test_single_detection_not_confirmed(self):
        """Single detection should not be confirmed (needs K=3 hits)."""
        detections = [{"class_name": "Handgun", "confidence": 0.95, "bbox": [100, 100, 150, 150]}]
        result = self.filter.update(detections)
        assert len(result) == 0, "Single frame detection should not be confirmed"
    
    def test_repeated_detection_confirmed(self):
        """Detection repeated in K frames should be confirmed."""
        detection = {"class_name": "Handgun", "confidence": 0.95, "bbox": [100, 100, 150, 150]}
        
        # Feed same detection for 3 frames
        for _ in range(3):
            result = self.filter.update([detection])
        
        # After 3 hits, should be confirmed
        assert len(result) > 0, "Repeated detection should be confirmed after K frames"
        assert result[0]["class_name"] == "Handgun"
    
    def test_low_confidence_ignored(self):
        """Low confidence detections should be filtered."""
        detection = {"class_name": "Handgun", "confidence": 0.20, "bbox": [100, 100, 150, 150]}
        
        # Feed low-confidence detection multiple times
        for _ in range(5):
            result = self.filter.update([detection])
        
        # Should never be confirmed due to confidence threshold
        assert len(result) == 0
    
    def test_reset(self):
        """Reset should clear buffer."""
        detection = {"class_name": "Handgun", "confidence": 0.95, "bbox": [100, 100, 150, 150]}
        for _ in range(2):
            self.filter.update([detection])
        
        self.filter.reset()
        result = self.filter.update([detection])
        assert len(result) == 0, "After reset, detection count should restart"
    
    def test_different_classes_tracked_separately(self):
        """Different weapon classes should be tracked separately."""
        handgun = {"class_name": "Handgun", "confidence": 0.95, "bbox": [100, 100, 150, 150]}
        rifle = {"class_name": "Rifle", "confidence": 0.95, "bbox": [200, 200, 300, 300]}
        
        # Alternate between handgun and rifle
        for _ in range(3):
            self.filter.update([handgun])
            self.filter.update([rifle])
        
        # Both should be confirmed
        result = self.filter.update([handgun, rifle])
        assert any(d["class_name"] == "Handgun" for d in result)


class TestConfidenceStabilizer:
    """Test EMA confidence smoothing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stabilizer = ConfidenceStabilizer(alpha=0.4)
    
    def test_init(self):
        """Test initialization."""
        assert self.stabilizer.alpha == 0.4
    
    def test_first_value(self):
        """First value should be stored as-is."""
        smoothed = self.stabilizer.smooth("Handgun", 0.8)
        assert smoothed == 0.8
    
    def test_convergence(self):
        """EMA should converge gradually to new values."""
        # First value
        smoothed1 = self.stabilizer.smooth("Handgun", 0.8)
        assert smoothed1 == 0.8
        
        # Second value - should be between 0.8 and 0.5
        smoothed2 = self.stabilizer.smooth("Handgun", 0.5)
        assert 0.5 < smoothed2 < 0.8, f"Expected smoothed value between 0.5 and 0.8, got {smoothed2}"
        
        # Should be closer to 0.5 after more iterations
        for _ in range(5):
            smoothed_next = self.stabilizer.smooth("Handgun", 0.5)
        assert smoothed_next < smoothed2, "EMA should converge toward new value"
    
    def test_high_certainty_override(self):
        """Values >= 0.95 should immediately set EMA."""
        self.stabilizer.smooth("Handgun", 0.5)
        smoothed = self.stabilizer.smooth("Handgun", 0.98)
        assert smoothed == 0.98, "High certainty (>= 0.95) should override EMA"
    
    def test_per_class_tracking(self):
        """Different classes should be tracked separately."""
        smoothed_hand = self.stabilizer.smooth("Handgun", 0.8)
        smoothed_rifle = self.stabilizer.smooth("Rifle", 0.5)
        
        assert smoothed_hand == 0.8
        assert smoothed_rifle == 0.5
        assert smoothed_hand != smoothed_rifle
    
    def test_reset_all(self):
        """Reset all should clear all classes."""
        self.stabilizer.smooth("Handgun", 0.8)
        self.stabilizer.smooth("Rifle", 0.6)
        self.stabilizer.reset()
        
        # After reset, first value should be stored as-is
        smoothed = self.stabilizer.smooth("Handgun", 0.5)
        assert smoothed == 0.5
    
    def test_reset_specific(self):
        """Reset specific class."""
        self.stabilizer.smooth("Handgun", 0.8)
        self.stabilizer.smooth("Rifle", 0.6)
        self.stabilizer.reset("Handgun")
        
        # Handgun should be reset, rifle preserved
        smoothed_rifle = self.stabilizer.get_ema("Rifle")
        assert smoothed_rifle != 0.0, "Rifle EMA should be preserved"


class TestROIMonitor:
    """Test ROI zone detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.roi_monitor = ROIMonitor()
        # Define a simple square ROI in center: [0.25, 0.25] to [0.75, 0.75]
        self.roi_square = [[[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]]
    
    def test_init(self):
        """Test initialization."""
        rois = self.roi_monitor.get_roi()
        assert rois == []
    
    def test_set_roi(self):
        """Test setting ROI zones."""
        self.roi_monitor.set_roi(self.roi_square)
        rois = self.roi_monitor.get_roi()
        assert len(rois) == 1
    
    def test_point_in_polygon(self):
        """Test point-in-polygon detection."""
        # Center point (0.5, 0.5) should be inside
        inside = ROIMonitor._point_in_polygon((0.5, 0.5), self.roi_square[0])
        assert inside, "Center point should be inside ROI"
        
        # Corner point (0.1, 0.1) should be outside
        outside = ROIMonitor._point_in_polygon((0.1, 0.1), self.roi_square[0])
        assert not outside, "Outside point should be outside ROI"
    
    def test_bbox_centroid_in_roi(self):
        """Test detection bbox centroid in ROI."""
        self.roi_monitor.set_roi(self.roi_square)
        
        # Bbox centered at (0.5, 0.5) - should be in ROI
        # Frame is 640x480, so bbox centered at pixel (320, 240)
        in_roi = self.roi_monitor.check_roi([300, 220, 340, 260], (480, 640))
        assert in_roi, "Centroid in ROI should return True"
    
    def test_bbox_centroid_outside_roi(self):
        """Test detection bbox centroid outside ROI."""
        self.roi_monitor.set_roi(self.roi_square)
        
        # Bbox centered at (0.1, 0.1) - should be outside ROI
        in_roi = self.roi_monitor.check_roi([50, 50, 100, 100], (480, 640))
        assert not in_roi, "Centroid outside ROI should return False"
    
    def test_clear_roi(self):
        """Test clearing ROI zones."""
        self.roi_monitor.set_roi(self.roi_square)
        self.roi_monitor.clear_roi()
        rois = self.roi_monitor.get_roi()
        assert rois == []


class TestRiskScorer:
    """Test risk scoring logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = RiskScorer(w1=0.5, w2=0.3, w3=0.2)
        self.frame_shape = (480, 640)  # H, W
    
    def test_init(self):
        """Test initialization."""
        assert self.scorer.w1 == 0.5
        assert self.scorer.w2 == 0.3
        assert self.scorer.w3 == 0.2
    
    def test_weights_sum_to_one(self):
        """Weights should sum to 1.0."""
        weight_sum = self.scorer.w1 + self.scorer.w2 + self.scorer.w3
        assert abs(weight_sum - 1.0) < 1e-6, "Weights should sum to 1.0"
    
    def test_area_score_computation(self):
        """Test area score calculation."""
        # Small bbox
        small_bbox = [0, 0, 100, 100]
        small_area = self.scorer._compute_area_score(small_bbox, self.frame_shape)
        
        # Large bbox
        large_bbox = [0, 0, 320, 240]
        large_area = self.scorer._compute_area_score(large_bbox, self.frame_shape)
        
        assert large_area > small_area, "Larger bbox should have higher area score"
    
    def test_spatial_priority_in_roi(self):
        """Test spatial priority with ROI."""
        # Bbox in ROI (should get 1.0)
        priority = self.scorer._compute_spatial_priority([0, 0, 100, 100], self.frame_shape, None, in_roi=True)
        assert priority == 1.0
    
    def test_spatial_priority_center(self):
        """Test spatial priority for center bbox."""
        # Bbox in center (should get 1.0)
        center_bbox = [200, 120, 400, 360]  # ~center of 640x480
        priority = self.scorer._compute_spatial_priority(center_bbox, self.frame_shape, None, in_roi=False)
        assert priority == 1.0, "Center bbox should get priority 1.0"
    
    def test_spatial_priority_corner(self):
        """Test spatial priority for corner bbox."""
        # Bbox in corner (should get 0.5)
        corner_bbox = [0, 0, 50, 50]
        priority = self.scorer._compute_spatial_priority(corner_bbox, self.frame_shape, None, in_roi=False)
        assert priority == 0.5, "Corner bbox should get priority 0.5"
    
    def test_risk_level_high(self):
        """Test high risk level."""
        level = RiskScorer.get_risk_level(0.75)
        assert level == "High"
    
    def test_risk_level_medium(self):
        """Test medium risk level."""
        level = RiskScorer.get_risk_level(0.50)
        assert level == "Medium"
    
    def test_risk_level_low(self):
        """Test low risk level."""
        level = RiskScorer.get_risk_level(0.25)
        assert level == "Low"


class TestAlertCooldown:
    """Test alert cooldown mechanism."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cooldown = AlertCooldown(cooldown_seconds=1.0)
    
    def test_first_alert_allowed(self):
        """First alert should always be allowed."""
        should_alert = self.cooldown.should_alert("Handgun")
        assert should_alert, "First alert should be allowed"
    
    def test_second_alert_blocked(self):
        """Second alert within cooldown should be blocked."""
        self.cooldown.should_alert("Handgun")
        should_alert = self.cooldown.should_alert("Handgun")
        assert not should_alert, "Alert within cooldown should be blocked"
    
    def test_different_classes_independent(self):
        """Different classes should have independent cooldowns."""
        self.cooldown.should_alert("Handgun")
        should_alert_rifle = self.cooldown.should_alert("Rifle")
        assert should_alert_rifle, "Different class should have independent cooldown"
    
    def test_different_regions_independent(self):
        """Different regions should have independent cooldowns."""
        self.cooldown.should_alert("Handgun", "region_1")
        should_alert_region2 = self.cooldown.should_alert("Handgun", "region_2")
        assert should_alert_region2, "Different region should have independent cooldown"
    
    def test_time_remaining(self):
        """Test time remaining calculation."""
        self.cooldown.should_alert("Handgun")
        remaining = self.cooldown.time_remaining("Handgun")
        assert 0 < remaining <= 1.0, "Time remaining should be between 0 and cooldown_seconds"
    
    def test_reset(self):
        """Test reset functionality."""
        self.cooldown.should_alert("Handgun")
        self.cooldown.reset("Handgun")
        should_alert = self.cooldown.should_alert("Handgun")
        assert should_alert, "After reset, alert should be allowed"


class TestEvidenceLogger:
    """Test evidence logging."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = EvidenceLogger(evidence_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization creates directory."""
        assert os.path.exists(self.temp_dir)
    
    def test_log_evidence(self):
        """Test logging detection evidence."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detection = {"class_name": "Handgun", "confidence": 0.95, "bbox": [100, 100, 150, 150]}
        risk_result = {"risk_score": 0.85, "risk_level": "High"}
        
        filename = self.logger.log(frame, detection, risk_result, "session_1")
        
        assert filename is not None, "Log should return filename"
        assert filename.startswith("alert_"), "Filename should start with 'alert_'"
        assert filename.endswith(".png"), "Filename should end with .png"
        assert os.path.exists(os.path.join(self.temp_dir, filename))
    
    def test_log_creates_json_metadata(self):
        """Test that logging creates JSON metadata."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detection = {"class_name": "Handgun", "confidence": 0.95, "bbox": [100, 100, 150, 150]}
        risk_result = {"risk_score": 0.85, "risk_level": "High"}
        
        filename = self.logger.log(frame, detection, risk_result, "session_1")
        json_filename = filename.replace(".png", ".json")
        
        assert os.path.exists(os.path.join(self.temp_dir, json_filename))
        
        # Verify JSON is valid
        with open(os.path.join(self.temp_dir, json_filename)) as f:
            metadata = json.load(f)
            assert metadata["class"] == "Handgun"
            assert metadata["risk_level"] == "High"


class TestIntegration:
    """Integration tests for complete pipeline."""
    
    def test_full_pipeline(self):
        """Test components working together."""
        # Initialize all components
        temporal = TemporalConsistencyFilter()
        stabilizer = ConfidenceStabilizer()
        roi_monitor = ROIMonitor()
        risk_scorer = RiskScorer()
        cooldown = AlertCooldown()
        
        # Create sample detection
        detection = {
            "class_name": "Handgun",
            "confidence": 0.85,
            "bbox": [100, 100, 150, 150]
        }
        
        # Process through pipeline
        # 1. Temporal filter (need 3 frames)
        for _ in range(3):
            confirmed = temporal.update([detection])
        
        # 2. Confidence stabilization
        if confirmed:
            smoothed = stabilizer.smooth(confirmed[0]["class_name"], confirmed[0]["confidence"])
            assert 0 < smoothed <= 1.0
        
        # 3. ROI check
        roi_monitor.set_roi([[[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]])
        in_roi = roi_monitor.check_roi([100, 100, 150, 150], (480, 640))
        assert isinstance(in_roi, (bool, np.bool_))
        
        # 4. Risk scoring
        risk = risk_scorer.score(detection, (480, 640))
        assert "risk_score" in risk
        assert "risk_level" in risk
        
        # 5. Alert cooldown
        should_alert = cooldown.should_alert(detection["class_name"])
        assert isinstance(should_alert, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
