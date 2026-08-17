"""Segment domain tools for AI Analyst Agent."""

from typing import Dict, Any, List
from backend.app.database.session import SessionLocal
from backend.app.services.segment_service import SegmentService


class SegmentTools:
    @staticmethod
    def get_all_segments() -> List[Dict[str, Any]]:
        """Retrieve all active customer clusters, statistical labels, and RFM profiles."""
        db = SessionLocal()
        try:
            return SegmentService.get_segment_summaries(db)
        finally:
            db.close()

    @staticmethod
    def compare_segments(segment_id_a: int, segment_id_b: int) -> Dict[str, Any]:
        """Perform head-to-head comparison between two customer segments."""
        db = SessionLocal()
        try:
            segments = SegmentService.get_segment_summaries(db)
            seg_a = next((s for s in segments if s["segment_id"] == segment_id_a), None)
            seg_b = next((s for s in segments if s["segment_id"] == segment_id_b), None)
            
            if not seg_a or not seg_b:
                return {"error": "One or both segment IDs not found."}

            return {
                "segment_a": seg_a,
                "segment_b": seg_b,
                "revenue_ratio_a_to_b": round(seg_a["avg_revenue"] / max(0.01, seg_b["avg_revenue"]), 2),
                "frequency_ratio_a_to_b": round(seg_a["avg_frequency_30d"] / max(0.01, seg_b["avg_frequency_30d"]), 2),
            }
        finally:
            db.close()
