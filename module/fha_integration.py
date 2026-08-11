"""FHA Rule + TensorRT AI + Fusion integration for the GUI pipeline."""

from module.upper_body import classify_fha


def fuse_fha(rule_result, ai_result):
    """Combine the FHA rule result and AI result using the project fusion rule."""
    if rule_result == "RULE_ABNORMAL":
        return "FUSION_ABNORMAL"

    if ai_result == "AI_ABNORMAL":
        return "FUSION_ABNORMAL"

    if rule_result == "RULE_NORMAL" and ai_result == "AI_NORMAL":
        return "FUSION_NORMAL"

    return "FUSION_BORDERLINE"


class FHAIntegration:
    """Lazy-load the TensorRT FHA classifier and return one integrated result."""

    def __init__(self):
        self._classifier = None

    def _get_classifier(self):
        if self._classifier is None:
            # Lazy import keeps the GUI importable on machines without TensorRT.
            from body_static_pose.fha_ai import FHAClassifier
            self._classifier = FHAClassifier()
        return self._classifier

    def analyze(self, frame, fha_angle):
        rule_result = classify_fha(fha_angle)
        ai_result = self._get_classifier().predict_result(frame)
        fusion_result = fuse_fha(rule_result, ai_result["result"])

        return {
            "angle": fha_angle,
            "rule": rule_result,
            "ai_score": ai_result["score"],
            "ai": ai_result["result"],
            "fusion": fusion_result,
        }
