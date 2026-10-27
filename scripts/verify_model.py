import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.model import load_or_create_model, predict_fundus_severity, SEVERITY_CLASSES
from src.preprocessing import preprocess_for_model, load_image

def main():
    weights_path = os.path.join("saved_models", "efficientnet_b0_dr.weights.h5")
    model = load_or_create_model(weights_path)

    test_cases = [
        ("User Healthy Image (Expected: 0)", r"C:\Users\Rishi\.gemini\antigravity-ide\brain\2299f3b3-d64a-4042-acca-5546c22d8e28\.user_uploaded\media_1793134209822.png", 0),
        ("User Mild Image (Expected: 1)", r"C:\Users\Rishi\.gemini\antigravity-ide\brain\2299f3b3-d64a-4042-acca-5546c22d8e28\.user_uploaded\media_1793136199441.jpg", 1),
        ("Dataset Grade 0 (Expected: 0)", "images/MAPLES_test_20051020_44923_0100_PP.jpg", 0),
        ("Dataset Grade 2 (Expected: 2)", "images/IDRID_test_IDRiD_57.jpg", 2),
        ("Dataset Grade 3 (Expected: 3)", "images/IDRID_test_IDRiD_55.jpg", 3)
    ]

    print("=== VERIFYING PREDICT_FUNDUS_SEVERITY (WITHOUT METADATA HINT) ===")
    all_pass = True
    for name, path, expected in test_cases:
        if os.path.exists(path):
            img = load_image(path)
            norm = preprocess_for_model(img)
            res = predict_fundus_severity(model, norm, metadata_hint=None)
            pred_g = res["predicted_grade"]
            conf = res["confidence"] * 100
            passed = (pred_g == expected)
            if not passed:
                all_pass = False
            status = "PASS" if passed else f"FAIL (got {pred_g})"
            print(f"[{status}] {name} -> Predicted Grade {pred_g} ({res['short_name']}) - Conf: {conf:.1f}%")
        else:
            print(f"[SKIP] File not found: {path}")

    if all_pass:
        print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")
    else:
        print("\n>>> SOME TESTS FAILED <<<")

if __name__ == "__main__":
    main()
