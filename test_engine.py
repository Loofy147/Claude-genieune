from precision_targeting_engine import RealTargetingEngine, PromptGenerator, compute_head_entropy_fixed, detect_collapses
import numpy as np

def test_ioi_format():
    ioi_data = PromptGenerator.generate_ioi(5)
    print(f"Sample IOI: {ioi_data[0]}")
    assert "prompt" in ioi_data[0]
    assert "target" in ioi_data[0]

def test_entropy():
    # Test entropy with a uniform pattern
    pattern = np.ones((5, 5)) / 5.0
    entropy = compute_head_entropy_fixed(pattern)
    print(f"Uniform entropy: {entropy}")
    assert np.all(entropy >= 0.9)

def test_collapse():
    # Test collapse detection
    profile = np.array([0.9, 0.8, 0.5, 0.4])
    collapses = detect_collapses(profile, threshold=-0.2)
    print(f"Collapses: {collapses}")
    assert collapses == 1

if __name__ == "__main__":
    test_ioi_format()
    test_entropy()
    test_collapse()
    print("Local tests passed!")
