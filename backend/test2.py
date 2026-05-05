from models.emotion_model import AdvancedEmotionModel
import traceback

print("Initializing model...")
model = AdvancedEmotionModel()

print("Testing predict...")
try:
    pred = model.predict("I am so excited and happy to be alive! But sometimes I feel a little bit scared about the future.")
    print("PREDICT OK:", list(pred.keys()))
except Exception:
    traceback.print_exc()

print("Testing explain...")
try:
    exp = model.explain("I am so excited and happy to be alive! But sometimes I feel a little bit scared about the future.")
    print("EXPLAIN OK:", exp)
except Exception:
    traceback.print_exc()

print("DONE!")
