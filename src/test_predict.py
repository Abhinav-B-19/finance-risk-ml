import pandas as pd
import time
from predict import predict, predict_batch

df = pd.read_csv("data/timeseries_final_v3.csv")

print("✅ Data loaded:", df.shape)

# SINGLE TEST
sample = df.iloc[1000].to_dict()
print("\nSingle:", predict(sample))

# MULTI TEST
print("\nMultiple:")
for s in df.sample(5).to_dict(orient="records"):
    print(predict(s)["risk_level"])

# BATCH TEST
batch = df.sample(50).to_dict(orient="records")
print("\nBatch:", predict_batch(batch)[:3])

# PERFORMANCE
start = time.time()
for _ in range(1000):
    predict(sample)
print("\nTime:", time.time() - start)

# DISTRIBUTION
dist = predict_batch(df.sample(200).to_dict(orient="records"))
scores = [r["risk_score"] for r in dist if r["status"] == "success"]
print("\nMin:", min(scores), "Max:", max(scores))

# import pandas as pd
# import time
# from predict import predict, predict_batch

# # ─────────────────────────────────────────
# # LOAD DATA
# # ─────────────────────────────────────────
# df = pd.read_csv("data/timeseries_final_v3.csv")

# print("✅ Data loaded:", df.shape)

# # ─────────────────────────────────────────
# # 1. SINGLE RECORD TEST
# # ─────────────────────────────────────────
# print("\n🔹 SINGLE RECORD TEST")

# sample = df.iloc[1000].to_dict()

# result = predict(sample)
# print("Prediction:", result)

# print("Actual distress_t2:", sample.get("distress_t2"))

# # ─────────────────────────────────────────
# # 2. MULTIPLE RANDOM TEST
# # ─────────────────────────────────────────
# print("\n🔹 MULTIPLE SAMPLE TEST")

# samples = df.sample(5).to_dict(orient="records")

# for i, s in enumerate(samples):
#     r = predict(s)
#     print(f"\nSample {i+1}")
#     print("Pred:", r["risk_level"], "| Score:", r["risk_score"])

# # ─────────────────────────────────────────
# # 3. BATCH TEST
# # ─────────────────────────────────────────
# print("\n🔹 BATCH TEST")

# batch_samples = df.sample(50).to_dict(orient="records")

# batch_results = predict_batch(batch_samples)

# print("Batch results (first 5):")
# print(batch_results[:5])

# # ─────────────────────────────────────────
# # 4. PERFORMANCE TEST
# # ─────────────────────────────────────────
# print("\n🔹 PERFORMANCE TEST")

# start = time.time()

# for _ in range(1000):
#     predict(sample)

# end = time.time()

# print("Time for 1000 predictions:", round(end - start, 4), "seconds")

# # ─────────────────────────────────────────
# # 5. DISTRIBUTION CHECK
# # ─────────────────────────────────────────
# print("\n🔹 DISTRIBUTION CHECK")

# dist_samples = df.sample(500).to_dict(orient="records")
# dist_results = predict_batch(dist_samples)

# scores = [r["risk_score"] for r in dist_results if r["status"] == "success"]

# print("Min score:", min(scores))
# print("Max score:", max(scores))
# print("Average score:", sum(scores) / len(scores))

# # ─────────────────────────────────────────
# # 6. ACCURACY CHECK (OPTIONAL)
# # ─────────────────────────────────────────
# print("\n🔹 BASIC ACCURACY CHECK")

# correct = 0
# total = 50

# test_rows = df.sample(total)

# for _, row in test_rows.iterrows():
#     pred = predict(row.to_dict())

#     if pred["status"] == "success":
#         predicted_label = 1 if pred["alert"] == 1 else 0
#         actual_label = row["distress_t2"]

#         if predicted_label == actual_label:
#             correct += 1

# accuracy = correct / total
# print("Approx accuracy:", round(accuracy, 2))