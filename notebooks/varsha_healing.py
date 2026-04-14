# ===== Cell 0 =====
from google.colab import files
uploaded = files.upload()

# ===== Cell 1 =====
import os

images = [f for f in os.listdir("/content") if f.endswith((".png", ".jpg", ".jpeg"))]

print("Images found:", images)

# ===== Cell 2 =====
import cv2
import numpy as np

def create_mask(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)
    return mask

# ===== Cell 3 =====
def healing(mask):
    kernel = np.ones((7,7),np.uint8)
    return [
        mask,
        cv2.erode(mask,kernel,2),
        cv2.erode(mask,kernel,4),
        cv2.erode(mask,kernel,6)
    ]

def non_healing(mask):
    kernel = np.ones((5,5),np.uint8)
    return [
        mask,
        cv2.erode(mask,kernel,1),
        cv2.dilate(mask,kernel,1),
        mask
    ]

# ===== Cell 4 =====
import os

image_dir = "/content"

for i, img in enumerate(images):
    path = os.path.join(image_dir, img)
    m0 = create_mask(path)

    if i % 2 == 0:
        seq = healing(m0)
        label_type = "healing"
    else:
        seq = non_healing(m0)
        label_type = "non_healing"

    wid = f"W{i}"
    mask_store[wid] = seq

    for d,m in zip(days,seq):
        area = np.sum(m==255)
        data.append([wid,d,area,label_type])

df = pd.DataFrame(data,columns=["wound_id","day","area","type"])

# ===== Cell 5 =====
import pandas as pd
import numpy as np

features = []

for w,g in df.groupby("wound_id"):
    g = g.sort_values("day")
    areas = g["area"].values

    initial = areas[0]
    final = areas[-1]

    reduction = (initial-final)/initial

    label = 1 if g["type"].iloc[0] == "healing" else 0

    features.append([
        w,
        initial,
        final,
        reduction,
        np.mean(areas),
        np.std(areas),
        label
    ])

features_df = pd.DataFrame(features,columns=[
    "wound_id","initial","final","reduction","mean","std","label"
])

# ===== Cell 6 =====
balanced = features_df
from sklearn.model_selection import train_test_split
import xgboost as xgb

X = balanced.drop(["wound_id","label"],axis=1)
y = balanced["label"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,stratify=y,random_state=42
    )

model = xgb.XGBClassifier()
model.fit(X_train,y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:,1]
import matplotlib.pyplot as plt

if len(mask_store) > 0:
    wid = list(mask_store.keys())[5]
    seq = mask_store[wid]

    sample = features_df[features_df.wound_id==wid].drop(["wound_id","label"],axis=1)

    # Check if sample is not empty before prediction
    if not sample.empty:
        p = model.predict_proba(sample)[0][1]

        result = "Healing" if p>0.5 else "Non-Healing"

        plt.figure(figsize=(12,3))
        titles = ["Day0","Day7","Day14","Day21"]

        for i,m in enumerate(seq):
            plt.subplot(1,4,i+1)
            plt.imshow(m,cmap="gray")
            plt.title(titles[i])
            plt.axis("off")

        plt.suptitle(f"{wid} → {result} (Prob={round(p,2)})")
        plt.show()
    else:
        print(f"No sample data found for wound_id: {wid}")
else:
    print("mask_store is empty. Cannot visualize.")

# ===== Cell 7 =====
print(features_df["label"].value_counts())

# ===== Cell 8 =====
print(features_df)
print("\nLabel count:")
print(features_df["label"].value_counts())
