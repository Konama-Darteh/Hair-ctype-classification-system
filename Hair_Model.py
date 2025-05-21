import streamlit as st
import torch
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms as T
import sqlite3

# ---------------------- CONFIGURATION ----------------------
MODEL_PATH = r"C:/Users/Konama/OneDrive/Final year/Capstone/saved_models/hair_data_triplet_run_resnet18_best_model_full_model.pt"
RF_MODEL_PATH = r"C:/Users/Konama/OneDrive/Final year/Capstone/saved_models/optimized_rf_model.joblib"

LABELS = ['curly', 'kinky', 'wavy', 'straight', 'braids']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# ---------------------- MODEL & TRANSFORMS ----------------------
def load_model():
    model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.eval()
    return model.to(DEVICE)

def preprocess_image(image):
    transform = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ---------------------- RECOMMENDATION LOGIC ----------------------
def recommend_products(hair_type):
    try:
        # Load the optimized random forest model
        rf_model = joblib.load(RF_MODEL_PATH)

        # Load product datasets
        serum_df = pd.read_csv(r"C:\Users\Konama\OneDrive\Final year\Capstone\Dataset\product_dataset\serum.csv")
        shampoo_df = pd.read_csv(r"C:\Users\Konama\OneDrive\Final year\Capstone\Dataset\product_dataset\shampoo_conditoner.csv")
        conditioner_df = pd.read_csv(r"C:\Users\Konama\OneDrive\Final year\Capstone\Dataset\product_dataset\shampoo_data.csv")

         # Combine
        df = pd.concat([serum_df, shampoo_df, conditioner_df], ignore_index=True)

        # Standardize column names
        df.columns = df.columns.str.strip()

        required_columns = ['Product Name', 'Product Cost', 'HighLight']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"'{col}' column missing from dataset.")

        # Drop rows missing required fields
        df = df[required_columns].dropna()
        df['HighLight'] = df['HighLight'].astype(str).str.lower()

        # Normalize hair type
        hair_type = hair_type.lower()

        # Step 1: Try exact or partial hair type match
        filtered = df[df['HighLight'].str.contains(hair_type, na=False)]

        # Step 2: Fallback to general match
        if filtered.empty:
            filtered = df[df['HighLight'].str.contains("all hair types|all hair", na=False)]

        # Step 3: Still nothing? Return fallback row
        if filtered.empty:
            return pd.DataFrame([{
                "Product Name": "No matching products found.",
                "Product Cost": "—",
                "HighLight": f"No results for '{hair_type.title()}'. Please try again."
            }])

        return filtered.head(5)

    except Exception as e:
        return pd.DataFrame([{
            "Product Name": "Error loading recommendations",
            "Product Cost": "N/A",
            "HighLight": str(e)
        }])
    
#saving the results of the predicted hair type and products in a db
def save_to_sqlite(predicted_hair_type, recommendations):
    try:
        conn = sqlite3.connect("hair_predictions.db")
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hair_type TEXT,
            product_name TEXT,
            product_cost TEXT,
            highlight TEXT
        )
        """)

        # Insert each recommendation
        for _, row in recommendations.iterrows():
            cursor.execute("""
            INSERT INTO predictions (hair_type, product_name, product_cost, highlight)
            VALUES (?, ?, ?, ?)
            """, (
                predicted_hair_type,
                row.get("Product Name", "Unknown"),
                row.get("Product Cost", "N/A"),
                row.get("HighLight", "N/A")
            ))

        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"❌ Failed to save to database: {e}")
# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(page_title="Hair Classifier & Product Recommender", layout="centered")
st.title("🧠 Hair Type Detection & 💇🏾 Product Recommendation")

st.markdown("Upload or take a picture of your hair to get your hair type and product recommendations.")

image_file = st.file_uploader("📁 Upload an image", type=['png', 'jpg', 'jpeg'])

def run_pipeline(image):
    st.image(image, caption='📷 Input Image', width=400)

    # Load hair classification model
    model = load_model()
    input_tensor = preprocess_image(image).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        pred_class_idx = torch.argmax(output, dim=1).item()
        pred_class = LABELS[pred_class_idx]

    st.success(f"✅ Predicted Hair Type: **{pred_class.title()}**")

    # Recommend products
    st.markdown("### 🧴 Recommended Products:")
    recommendations = recommend_products(pred_class)

    for _, row in recommendations.iterrows():
        st.markdown(f"**🧴 {row.get('Product Name', 'Unknown')}**")
        st.markdown(f"💰 **Price:** {row.get('Product Cost', 'N/A')}")
        st.markdown(f"📋 **Details:** {row.get('HighLight', 'N/A')}")
        st.markdown("---")
    # Save results to SQLite
    save_to_sqlite(pred_class, recommendations)

#showing the saved results in the db
def show_saved_results():
    try:
        conn = sqlite3.connect("hair_predictions.db")
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
        conn.close()

        if df.empty:
            st.warning("📭 No saved predictions yet.")
        else:
            st.markdown("### 📊 Saved Predictions")
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Could not load saved results: {e}")

# Uploaded image
if image_file:
    image = Image.open(image_file).convert("RGB")
    run_pipeline(image)

# Camera input
st.markdown("---")
st.markdown("Or take a new picture:")
captured_img = st.camera_input("📸 Take a photo")

#botton to click to show all the results in the db
st.markdown("---")
if st.button("📂 View Saved Results"):
    show_saved_results()

if captured_img:
    image = Image.open(captured_img).convert("RGB")
    run_pipeline(image)
