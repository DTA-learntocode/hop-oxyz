import streamlit as st
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Cấu hình trang
st.set_page_config(
    page_title="Hộp Oxyz Thông Minh - Chiến Dịch Cứu Trợ",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 HỘP OXYZ THÔNG MINH - TÍCH HỢP AI & NHIỆM VỤ 3")
st.markdown("""
*Ứng dụng STEM tự động phân tích không gian Oxyz, nhận diện điểm mốc qua Google Teachable Machine và giải quyết bài toán quỹ đạo cứu trợ.*
""")

# --- TẢI MODEL VÀ LABELS ĐÃ ĐƯA LÊN GITHUB ---
@st.cache_resource
def load_ai_model():
    try:
        # Load đúng tên file trong thư mục gốc GitHub
        model = load_model('keras_model.h5', compile=False)
        with open('labels.txt', 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines()]
        return model, class_names
    except Exception as e:
        return None, None

model, class_names = load_ai_model()

# --- GIAO DIỆN TẢI ẢNH HOẶC CAMERA ---
st.sidebar.subheader("📷 Nguồn Dữ Liệu AI Camera")
input_option = st.sidebar.radio("Chọn phương thức:", ["Tải ảnh chụp mô hình", "Nhập tọa độ thủ công"])

# Tọa độ mặc định theo đề bài/mô hình
coords = {
    "A": [2.0, 3.0, 1.0],  # Xanh dương (Kho hàng)
    "B": [8.0, 6.0, 4.0],  # Đỏ (Khu cứu trợ)
    "C": [5.0, 1.0, 3.0]   # Vàng (Trạm trung chuyển)
}

if input_option == "Tải ảnh chụp mô hình":
    uploaded_file = st.sidebar.file_uploader("Tải lên ảnh mô hình (Ghim Đỏ, Xanh, Vàng)", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.sidebar.image(image, caption="Ảnh mô hình thực tế", use_container_width=True)
        
        if model is not None and class_names is not None:
            # Xử lý ảnh đưa vào mô hình Keras/Teachable Machine
            image_resized = image.resize((224, 224))
            image_array = np.asarray(image_resized)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array
            
            # Dự đoán
            prediction = model.predict(data)
            index = np.argmax(prediction)
            confidence_score = prediction[0][index]
            detected_label = class_names[index]
            
            st.sidebar.success(f"🤖 AI Nhận diện: **{detected_label}** (Độ tin cậy: {confidence_score*100:.1f}%)")
        else:
            st.sidebar.warning("⚠️ Không tìm thấy file `keras_model.h5` hoặc `labels.txt` ở thư mục gốc GitHub. Đang dùng tọa độ mặc định.")
    else:
        st.info("💡 Hãy tải ảnh mô hình lên để AI phân tích.")
else:
    st.sidebar.markdown("Nhập tọa độ thực tế từ mô hình:")
    coords["A"][0] = st.sidebar.number_input("Ax (Xanh dương)", value=2.0)
    coords["A"][1] = st.sidebar.number_input("Ay", value=3.0)
    coords["A"][2] = st.sidebar.number_input("Az", value=1.0)
    
    coords["B"][0] = st.sidebar.number_input("Bx (Đỏ)", value=8.0)
    coords["B"][1] = st.sidebar.number_input("By", value=6.0)
    coords["B"][2] = st.sidebar.number_input("Bz", value=4.0)

ax, ay, az = coords["A"]
bx, by, bz = coords["B"]
cx, cy, cz = coords["C"]

# --- CÁC TAB CHỨC NĂNG ---
tab1, tab2 = st.tabs(["🚀 Giải Pháp Nhiệm Vụ 3 & Tối Ưu", "🌐 Không Gian Oxyz 3D Tương Tác"])

with tab1:
    st.subheader("⚙️ Phân Tích & Tính Toán Quỹ Đạo Thả Hàng (Nhiệm Vụ 3)")
    st.markdown("Xác định vị trí điểm thả hàng $M$ tối ưu dựa trên hệ số vectơ từ kho hàng $A$ và trạm trung chuyển $C$.")
    
    k_factor = st.slider("Hệ số tỉ lệ vectơ $k$ (Ví dụ: $\\vec{AM} = k\\cdot\\vec{AC}$)", 0.5, 3.0, 2.0, 0.5)
    
    # Tính tọa độ M
    mx = ax + k_factor * (cx - ax)
    my = ay + k_factor * (cy - ay)
    mz = az + k_factor * (cz - az)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Kho hàng A (Xanh dương)", f"({ax}, {ay}, {az})")
        st.metric("Trạm trung chuyển C (Vàng)", f"({cx}, {cy}, {cz})")
    with col2:
        st.metric("Khu cứu trợ B (Đỏ)", f"({bx}, {by}, {bz})")
        st.metric(f"Vị trí thả hàng M (k={k_factor})", f"({mx:.1f}, {my:.1f}, {mz:.1f})")

    # Kiểm tra biểu thức tối ưu
    ma = np.linalg.norm(np.array([ax - mx, ay - my, az - mz]))
    mb = np.linalg.norm(np.array([bx - mx, by - my, bz - mz]))
    mc = np.linalg.norm(np.array([cx - mx, cy - my, cz - mz]))
    t_score = abs(ma - 2*mb - mc)
    st.info(f"📊 **Biểu thức tối ưu $T = \vert{}MA - 2MB - MC\vert{}$ tại điểm M:** `{t_score:.2f}`")

with tab2:
    st.subheader("🌐 Mô Hình Trực Quan Họa Oxyz 3D")
    
    fig = go.Figure()
    
    points_config = {
        "O (Gốc tọa độ)": [0, 0, 0, "gray"],
        "A (Kho hàng - Xanh)": [ax, ay, az, "blue"],
        "B (Khu cứu trợ - Đỏ)": [bx, by, bz, "red"],
        "C (Trạm - Vàng)": [cx, cy, cz, "gold"],
        "M (Thả hàng)": [mx, my, mz, "purple"]
    }

    for name, (px, py, pz, color) in points_config.items():
        fig.add_trace(go.Scatter3d(
            x=[px], y=[py], z=[pz],
            mode='text+markers',
            marker=dict(size=10, color=color),
            text=[name],
            textposition="top center",
            name=name
        ))

    # Đường bay AB và AC-M
    fig.add_trace(go.Scatter3d(
        x=[ax, bx], y=[ay, by], z=[az, bz],
        mode='lines', line=dict(color='red', width=5), name='Đường bay AB'
    ))
    fig.add_trace(go.Scatter3d(
        x=[ax, cx, mx], y=[ay, cy, my], z=[az, cz, mz],
        mode='lines', line=dict(color='gold', width=4, dash='dash'), name='Quỹ đạo A-C-M'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Ox', range=[0, 11]),
            yaxis=dict(title='Oy', range=[0, 11]),
            zaxis=dict(title='Oz (Độ cao)', range=[0, 11]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("🚀 Ứng dụng STEM Hộp Oxyz Thông Minh - Sẵn sàng deploy hoàn hảo lên Streamlit Cloud.")
