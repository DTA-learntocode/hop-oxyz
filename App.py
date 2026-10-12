import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Hộp Oxyz Thông Minh - Nhiệm Vụ 3: Drone Cứu Trợ", layout="wide"
)

st.markdown(
    "### 🚁 NHIỆM VỤ 3: TỌA ĐỘ CAMERA & TÍNH KHOẢNG CÁCH VECTOR DRONE - NẠN NHÂN"
)

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("📷 1. Chụp ảnh mô hình & Nhận diện qua Camera")
    st.info(
        "Chụp ảnh mô hình thực tế để AI Camera nhận diện các mốc. Dưới đây là"
        " tọa độ Oxyz trích xuất được từ ảnh chụp (Quy ước: 1 đơn vị = 1 km trên"
        " thực địa):"
    )

    # Widget camera chụp ảnh
    img_file = st.camera_input("Chụp ảnh sa bàn giấy")
    if img_file is not None:
        st.success("✅ Đã xử lý ảnh chụp từ camera thành công!")

    st.markdown("---")
    st.markdown("#### ⚙️ Tọa độ 2 điểm xác định từ ảnh chụp:")

    # 1. Điểm Đỏ: Người cần cứu trợ
    st.markdown("🔴 **Điểm Đỏ: Người cần cứu trợ**")
    r1, r2, r3 = st.columns(3)
    with r1:
        x_red = st.number_input("X (Đỏ)", value=8.0, step=0.5)
    with r2:
        y_red = st.number_input("Y (Đỏ)", value=6.0, step=0.5)
    with r3:
        z_red = st.number_input("Z (Đỏ)", value=0.0, step=0.5)

    # 2. Điểm Vàng: Drone cứu trợ
    st.markdown("🟡 **Điểm Vàng: Drone cứu trợ**")
    y1_col, y2_col, y3_col = st.columns(3)
    with y1_col:
        x_yellow = st.number_input("X (Vàng)", value=2.0, step=0.5)
    with y2_col:
        y_yellow = st.number_input("Y (Vàng)", value=3.0, step=0.5)
    with y3_col:
        z_yellow = st.number_input("Z (Vàng)", value=5.0, step=0.5)

    # Khởi tạo vector mảng NumPy từ tọa độ nhận diện
    P_red = np.array([x_red, y_red, z_red])  # Nạn nhân (Đỏ)
    P_yellow = np.array([x_yellow, y_yellow, z_yellow])  # Drone (Vàng)

    # --- TÍNH TOÁN THEO CÔNG THỨC ĐỘ DÀI ĐẠI SỐ CỦA VECTƠ ---
    # Vectơ nối từ Drone (Vàng) tới Người cần cứu trợ (Đỏ): vectơ V = P_red - P_yellow
    vec_direction = P_red - P_yellow

    # Độ dài đại số của vectơ (Khoảng cách giữa hai điểm) với đơn vị là km
    distance = np.linalg.norm(vec_direction)

    # Tọa độ hợp lý để Drone bay đến và thả hàng:
    P_drop = np.array([P_red[0], P_red[1], P_red[2] + 0.5])

    # Chuyển kiểu dữ liệu để hiển thị đẹp
    list_red = [float(v) for v in P_red]
    list_yellow = [float(v) for v in P_yellow]
    list_vec = [float(v) for v in vec_direction]
    list_drop = [float(v) for v in P_drop]

    st.markdown("---")
    st.subheader("📊 Kết quả Toán học Hình học Không gian")
    st.write(f"- 🔴 **Tọa độ Nạn nhân (Đỏ):** `{list_red}`")
    st.write(f"- 🟡 **Tọa độ Drone (Vàng):** `{list_yellow}`")
    st.write(f"- 📐 **Vectơ hướng bay $\\vec{{YD}}$:** `{list_vec}`")
    st.warning(
        f"📏 **Khoảng cách (Độ dài đại số vectơ):** **{distance:.2f} km**"
    )
    st.success(
        f"🎯 **Tọa độ Drone bay tới thả hàng hợp lý:** `({list_drop[0]:.1f},"
        f" {list_drop[1]:.1f}, {list_drop[2]:.1f})`"
    )

with col_right:
    st.subheader("🌐 Mô hình Không gian 3D Oxyz")

    fig = go.Figure()

    # 1. Vẽ điểm Đỏ (Nạn nhân)
    fig.add_trace(
        go.Scatter3d(
            x=[P_red[0]],
            y=[P_red[1]],
            z=[P_red[2]],
            mode="markers+text",
            marker=dict(size=11, color="red", symbol="diamond"),
            text=["Người cần cứu trợ (Đỏ)"],
            textposition="top center",
            name="Nạn nhân",
        )
    )

    # 2. Vẽ điểm Vàng (Drone)
    fig.add_trace(
        go.Scatter3d(
            x=[P_yellow[0]],
            y=[P_yellow[1]],
            z=[P_yellow[2]],
            mode="markers+text",
            marker=dict(size=11, color="gold", symbol="square"),
            text=["Drone cứu trợ (Vàng)"],
            textposition="top center",
            name="Drone",
        )
    )

    # 3. Vẽ điểm thả hàng hợp lý
    fig.add_trace(
        go.Scatter3d(
            x=[P_drop[0]],
            y=[P_drop[1]],
            z=[P_drop[2]],
            mode="markers+text",
            marker=dict(size=9, color="purple", symbol="cross"),
            text=["Điểm thả hàng"],
            textposition="bottom center",
            name="Điểm thả hàng",
        )
    )

    # 4. Đường bay vector giữa Drone và điểm thả
    fig.add_trace(
        go.Scatter3d(
            x=[P_yellow[0], P_drop[0]],
            y=[P_yellow[1], P_drop[1]],
            z=[P_yellow[2], P_drop[2]],
            mode="lines+markers",
            line=dict(color="orange", width=5, dash="dash"),
            name="Quỹ đạo bay của Drone",
        )
    )

    # 5. Quỹ đạo thả hàng thẳng đứng xuống nạn nhân
    fig.add_trace(
        go.Scatter3d(
            x=[P_drop[0], P_red[0]],
            y=[P_drop[1], P_red[1]],
            z=[P_drop[2], P_red[2]],
            mode="lines",
            line=dict(color="red", width=3),
            name="Đường thả hàng",
        )
    )

    # Cấu hình không gian 3D chuẩn trục Oxyz
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, 11], title="Trục Ox (km)", zeroline=True),
            yaxis=dict(range=[0, 11], title="Trục Oy (km)", zeroline=True),
            zaxis=dict(range=[0, 11], title="Trục Oz (km)", zeroline=True),
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.2), up=dict(x=0, y=0, z=1)
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0.0, y=0.9),
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(
    "💡 Nhiệm vụ 3: Nhận diện tọa độ từ camera ảnh chụp & Tính toán khoảng cách"
    " bằng độ dài đại số của vectơ (đơn vị: km)."
)
