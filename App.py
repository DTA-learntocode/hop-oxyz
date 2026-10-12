import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Hộp Oxyz Thông Minh - Nhiệm Vụ 3: Thả Hàng Cứu Trợ",
    layout="wide",
)

st.markdown(
    "### 🎯 NHIỆM VỤ 3: XÁC ĐỊNH VỊ TRÍ CỨU TRỢ & ĐIỂM THẢ HÀNG TỪ CAMERA AI"
)

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("📷 1. Chụp ảnh mô hình giấy (3 điểm màu)")
    st.info(
        "Sử dụng camera chụp lại mô hình giấy chứa 3 điểm màu trên trục Oxyz:"
        " **Xanh dương**, **Đỏ**, **Vàng**."
    )

    # Widget camera chụp ảnh trực tiếp
    img_file = st.camera_input("Chụp ảnh mô hình giấy")

    if img_file is not None:
        st.success(
            "✅ Đã chụp ảnh thành công! Hệ thống giả lập phân tích 3 điểm màu từ"
            " ảnh:"
        )

    st.markdown("---")
    st.markdown("#### ⚙️ Tọa độ 3 điểm nhận diện qua Camera AI:")

    # 3 điểm tương ứng với 3 màu trên mô hình giấy
    st.markdown("🔵 **Điểm 1: Xanh dương (Ví dụ: Trạm trung chuyển / Gốc chuẩn)**")
    c1, c2, c3 = st.columns(3)
    with c1:
        x_blue = st.number_input("X (Xanh dương)", value=2.0, step=0.5)
    with c2:
        y_blue = st.number_input("Y (Xanh dương)", value=3.0, step=0.5)
    with c3:
        z_blue = st.number_input("Z (Xanh dương)", value=1.0, step=0.5)

    st.markdown("🔴 **Điểm 2: Đỏ (Vị trí người cần cứu trợ)**")
    r1, r2, r3 = st.columns(3)
    with r1:
        x_red = st.number_input("X (Đỏ)", value=8.0, step=0.5)
    with r2:
        y_red = st.number_input("Y (Đỏ)", value=6.0, step=0.5)
    with r3:
        z_red = st.number_input("Z (Đỏ)", value=4.0, step=0.5)

    st.markdown("🟡 **Điểm 3: Vàng (Điểm mốc / Hướng dẫn chiếu)**")
    y1_col, y2_col, y3_col = st.columns(3)
    with y1_col:
        x_yellow = st.number_input("X (Vàng)", value=5.0, step=0.5)
    with y2_col:
        y_yellow = st.number_input("Y (Vàng)", value=1.0, step=0.5)
    with y3_col:
        z_yellow = st.number_input("Z (Vàng)", value=3.0, step=0.5)

    # Chuyển đổi thành các vector NumPy
    P_blue = np.array([x_blue, y_blue, z_blue])  # Điểm cơ sở A
    P_red = np.array([x_red, y_red, z_red])  # Điểm người cần cứu trợ B
    P_yellow = np.array([x_yellow, y_yellow, z_yellow])  # Điểm mốc C

    # --- TÍNH TOÁN VỊ TRÍ THẢ HÀNG HỢP LÝ (NHIỆM VỤ 3) ---
    # Giả sử vị trí thả hàng tối ưu (Điểm D) được tính dựa trên quan hệ vectơ hoặc trung điểm đường bay / tỉ lệ hình học
    # Ví dụ: Vectơ AD = 2 * AC hoặc tính điểm rơi tối ưu dựa trên trọng tâm / hình chiếu vuông góc xuống mặt đất
    vec_AC = P_yellow - P_blue
    P_drop = (
        P_blue + 2 * vec_AC
    )  # Điểm thả hàng dự phòng/hợp lý dựa theo quy tắc vectơ

    list_drop = [float(v) for v in P_drop]

    st.markdown("---")
    st.subheader("📊 Kết quả phân tích & Vị trí thả hàng")
    st.write(f"- **Vị trí Người cần cứu trợ (Đỏ):** `{list(P_red)}`")
    st.success(
        f"🎯 **Vị trí thả hàng cứu trợ hợp lý (Điểm tính toán):**"
        f" `({list_drop[0]:.1f}, {list_drop[1]:.1f}, {list_drop[2]:.1f})`"
    )

with col_right:
    st.subheader("🌐 Mô hình Không gian 3D Oxyz (Mô phỏng Sa bàn Giấy)")

    fig = go.Figure()

    # 1. Vẽ 3 điểm cốt lõi từ Camera màu sắc tương ứng
    fig.add_trace(
        go.Scatter3d(
            x=[P_blue[0]],
            y=[P_blue[1]],
            z=[P_blue[2]],
            mode="markers+text",
            marker=dict(size=9, color="blue"),
            text=["Điểm Xanh Dương"],
            textposition="top center",
            name="Xanh dương",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[P_red[0]],
            y=[P_red[1]],
            z=[P_red[2]],
            mode="markers+text",
            marker=dict(size=11, color="red", symbol="diamond"),
            text=["Người cần cứu trợ (Đỏ)"],
            textposition="top center",
            name="Người cần cứu trợ (Đỏ)",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[P_yellow[0]],
            y=[P_yellow[1]],
            z=[P_yellow[2]],
            mode="markers+text",
            marker=dict(size=9, color="gold"),
            text=["Điểm Vàng"],
            textposition="top center",
            name="Điểm Vàng",
        )
    )

    # 2. Vẽ vị trí thả hàng tính toán
    fig.add_trace(
        go.Scatter3d(
            x=[P_drop[0]],
            y=[P_drop[1]],
            z=[P_drop[2]],
            mode="markers+text",
            marker=dict(size=10, color="purple", symbol="cross"),
            text=["Vị trí Thả Hàng Hợp Lý"],
            textposition="bottom center",
            name="Điểm thả hàng",
        )
    )

    # 3. Đường kết nối vectơ chỉ hướng thả hàng
    fig.add_trace(
        go.Scatter3d(
            x=[P_blue[0], P_yellow[0], P_drop[0]],
            y=[P_blue[1], P_yellow[1], P_drop[1]],
            z=[P_blue[2], P_yellow[2], P_drop[2]],
            mode="lines+markers",
            line=dict(color="green", width=4, dash="dash"),
            name="Đường định hướng thả hàng",
        )
    )

    # Cấu hình khung không gian 3D chuẩn trục Oxyz theo yêu cầu
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, 11], title="Trục Ox", zeroline=True),
            yaxis=dict(range=[0, 11], title="Trục Oy", zeroline=True),
            zaxis=dict(range=[0, 11], title="Trục Oz", zeroline=True),
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
    "💡 Ứng dụng chuyên biệt Nhiệm vụ 3: Nhận diện màu sắc sa bàn giấy qua"
    " Camera & Tính toán điểm thả hàng."
)
