import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Hộp Oxyz Thông Minh - Nhiệm Vụ 3: Thả Hàng Cứu Trợ", layout="wide"
)

st.markdown(
    "### 🚁 NHIỆM VỤ 3: TỌA ĐỘ CAMERA, KHOẢNG CÁCH VÀ TÍNH TOÁN VẬT LÝ THẢ"
    " HÀNG"
)

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("📷 1. Chụp ảnh sa bàn & Xác định tọa độ")
    st.info(
        "Chụp ảnh hộp sa bàn giấy để AI nhận diện. Tọa độ được quy ước:"
        " **1 đơn vị = 1 km** trên thực địa."
    )

    img_file = st.camera_input("Chụp ảnh mô hình sa bàn Oxyz")
    if img_file is not None:
        st.success("✅ Đã ghi nhận ảnh chụp từ camera thành công!")

    st.markdown("---")
    st.markdown("#### ⚙️ Thiết lập tọa độ 2 điểm trọng tâm:")

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
    st.markdown("🟡 **Điểm Vàng: Vị trí Drone cứu trợ**")
    y1_col, y2_col, y3_col = st.columns(3)
    with y1_col:
        x_yellow = st.number_input("X (Vàng)", value=2.0, step=0.5)
    with y2_col:
        y_yellow = st.number_input("Y (Vàng)", value=3.0, step=0.5)
    with y3_col:
        z_yellow = st.number_input("Z (Vàng)", value=5.0, step=0.5)

    # Tạo vector mảng numpy
    P_red = np.array([x_red, y_red, z_red])  # Nạn nhân (Đỏ)
    P_yellow = np.array([x_yellow, y_yellow, z_yellow])  # Drone (Vàng)

    # --- TÍNH TOÁN CƠ BẢN ---
    # Vectơ nối từ Drone tới Nạn nhân
    vec_drone_to_victim = P_red - P_yellow
    # Độ dài đại số của vectơ (khoảng cách giữa 2 điểm) tính theo đơn vị km
    distance = np.linalg.norm(vec_drone_to_victim)

    # --- PHẦN 1: THẢ HÀNG THẲNG ĐỨNG (GIỮ NGUYÊN ĐỘ CAO DRONE) ---
    # Điểm thả hàng thẳng đứng ngay phía trên hoặc tại chỗ nhưng giữ nguyên cao độ Z của drone
    P_drop_vertical = np.array([P_red[0], P_red[1], P_yellow[2]])

    # --- PHẦN 2: THẢ HÀNG KHI ĐANG BAY (VẬT LÝ RƠI TỰ DO) ---
    # Lấy độ cao h của drone (trục Z) đổi ra mét (1 đơn vị tọa độ = 1 km = 1000 mét)
    h_meters = P_yellow[2] * 1000.0  # mét
    g = 9.8  # gia tốc trọng trường (m/s^2)

    # Vận tốc drone cố định (chú thích)
    v_drone = 5.0  # m/s

    # Tính thời gian hàng rơi chạm đất: t = sqrt(2h / g)
    if h_meters > 0:
        t_fall = np.sqrt((2 * h_meters) / g)
    else:
        t_fall = 0.0

    # Khoảng cách theo phương ngang cần thả trước: d = v * t (tính bằng mét, quy đổi ra km để hiển thị trên hệ tọa độ)
    d_horizontal_meters = v_drone * t_fall
    d_horizontal_km = d_horizontal_meters / 1000.0

    # Tính tọa độ điểm bắt đầu thả hàng khi đang bay (drone bay ngược hướng vector phương ngang tới nạn nhân một khoảng d_horizontal_km)
    horizontal_dir = P_red[:2] - P_yellow[:2]
    norm_dir = np.linalg.norm(horizontal_dir)

    if norm_dir > 0:
        unit_dir = horizontal_dir / norm_dir
        release_xy = P_yellow[:2] + unit_dir * min(
            d_horizontal_km, norm_dir
        )  # Giới hạn không vượt quá khoảng cách thực tế
    else:
        release_xy = P_yellow[:2]

    P_release_flying = np.array([release_xy[0], release_xy[1], P_yellow[2]])

    # Chuyển đổi list để hiển thị
    list_red = [float(v) for v in P_red]
    list_yellow = [float(v) for v in P_yellow]
    list_vec = [float(v) for v in vec_drone_to_victim]

    st.markdown("---")
    st.subheader("📊 Kết quả Tính toán & Vật lý Cứu trợ")
    st.write(f"- 🔴 **Tọa độ Nạn nhân (Đỏ):** `{list_red}`")
    st.write(f"- 🟡 **Tọa độ Drone (Vàng):** `{list_yellow}`")
    st.write(f"- 📐 **Vectơ khoảng cách $\\vec{{DN}}$:** `{list_vec}`")
    st.warning(
        f"📏 **Khoảng cách trực diện (Độ dài đại số vectơ):** **{distance:.2f}"
        " km**"
    )

    st.markdown("#### 📦 1. Kịch bản Thả hàng thẳng đứng")
    st.success(
        f"• Tọa độ điểm thả (giữ nguyên độ cao Drone):"
        f" `({P_drop_vertical[0]:.1f}, {P_drop_vertical[1]:.1f},"
        f" {P_drop_vertical[2]:.1f})`"
    )

    st.markdown("#### ✈️ 2. Kịch bản Thả hàng khi đang bay (Ném ngang)")
    st.info(
        f"• Vận tốc drone cố định: **v = {v_drone} m/s** (chú thích: vận tốc bay"
        " ngang tiêu chuẩn của drone).\n"
        f"• Độ cao drone ($h$): **{h_meters:.1f} m** (tương ứng"
        f" {P_yellow[2]:.1f} km).\n"
        f"• Thời gian rơi chạm đất ($t = \\sqrt{{2h/g}}$): **{t_fall:.2f}"
        " giây**.\n"
        f"• Khoảng cách ngang cần thả trước ($d = v \\cdot t$):"
        f" **{d_horizontal_meters:.2f} m** (tương ứng"
        f" **{d_horizontal_km:.3f} km**).\n"
        f"• Tọa độ điểm cần kích hoạt thả hàng: `({P_release_flying[0]:.2f},"
        f" {P_release_flying[1]:.2f}, {P_release_flying[2]:.1f})`"
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
            text=["Nạn nhân (Đỏ)"],
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
            text=["Drone (Vàng)"],
            textposition="top center",
            name="Drone",
        )
    )

    # 3. Vẽ điểm thả hàng khi đang bay
    fig.add_trace(
        go.Scatter3d(
            x=[P_release_flying[0]],
            y=[P_release_flying[1]],
            z=[P_release_flying[2]],
            mode="markers+text",
            marker=dict(size=9, color="purple", symbol="cross"),
            text=["Điểm thả hàng khi bay"],
            textposition="bottom center",
            name="Điểm thả khi bay",
        )
    )

    # 4. Đường vector khoảng cách trực diện giữa Drone và Nạn nhân
    fig.add_trace(
        go.Scatter3d(
            x=[P_yellow[0], P_red[0]],
            y=[P_yellow[1], P_red[1]],
            z=[P_yellow[2], P_red[2]],
            mode="lines+markers",
            line=dict(color="orange", width=4, dash="dash"),
            name="Khoảng cách trực diện (Vectơ)",
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
    "💡 Dự án STEM 'Hộp Oxyz Thông Minh' - Kết hợp hình học vectơ không gian và"
    " vật lý chuyển động ném ngang thực tế."
)
