import cv2
import cv2.aruco as aruco
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Hộp Oxyz Thông Minh - Nhận Diện ArUco Oxz & Oxy",
    layout="wide",
)

st.markdown(
    "### 🚁 HỆ THỐNG OXYZ THÔNG MINH: CHỤP TÁCH MẶT PHẲNG OXZ & OXY TÍNH TỌA"
    " ĐỘ"
)

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("📷 1. Quét ArUco theo mặt phẳng (Khung hình 16:9)")
    st.info(
        "Tách thành 2 góc chụp vuông góc giúp camera loại bỏ độ méo phối cảnh"
        " và nhận diện 100% các mã ArUco!"
    )

    # Chia tab chụp ảnh chuẩn Oxz và Oxy
    tab_tuong, tab_san = st.tabs(
        ["🧱 1. Chụp Mặt Tường (Oxz)", "🟫 2. Chụp Mặt Sàn (Oxy)"]
    )

    with tab_tuong:
        st.markdown(
            "**Mặt tường ($Oxz$):** Chứa gốc **ID 0** $(O)$ và mã **ID 3**"
            " $(Oz)$."
        )
        img_tuong = st.camera_input("Chụp mặt tường Oxz", key="cam_tuong")

        if img_tuong is not None:
            bytes_data = img_tuong.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters()
            detector = aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                flat_ids = ids.flatten()
                st.success(
                    f"🎯 Đã nhận diện mã trên tường Oxz: IDs"
                    f" {list(flat_ids)}"
                )
            else:
                st.warning("⚠️ Chưa phát hiện mã trên mặt tường Oxz.")

    with tab_san:
        st.markdown(
            "**Mặt sàn ($Oxy$):** Chứa gốc **ID 0** $(O)$, mã **ID 1**"
            " $(Ox)$ và mã **ID 2** $(Oy)$."
        )
        img_san = st.camera_input("Chụp mặt sàn Oxy", key="cam_san")

        if img_san is not None:
            bytes_data = img_san.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters()
            detector = aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                flat_ids = ids.flatten()
                st.success(
                    f"🎯 Đã nhận diện mã dưới sàn Oxy: IDs {list(flat_ids)}"
                )
            else:
                st.warning("⚠️ Chưa phát hiện mã trên mặt sàn Oxy.")

    st.markdown("---")
    st.markdown("#### ⚙️ Thiết lập tọa độ điểm trên sa bàn:")

    # Điểm Đỏ: Nạn nhân (3, 2, 0)
    st.markdown("🔴 **Điểm Đỏ: Nạn nhân cứu trợ**")
    r1, r2, r3 = st.columns(3)
    with r1:
        x_red = st.number_input("X (Đỏ)", value=3.0, step=0.5)
    with r2:
        y_red = st.number_input("Y (Đỏ)", value=2.0, step=0.5)
    with r3:
        z_red = st.number_input("Z (Đỏ)", value=0.0, step=0.5)

    # Điểm Vàng: Drone (5, 0, 1)
    st.markdown("🟡 **Điểm Vàng: Drone cứu trợ**")
    y1_col, y2_col, y3_col = st.columns(3)
    with y1_col:
        x_yellow = st.number_input("X (Vàng)", value=5.0, step=0.5)
    with y2_col:
        y_yellow = st.number_input("Y (Vàng)", value=0.0, step=0.5)
    with y3_col:
        z_yellow = st.number_input("Z (Vàng)", value=1.0, step=0.5)

    P_red = np.array([x_red, y_red, z_red])
    P_yellow = np.array([x_yellow, y_yellow, z_yellow])

    # --- BÀI TOÁN VẬT LÝ & HÌNH HỌC ---
    vec_direction = P_red - P_yellow
    distance = np.linalg.norm(vec_direction)

    P_drop_vertical = np.array([P_red[0], P_red[1], P_yellow[2]])

    h_meters = P_yellow[2] * 1000.0  # Quy ước 1 đơn vị sa bàn = 1 km = 1000 m
    g = 9.8  # m/s^2
    v_drone = 5.0  # m/s

    t_fall = np.sqrt((2 * h_meters) / g) if h_meters > 0 else 0.0
    d_horizontal_meters = v_drone * t_fall
    d_horizontal_km = d_horizontal_meters / 1000.0

    horizontal_dir = P_red[:2] - P_yellow[:2]
    norm_dir = np.linalg.norm(horizontal_dir)
    if norm_dir > 0:
        unit_dir = horizontal_dir / norm_dir
        release_xy = P_yellow[:2] + unit_dir * min(
            d_horizontal_km, norm_dir
        )
    else:
        release_xy = P_yellow[:2]
    P_release_flying = np.array([release_xy[0], release_xy[1], P_yellow[2]])

    st.markdown("---")
    st.subheader("📊 Kết quả Phân tích & Tính toán")
    st.write(f"- 🔴 **Tọa độ Nạn nhân (Đỏ):** `{list(P_red)}`")
    st.write(f"- 🟡 **Tọa độ Drone (Vàng):** `{list(P_yellow)}`")
    st.write(f"- 📐 **Vectơ khoảng cách $\\vec{{DN}}$:** `{list(vec_direction)}`")
    st.warning(
        f"📏 **Khoảng cách trực diện (Độ dài vectơ):** **{distance:.2f} km**"
    )

    st.markdown("#### 📦 1. Kịch bản Thả hàng thẳng đứng")
    st.success(
        f"• Tọa độ điểm thả (giữ nguyên cao độ):"
        f" `({P_drop_vertical[0]:.1f}, {P_drop_vertical[1]:.1f},"
        f" {P_drop_vertical[2]:.1f})`"
    )

    st.markdown("#### ✈️ 2. Kịch bản Thả hàng khi đang bay (Ném ngang)")
    st.info(
        f"• Vận tốc drone cố định: v = {v_drone} m/s\n"
        f"• Độ cao drone (h): {h_meters:.1f} m (tương ứng {P_yellow[2]:.1f}"
        f" km)\n"
        f"• Thời gian rơi (t): {t_fall:.2f} giây (với g = {g} m/s^2)\n"
        f"• Khoảng cách ngang cần thả trước (d): {d_horizontal_meters:.2f} m"
        f" (~{d_horizontal_km:.3f} km)\n"
        f"• Tọa độ điểm kích hoạt thả: `({P_release_flying[0]:.2f},"
        f" {P_release_flying[1]:.2f}, {P_release_flying[2]:.1f})`"
    )

with col_right:
    st.subheader("🌐 Mô hình Không gian 3D Oxyz")

    fig = go.Figure()

    # Điểm Đỏ (Nạn nhân)
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

    # Điểm Vàng (Drone)
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

    # Điểm thả khi ném ngang
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

    # Vectơ nối khoảng cách
    fig.add_trace(
        go.Scatter3d(
            x=[P_yellow[0], P_red[0]],
            y=[P_yellow[1], P_red[1]],
            z=[P_yellow[2], P_red[2]],
            mode="lines+markers",
            line=dict(color="orange", width=4, dash="dash"),
            name="Khoảng cách trực diện",
        )
    )

    # Cấu hình không gian
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
    "💡 Dự án STEM 'Hộp Oxyz Thông Minh' - Tích hợp chụp tách mặt phẳng Oxz &"
    " Oxy & Mô phỏng Vật lý 3D."
)
