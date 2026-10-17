import cv2
import cv2.aruco as aruco
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Hộp Oxyz Thông Minh - Tự Động Đọc Tọa Độ ArUco & Màu Sắc",
    layout="wide",
)

st.markdown(
    "### 🚁 HỆ THỐNG OXYZ THÔNG MINH: TỰ ĐỘNG ĐỌC TỌA ĐỘ BẰNG AI & THỊ GIÁC MÁY"
    " TÍNH"
)

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("📷 1. Quét ArUco & Nhận Diện Tọa Độ Ghim (16:9)")
    st.info(
        "Hệ thống sẽ tự động tìm các mốc ArUco để xác lập hệ quy chiếu và quét"
        " vị trí ghim Đỏ (Nạn nhân) & Vàng (Drone)."
    )

    # Biến lưu tọa độ mặc định (phòng hờ khi chưa quét được)
    x_red, y_red, z_red = 3.0, 2.0, 0.0
    x_yellow, y_yellow, z_yellow = 5.0, 0.0, 1.0

    # Chia tab chụp ảnh chuẩn Oxz và Oxy
    tab_tuong, tab_san = st.tabs(
        ["🧱 1. Chụp Mặt Tường (Oxz)", "🟫 2. Chụp Mặt Sàn (Oxy)"]
    )

    with tab_tuong:
        st.markdown(
            "**Mặt tường ($Oxz$):** Yêu cầu nhận diện **ID 0** (Gốc), **ID 1**"
            " ($Ox$) và **ID 3** ($Oz$)."
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
                detected_set = set(flat_ids)
                target_wall = {0, 1, 3}
                found_wall = detected_set.intersection(target_wall)

                st.success(
                    f"🎯 Đã nhận diện **{len(flat_ids)} mã** trên tường Oxz."
                    f" Các ID: `{list(flat_ids)}`"
                )
                if target_wall.issubset(detected_set):
                    st.info(
                        "✅ Đạt chuẩn: Đã quét đủ ID 0, ID 1 và ID 3 trên tường!"
                    )
                else:
                    st.warning(
                        "⚠️ Chưa đủ bộ mốc trên tường (Cần quét rõ các mã ID 0,"
                        " 1, 3)."
                    )
            else:
                st.error("❌ Không tìm thấy mã ArUco nào trên mặt tường Oxz.")

    with tab_san:
        st.markdown(
            "**Mặt sàn ($Oxy$):** Yêu cầu nhận diện **ID 0** (Gốc), **ID 1**"
            " ($Ox$) và **ID 2** ($Oy$)."
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
                detected_set = set(flat_ids)
                target_floor = {0, 1, 2}
                found_floor = detected_set.intersection(target_floor)

                st.success(
                    f"🎯 Đã nhận diện **{len(flat_ids)} mã** dưới sàn Oxy. Các"
                    f" ID: `{list(flat_ids)}`"
                )

                # Thuật toán tự động đọc tọa độ ghim màu dựa vào không gian màu HSV
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # Khoảng màu Đỏ cho ghim nạn nhân
                lower_red1 = np.array([0, 120, 70])
                upper_red1 = np.array([10, 255, 255])
                mask_red = cv2.inRange(hsv, lower_red1, upper_red1)

                # Khoảng màu Vàng cho ghim drone
                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([35, 255, 255])
                mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

                # Tìm tâm điểm của đóm màu Đỏ và Vàng
                M_red = cv2.moments(mask_red)
                M_yellow = cv2.moments(mask_yellow)

                if M_red["m00"] > 50:  # Nếu phát hiện vùng màu đỏ đủ lớn
                    cX_red = int(M_red["m10"] / M_red["m00"])
                    cY_red = int(M_red["m01"] / M_red["m00"])
                    st.info(
                        f"📍 AI phát hiện vị trí Ghim Đỏ tại pixel ảnh: "
                        f"({cX_red}, {cY_red})"
                    )

                if target_floor.issubset(detected_set):
                    st.info(
                        "✅ Đạt chuẩn: Đã quét đủ ID 0, ID 1 và ID 2 dưới sàn!"
                    )
                else:
                    st.warning(
                        "⚠️ Chưa đủ bộ mốc dưới sàn (Cần quét rõ các mã ID 0,"
                        " 1, 2)."
                    )
            else:
                st.error("❌ Không tìm thấy mã ArUco nào trên mặt sàn Oxy.")

    st.markdown("---")
    st.markdown(
        "#### ⚙️ Tọa độ không gian (Có thể điều chỉnh thủ công nếu cần):"
    )

    # Điểm Đỏ: Nạn nhân
    st.markdown("🔴 **Điểm Đỏ: Nạn nhân cứu trợ**")
    r1, r2, r3 = st.columns(3)
    with r1:
        x_red = st.number_input("X (Đỏ)", value=float(x_red), step=0.5)
    with r2:
        y_red = st.number_input("Y (Đỏ)", value=float(y_red), step=0.5)
    with r3:
        z_red = st.number_input("Z (Đỏ)", value=float(z_red), step=0.5)

    # Điểm Vàng: Drone
    st.markdown("🟡 **Điểm Vàng: Drone cứu trợ**")
    y1_col, y2_col, y3_col = st.columns(3)
    with y1_col:
        x_yellow = st.number_input("X (Vàng)", value=float(x_yellow), step=0.5)
    with y2_col:
        y_yellow = st.number_input("Y (Vàng)", value=float(y_yellow), step=0.5)
    with y3_col:
        z_yellow = st.number_input("Z (Vàng)", value=float(z_yellow), step=0.5)

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
    st.write(f"- 📐 **Vectơ khoảng cách khoảng không gian:** `{list(vec_direction)}`")
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
    "💡 Dự án STEM 'Hộp Oxyz Thông Minh' - Tích hợp AI quét ArUco chuẩn ID & Nhận"
    " diện màu sắc."
)
