import cv2
import cv2.aruco as aruco
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Hộp Oxyz Thông Minh - Quét Ghim Đỏ & Vàng", layout="wide"
)

# Khởi tạo Session State lưu tọa độ ghim Đỏ (Sàn) và ghim Vàng (Tường)
if "x_red" not in st.session_state:
    st.session_state.x_red = 3.0
if "y_red" not in st.session_state:
    st.session_state.y_red = 2.0
if "z_red" not in st.session_state:
    st.session_state.z_red = 0.0

if "x_yellow" not in st.session_state:
    st.session_state.x_yellow = 5.0
if "y_yellow" not in st.session_state:
    st.session_state.y_yellow = 0.0
if "z_yellow" not in st.session_state:
    st.session_state.z_yellow = 1.0

st.markdown(
    "### 🚁 HỆ THỐNG OXYZ THÔNG MINH: TỰ ĐỘNG QUÉT TỌA ĐỘ ĐỎ (SÀN) & VÀNG"
    " (TƯỜNG)"
)

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("📷 1. Quét ArUco & Tự động đọc tọa độ ghim")
    st.info(
        "Chụp tường (Oxz) để lấy tọa độ Drone (Vàng) | Chụp sàn (Oxy) để lấy"
        " tọa độ Nạn nhân (Đỏ)."
    )

    tab_tuong, tab_san = st.tabs(
        ["🧱 1. Chụp Mặt Tường (Oxz)", "🟫 2. Chụp Mặt Sàn (Oxy)"]
    )

    with tab_tuong:
        st.markdown(
            "**Mặt tường ($Oxz$):** Quét mã **ID 0, 1, 3** & Tự động tìm vị trí"
            " **Ghim Vàng** (Drone)."
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
                    f"🎯 Đã phát hiện ArUco trên tường Oxz: `{list(flat_ids)}`"
                )

                # Thuật toán quét Ghim Vàng trên vách tường
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h_img, w_img = img.shape[:2]

                lower_yellow = np.array([15, 80, 80])
                upper_yellow = np.array([35, 255, 255])
                mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

                M_yellow = cv2.moments(mask_yellow)
                if M_yellow["m00"] > 50:
                    cX = int(M_yellow["m10"] / M_yellow["m00"])
                    cY = int(M_yellow["m01"] / M_yellow["m00"])

                    # Quy đổi pixel sang tọa độ Oxz (X ngang, Z cao)
                    auto_x = round(
                        np.clip((cX / w_img) * 10.0, 0.0, 10.0), 1
                    )
                    auto_z = round(
                        np.clip((1.0 - (cY / h_img)) * 10.0, 0.0, 10.0), 1
                    )

                    st.session_state.x_yellow = auto_x
                    st.session_state.z_yellow = auto_z
                    st.session_state.y_yellow = 0.0  # Tường sau Y = 0

                    st.success(
                        f"🤖 AI tự động tính Ghim Vàng (Drone): X = `{auto_x}`,"
                        f" Z = `{auto_z}`"
                    )
                else:
                    st.warning("⚠️ Không quét thấy ghim Vàng nổi bật trên tường.")
            else:
                st.error("❌ Không tìm thấy mã ArUco trên mặt tường Oxz.")

    with tab_san:
        st.markdown(
            "**Mặt sàn ($Oxy$):** Quét mã **ID 0, 1, 2** & Tự động tìm vị trí"
            " **Ghim Đỏ** (Nạn nhân)."
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
                    f"🎯 Đã phát hiện ArUco dưới sàn Oxy: `{list(flat_ids)}`"
                )

                # Thuật toán quét Ghim Đỏ dưới mặt sàn
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h_img, w_img = img.shape[:2]

                lower_red1 = np.array([0, 100, 70])
                upper_red1 = np.array([10, 255, 255])
                mask_red = cv2.inRange(hsv, lower_red1, upper_red1)

                M_red = cv2.moments(mask_red)
                if M_red["m00"] > 50:
                    cX = int(M_red["m10"] / M_red["m00"])
                    cY = int(M_red["m01"] / M_red["m00"])

                    # Quy đổi pixel sang tọa độ Oxy (X ngang, Y sâu)
                    auto_x = round(
                        np.clip((cX / w_img) * 10.0, 0.0, 10.0), 1
                    )
                    auto_y = round(
                        np.clip((1.0 - (cY / h_img)) * 10.0, 0.0, 10.0), 1
                    )

                    st.session_state.x_red = auto_x
                    st.session_state.y_red = auto_y
                    st.session_state.z_red = 0.0  # Nằm trên sàn Z = 0

                    st.success(
                        f"🤖 AI tự động tính Ghim Đỏ (Nạn nhân): X = `{auto_x}`,"
                        f" Y = `{auto_y}`"
                    )
                else:
                    st.warning("⚠️ Không quét thấy ghim Đỏ nổi bật dưới sàn.")
            else:
                st.error("❌ Không tìm thấy mã ArUco trên mặt sàn Oxy.")

    st.markdown("---")
    st.markdown("#### ⚙️ Thiết lập & Hiệu chỉnh tọa độ không gian:")

    # 🔴 Điểm Đỏ
    st.markdown("🔴 **Điểm Đỏ: Nạn nhân cứu trợ**")
    r1, r2, r3 = st.columns(3)
    with r1:
        x_red = st.number_input(
            "X (Đỏ)", value=float(st.session_state.x_red), step=0.5, key="rx"
        )
    with r2:
        y_red = st.number_input(
            "Y (Đỏ)", value=float(st.session_state.y_red), step=0.5, key="ry"
        )
    with r3:
        z_red = st.number_input(
            "Z (Đỏ)", value=float(st.session_state.z_red), step=0.5, key="rz"
        )

    # 🟡 Điểm Vàng
    st.markdown("🟡 **Điểm Vàng: Drone cứu trợ**")
    y1_col, y2_col, y3_col = st.columns(3)
    with y1_col:
        x_yellow = st.number_input(
            "X (Vàng)",
            value=float(st.session_state.x_yellow),
            step=0.5,
            key="yx",
        )
    with y2_col:
        y_yellow = st.number_input(
            "Y (Vàng)",
            value=float(st.session_state.y_yellow),
            step=0.5,
            key="yy",
        )
    with y3_col:
        z_yellow = st.number_input(
            "Z (Vàng)",
            value=float(st.session_state.z_yellow),
            step=0.5,
            key="yz",
        )

    P_red = np.array([x_red, y_red, z_red])
    P_yellow = np.array([x_yellow, y_yellow, z_yellow])

    # --- TÍNH TOÁN VẬT LÝ & HÌNH HỌC ---
    vec_direction = P_red - P_yellow
    distance = np.linalg.norm(vec_direction)

    P_drop_vertical = np.array([P_red[0], P_red[1], P_yellow[2]])

    h_meters = P_yellow[2] * 1000.0
    g = 9.8
    v_drone = 5.0

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
            name="Nạn nhân (Đỏ)",
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
            name="Drone (Vàng)",
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
    "💡 Dự án STEM 'Hộp Oxyz Thông Minh' - Tự động quét Ghim Đỏ (Sàn) & Ghim Vàng"
    " (Tường)."
)
