import cv2
import cv2.aruco as aruco
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Hộp Oxyz Thông Minh - Quét Đủ 3 Màu Ghim (Đỏ, Vàng, Xanh)",
    layout="wide",
)

# Khởi tạo session state lưu tọa độ 3 điểm
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

if "x_blue" not in st.session_state:
    st.session_state.x_blue = 1.0
if "y_blue" not in st.session_state:
    st.session_state.y_blue = 4.0
if "z_blue" not in st.session_state:
    st.session_state.z_blue = 0.0

st.markdown(
    "### 🚁 HỆ THỐNG OXYZ THÔNG MINH: TỰ ĐỘNG ĐỌC TỌA ĐỘ ĐỎ - VÀNG - XANH"
)

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("📷 1. Quét ArUco & Tự động đọc ghim Đỏ, Vàng, Xanh")
    st.info(
        "Chụp mặt sàn (Oxy) để AI tự động phát hiện ghim Đỏ (Nạn nhân), Vàng"
        " (Drone) và Xanh (Mốc/Trạm)."
    )

    tab_tuong, tab_san = st.tabs(
        ["🧱 1. Chụp Mặt Tường (Oxz)", "🟫 2. Chụp Mặt Sàn (Oxy)"]
    )

    with tab_tuong:
        st.markdown(
            "**Mặt tường ($Oxz$):** Quét các mốc **ID 0**, **ID 1**, **ID 3**."
        )
        img_tuong = st.camera_input("Chụp mặt tường Oxz", key="cam_tuong")
        if img_tuong is not None:
            st.success("✅ Đã ghi nhận ảnh mặt tường Oxz!")

    with tab_san:
        st.markdown(
            "**Mặt sàn ($Oxy$):** Quét các mốc **ID 0, 1, 2** và các ghim màu."
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
                    f"🎯 Đã phát hiện các mã ArUco dưới sàn: `{list(flat_ids)}`"
                )

                # Chuyển không gian màu sang HSV
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h_img, w_img = img.shape[:2]

                # 1. Mask màu Đỏ
                lower_red = np.array([0, 120, 70])
                upper_red = np.array([10, 255, 255])
                mask_red = cv2.inRange(hsv, lower_red, upper_red)
                M_red = cv2.moments(mask_red)
                if M_red["m00"] > 50:
                    cX = int(M_red["m10"] / M_red["m00"])
                    cY = int(M_red["m01"] / M_red["m00"])
                    st.session_state.x_red = round(
                        np.clip((cX / w_img) * 10.0, 0.0, 10.0), 1
                    )
                    st.session_state.y_red = round(
                        np.clip((1.0 - (cY / h_img)) * 10.0, 0.0, 10.0), 1
                    )

                # 2. Mask màu Vàng
                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([35, 255, 255])
                mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
                M_yellow = cv2.moments(mask_yellow)
                if M_yellow["m00"] > 50:
                    cX = int(M_yellow["m10"] / M_yellow["m00"])
                    cY = int(M_yellow["m01"] / M_yellow["m00"])
                    st.session_state.x_yellow = round(
                        np.clip((cX / w_img) * 10.0, 0.0, 10.0), 1
                    )
                    st.session_state.y_yellow = round(
                        np.clip((1.0 - (cY / h_img)) * 10.0, 0.0, 10.0), 1
                    )

                # 3. Mask màu Xanh (Xanh dương / Blue)
                lower_blue = np.array([100, 100, 100])
                upper_blue = np.array([130, 255, 255])
                mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
                M_blue = cv2.moments(mask_blue)
                if M_blue["m00"] > 50:
                    cX = int(M_blue["m10"] / M_blue["m00"])
                    cY = int(M_blue["m01"] / M_blue["m00"])
                    st.session_state.x_blue = round(
                        np.clip((cX / w_img) * 10.0, 0.0, 10.0), 1
                    )
                    st.session_state.y_blue = round(
                        np.clip((1.0 - (cY / h_img)) * 10.0, 0.0, 10.0), 1
                    )

                st.success(
                    "🤖 AI đã hoàn tất tự động nhận diện vị trí các ghim Đỏ,"
                    " Vàng, Xanh!"
                )
            else:
                st.error("❌ Không tìm thấy mã ArUco dưới sàn.")

    st.markdown("---")
    st.markdown("#### ⚙️ Tọa độ các điểm trên sa bàn:")

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

    # 🔵 Điểm Xanh
    st.markdown("🔵 **Điểm Xanh: Trạm / Mốc phụ**")
    b1, b2, b3 = st.columns(3)
    with b1:
        x_blue = st.number_input(
            "X (Xanh)", value=float(st.session_state.x_blue), step=0.5, key="bx"
        )
    with b2:
        y_blue = st.number_input(
            "Y (Xanh)", value=float(st.session_state.y_blue), step=0.5, key="by"
        )
    with b3:
        z_blue = st.number_input(
            "Z (Xanh)", value=float(st.session_state.z_blue), step=0.5, key="bz"
        )

    P_red = np.array([x_red, y_red, z_red])
    P_yellow = np.array([x_yellow, y_yellow, z_yellow])
    P_blue = np.array([x_blue, y_blue, z_blue])

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
    st.write(f"- 🔵 **Tọa độ Trạm mốc (Xanh):** `{list(P_blue)}`")
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

    # Điểm Xanh (Trạm mốc)
    fig.add_trace(
        go.Scatter3d(
            x=[P_blue[0]],
            y=[P_blue[1]],
            z=[P_blue[2]],
            mode="markers+text",
            marker=dict(size=11, color="blue", symbol="circle"),
            text=["Trạm mốc (Xanh)"],
            textposition="top center",
            name="Trạm mốc (Xanh)",
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
    "💡 Dự án STEM 'Hộp Oxyz Thông Minh' - Nhận diện và mô phỏng đồng thời 3 điểm"
    " Đỏ, Vàng, Xanh."
)
