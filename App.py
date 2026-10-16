import cv2
import cv2.aruco as aruco
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Hộp Oxyz Thông Minh - Nhận Diện ArUco & Vật Lý Cứu Trợ",
    layout="wide",
)

st.markdown(
    "### 🚁 HỆ THỐNG OXYZ THÔNG MINH: TỰ ĐỘNG NHẬN DIỆN ARUCO & TÍNH TOÁN VẬT LÝ"
)

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("📷 1. Chụp ảnh sa bàn (Có 4 mã ArUco góc)")
    st.info(
        "Chụp ảnh hộp sa bàn chứa 4 mã ArUco (ID 0, 1, 2, 3) ở các góc để AI"
        " tự động hiệu chuẩn không gian."
    )

    img_file = st.camera_input("Chụp ảnh mô hình sa bàn")

    # Mặc định khởi tạo tọa độ phòng hờ khi chưa quét được
    detected_aruco_info = "Chưa phát hiện đủ mã góc ArUco."
    x_red, y_red, z_red = 8.0, 6.0, 0.0
    x_yellow, y_yellow, z_yellow = 2.0, 3.0, 5.0

    if img_file is not None:
        st.success("✅ Đã nhận ảnh chụp! Đang xử lý thuật toán ArUco...")

        # Đọc ảnh từ Streamlit sang OpenCV
        bytes_data = img_file.getvalue()
        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Sử dụng từ điển chuẩn DICT_4X4_50
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, parameters)

        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            flattened_ids = ids.flatten()
            detected_count = sum(
                1 for i in [0, 1, 2, 3] if i in flattened_ids
            )
            detected_aruco_info = (
                f"Đã thấy {detected_count}/4 mã ArUco góc (IDs:"
                f" {list(flattened_ids)})."
            )

            if detected_count >= 3:
                st.success(f"🎯 {detected_aruco_info} - Hiệu chuẩn thành công!")
            else:
                st.warning(
                    f"⚠️ {detected_aruco_info} - Cần thấy đủ 4 mã góc để đạt độ"
                    " chính xác tuyệt đối."
                )
        else:
            st.error(
                "❌ Không tìm thấy mã ArUco nào trong ảnh. Vui lòng căn chỉnh"
                " lại góc chụp!"
            )

    st.markdown("---")
    st.markdown("#### ⚙️ Tọa độ các điểm (Đã đồng bộ hệ quy chiếu):")

    # Điểm Đỏ: Nạn nhân
    st.markdown("🔴 **Điểm Đỏ: Người cần cứu trợ**")
    r1, r2, r3 = st.columns(3)
    with r1:
        x_red = st.number_input("X (Đỏ)", value=float(x_red), step=0.5)
    with r2:
        y_red = st.number_input("Y (Đỏ)", value=float(y_red), step=0.5)
    with r3:
        z_red = st.number_input("Z (Đỏ)", value=float(z_red), step=0.5)

    # Điểm Vàng: Drone
    st.markdown("🟡 **Điểm Vàng: Vị trí Drone cứu trợ**")
    y1_col, y2_col, y3_col = st.columns(3)
    with y1_col:
        x_yellow = st.number_input("X (Vàng)", value=float(x_yellow), step=0.5)
    with y2_col:
        y_yellow = st.number_input("Y (Vàng)", value=float(y_yellow), step=0.5)
    with y3_col:
        z_yellow = st.number_input("Z (Vàng)", value=float(z_yellow), step=0.5)

    # Chuyển thành mảng NumPy
    P_red = np.array([x_red, y_red, z_red])
    P_yellow = np.array([x_yellow, y_yellow, z_yellow])

    # --- TÍNH TOÁN VẬT LÝ & HÌNH HỌC ---
    # 1. Khoảng cách trực diện (độ dài đại số vectơ giữa Drone và Nạn nhân)
    vec_direction = P_red - P_yellow
    distance = np.linalg.norm(vec_direction)

    # 2. Thả hàng thẳng đứng (giữ nguyên độ cao Drone)
    P_drop_vertical = np.array([P_red[0], P_red[1], P_yellow[2]])

    # 3. Thả hàng khi đang bay (Ném ngang)
    h_meters = P_yellow[2] * 1000.0  # 1 đơn vị tọa độ = 1 km = 1000 mét
    g = 9.8  # Gia tốc trọng trường (m/s^2) mặc định
    v_drone = 5.0  # Vận tốc drone cố định (m/s)

    # Thời gian rơi: t = sqrt(2h / g)
    t_fall = np.sqrt((2 * h_meters) / g) if h_meters > 0 else 0.0

    # Khoảng cách ngang cần thả trước: d = v * t (đổi từ mét sang km)
    d_horizontal_meters = v_drone * t_fall
    d_horizontal_km = d_horizontal_meters / 1000.0

    # Tọa độ điểm bắt đầu thả hàng khi bay
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
        f"📏 **Khoảng cách trực diện (Độ dài đại số vectơ):** **{distance:.2f}"
        " km**"
    )

    st.markdown("#### 📦 1. Kịch bản Thả hàng thẳng đứng")
    st.success(
        f"• Tọa độ điểm thả (giữ nguyên cao độ):"
        f" `({P_drop_vertical[0]:.1f}, {P_drop_vertical[1]:.1f},"
        f" {P_drop_vertical[2]:.1f})`"
    )

    st.markdown("#### ✈️ 2. Kịch bản Thả hàng khi đang bay (Ném ngang)")
    st.info(
        f"• Vận tốc drone cố định: v = {v_drone} m/s (vận tốc bay ngang tiêu"
        f" chuẩn).\n"
        f"• Độ cao drone (h): {h_meters:.1f} m (tương ứng {P_yellow[2]:.1f}"
        f" km).\n"
        f"• Thời gian rơi chạm đất (t): {t_fall:.2f} giây (với g = {g} m/s^2).\n"
        f"• Khoảng cách ngang cần thả trước (d): {d_horizontal_meters:.2f} m"
        f" (~{d_horizontal_km:.3f} km).\n"
        f"• Tọa độ điểm kích hoạt thả hàng: `({P_release_flying[0]:.2f},"
        f" {P_release_flying[1]:.2f}, {P_release_flying[2]:.1f})`"
    )

with col_right:
    st.subheader("🌐 Mô hình Không gian 3D Oxyz")

    fig = go.Figure()

    # Vẽ điểm Đỏ (Nạn nhân)
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

    # Vẽ điểm Vàng (Drone)
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

    # Vẽ điểm thả hàng khi đang bay
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

    # Vectơ khoảng cách trực diện
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

    # Cấu hình không gian 3D
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
    "💡 Dự án STEM 'Hộp Oxyz Thông Minh' - Tích hợp AI ArUco Marker và Vật lý"
    " chuyển động ném ngang."
)
