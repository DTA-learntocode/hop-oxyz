import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Hộp Oxyz Thông Minh - AI Camera Nhận Diện Màu", layout="wide"
)

st.markdown(
    "### 🤖 NHIỆM VỤ 3: AI CAMERA TỰ ĐỘNG NHẬN DIỆN TỌA ĐỘ GHIM ĐỎ & VÀNG"
)

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("📷 1. Chụp ảnh sa bàn qua Camera")
    img_file = st.camera_input(
        "Chụp ảnh hộp sa bàn chứa ghim Vàng và Ghim Đỏ rõ nét"
    )

    # Khởi tạo giá trị tọa độ mặc định (phòng khi chưa chụp ảnh)
    P_red = np.array([8.0, 6.0, 0.0])
    P_yellow = np.array([2.0, 3.0, 5.0])

    if img_file is not None:
        st.success("✅ Đang phân tích hình ảnh bằng AI Computer Vision...")

        # Đọc ảnh từ Streamlit UploadedFile bằng OpenCV
        bytes_data = img_file.getvalue()
        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Chuyển đổi sang không gian màu HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height, width, _ = img.shape

        # --- 1. LỌC VÀ TÌM ĐIỂM ĐỎ (NẠN NHÂN) ---
        # Ngưỡng màu đỏ trong OpenCV HSV thường chia làm 2 khoảng
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask_red = cv2.add(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2),
        )

        # Tìm tọa độ tâm đốm màu đỏ lớn nhất
        contours_r, _ = cv2.findContours(
            mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours_r:
            c_r = max(contours_r, key=cv2.contourArea)
            M_r = cv2.moments(c_r)
            if M_r["m00"] > 0:
                cx_r = int(M_r["m10"] / M_r["m00"])
                cy_r = int(M_r["m01"] / M_r["m00"])
                # Ánh xạ từ pixel ảnh sang tọa độ sa bàn (giả lập tỷ lệ lưới 0-10)
                mapped_x = round(cx_r * (10.0 / width), 1)
                mapped_y = round((height - cy_r) * (10.0 / height), 1)
                P_red = np.array([mapped_x, mapped_y, 0.0])

        # --- 2. LỌC VÀ TÌM ĐIỂM VÀNG (DRONE) ---
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        contours_y, _ = cv2.findContours(
            mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours_y:
            c_y = max(contours_y, key=cv2.contourArea)
            M_y = cv2.moments(c_y)
            if M_y["m00"] > 0:
                cx_y = int(M_y["m10"] / M_y["m00"])
                cy_y = int(M_y["m01"] / M_y["m00"])
                mapped_xy = round(cx_y * (10.0 / width), 1)
                mapped_yy = round((height - cy_y) * (10.0 / height), 1)
                P_yellow = np.array([mapped_xy, mapped_yy, 5.0])  # Cao độ cố định

    st.markdown("---")
    st.markdown(
        "#### ⚙️ Tọa độ trích xuất tự động (Có thể tinh chỉnh thủ công):"
    )

    r1, r2, r3 = st.columns(3)
    with r1:
        x_red = st.number_input("X (Đỏ)", value=float(P_red[0]), step=0.5)
    with r2:
        y_red = st.number_input("Y (Đỏ)", value=float(P_red[1]), step=0.5)
    with r3:
        z_red = st.number_input("Z (Đỏ)", value=float(P_red[2]), step=0.5)

    y1_col, y2_col, y3_col = st.columns(3)
    with y1_col:
        x_yellow = st.number_input(
            "X (Vàng)", value=float(P_yellow[0]), step=0.5
        )
    with y2_col:
        y_yellow = st.number_input(
            "Y (Vàng)", value=float(P_yellow[1]), step=0.5
        )
    with y3_col:
        z_yellow = st.number_input(
            "Z (Vàng)", value=float(P_yellow[2]), step=0.5
        )

    # Cập nhật mảng chính thức
    P_red = np.array([x_red, y_red, z_red])
    P_yellow = np.array([x_yellow, y_yellow, z_yellow])

    # --- TÍNH TOÁN KHOẢNG CÁCH VÀ ĐIỀU HƯỚNG ---
    vec_direction = P_red - P_yellow
    distance = np.linalg.norm(vec_direction)  # Độ dài đại số vector
    P_drop = np.array([P_red[0], P_red[1], P_red[2] + 0.5])

    st.markdown("---")
    st.subheader("📊 Kết quả Toán học Hình học Không gian")
    st.write(f"- 🔴 **Tọa độ Nạn nhân (Đỏ):** `{list(P_red)}`")
    st.write(f"- 🟡 **Tọa độ Drone (Vàng):** `{list(P_yellow)}`")
    st.warning(
        f"📏 **Khoảng cách (Độ dài đại số vectơ):** **{distance:.2f} km**"
    )
    st.success(
        f"🎯 **Tọa độ Drone bay tới thả hàng:** `({P_drop[0]:.1f},"
        f" {P_drop[1]:.1f}, {P_drop[2]:.1f})`"
    )

with col_right:
    st.subheader("🌐 Mô hình Không gian 3D Oxyz")
    fig = go.Figure()

    # Vẽ điểm Đỏ
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

    # Vẽ điểm Vàng
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

    # Vẽ điểm thả hàng
    fig.add_trace(
        go.Scatter3d(
            x=[P_drop[0]],
            y=[P_drop[1]],
            z=[P_drop[2]],
            mode="markers+text",
            marker=dict(size=9, color="purple", symbol="cross"),
            text=["Điểm thả hàng"],
            textposition="bottom center",
            name="Điểm thả",
        )
    )

    # Đường bay
    fig.add_trace(
        go.Scatter3d(
            x=[P_yellow[0], P_drop[0]],
            y=[P_yellow[1], P_drop[1]],
            z=[P_yellow[2], P_drop[2]],
            mode="lines+markers",
            line=dict(color="orange", width=5, dash="dash"),
            name="Quỹ đạo bay Drone",
        )
    )

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
