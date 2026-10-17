import cv2
import cv2.aruco as aruco
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Hộp Oxyz - ID 0 1 2 3",
    layout="wide",
)

# Session state
for key, val in [("x_red", 3), ("y_red", 2), ("z_red", 0),
                 ("x_yellow", 5), ("y_yellow", 0), ("z_yellow", 1)]:
    if key not in st.session_state:
        st.session_state[key] = val

st.markdown("### 🚁 HỆ THỐNG OXYZ – Đúng ID 0, 1, 2, 3")

col_left, col_right = st.columns([1, 1.2])

# ====================== HÀM TIỆN ÍCH ======================
def get_marker_centers(corners, ids):
    centers = {}
    if ids is None:
        return centers
    for i, mid in enumerate(ids.flatten()):
        c = corners[i][0]
        centers[int(mid)] = (int(np.mean(c[:, 0])), int(np.mean(c[:, 1])))
    return centers

def compute_affine_coords(pin_pt, pts_src, pts_dst):
    if len(pts_src) < 3:
        return None
    M = cv2.getAffineTransform(np.float32(pts_src), np.float32(pts_dst))
    pt = np.array([[pin_pt]], dtype=np.float32)
    transformed = cv2.transform(pt, M)[0][0]
    x = float(np.clip(transformed[0], 0, 10))
    y = float(np.clip(transformed[1], 0, 10))
    return x, y

def detect_color_centroid(img, color="yellow"):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color == "yellow":
        mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
    else:
        mask1 = cv2.inRange(hsv, np.array([0, 100, 70]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 100, 70]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 80:
        return None
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

# ====================== CẤU HÌNH ARUCO PARAMETERS ======================
parameters = aruco.DetectorParameters()

# Tăng khả năng nhận khi mã nhỏ hoặc hơi mờ
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10
parameters.adaptiveThreshConstant = 7

# Cho phép mã hơi nghiêng và méo
parameters.minMarkerPerimeterRate = 0.02
parameters.maxMarkerPerimeterRate = 4.0
parameters.polygonalApproxAccuracyRate = 0.05
parameters.minCornerDistanceRate = 0.05
parameters.minDistanceToBorder = 3

# Giảm nhiễu
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
parameters.cornerRefinementWinSize = 5
parameters.cornerRefinementMaxIterations = 30
parameters.cornerRefinementMinAccuracy = 0.1

# ====================== GIAO DIỆN ======================
with col_left:
    st.subheader("📷 Quét ArUco & Ước lượng tọa độ")
    st.info(
        "**Vị trí mã ArUco:**  \n"
        "- **ID 0** = (0, 0, 0) – Gốc  \n"
        "- **ID 1** = (10, 0, 0) – Ox  \n"
        "- **ID 2** = (0, 10, 0) – Oy  \n"
        "- **ID 3** = (0, 0, 10) – Oz"
    )

    tab_tuong, tab_san = st.tabs(["🧱 Mặt Tường (Oxz)", "🟫 Mặt Sàn (Oxy)"])

    # ---------- TƯỜNG Oxz (ID 0, 1, 3) ----------
    with tab_tuong:
        st.markdown("**Mặt tường Oxz:** Cần **ID 0, 1, 3**")
        img_tuong = st.camera_input("Chụp mặt tường Oxz", key="cam_tuong")

        if img_tuong is not None:
            img = cv2.imdecode(np.frombuffer(img_tuong.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detector = aruco.ArucoDetector(
                aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
                parameters
            )
            corners, ids, _ = detector.detectMarkers(gray)

            img_draw = img.copy()
            if ids is not None:
                aruco.drawDetectedMarkers(img_draw, corners, ids)
                centers = get_marker_centers(corners, ids)
                st.success(f"Phát hiện: `{list(ids.flatten())}`")

                pin = detect_color_centroid(img, "yellow")
                if pin is not None:
                    cv2.circle(img_draw, pin, 12, (0, 255, 255), 3)
                    cv2.putText(img_draw, "Ghim Vang", (pin[0]+15, pin[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    if all(k in centers for k in [0, 1, 3]):
                        pts_src = [centers[0], centers[1], centers[3]]
                        pts_dst = [(0, 0), (10, 0), (0, 10)]  # XZ
                        result = compute_affine_coords(pin, pts_src, pts_dst)

                        if result:
                            x_cont, z_cont = result
                            auto_x = int(round(x_cont))
                            auto_z = int(round(z_cont))
                            st.session_state.x_yellow = auto_x
                            st.session_state.z_yellow = auto_z
                            st.session_state.y_yellow = 0

                            st.success(
                                f"🤖 **Affine (ID 0-1-3)**  \n"
                                f"Liên tục: X ≈ `{x_cont:.2f}`, Z ≈ `{z_cont:.2f}`  \n"
                                f"→ **Số nguyên: X = {auto_x}, Z = {auto_z}**"
                            )
                        else:
                            st.warning("Không tính được Affine.")
                    else:
                        st.warning("⚠️ Chưa đủ ID 0, 1, 3 trên tường.")
                else:
                    st.warning("⚠️ Không thấy ghim Vàng.")

                st.image(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.error("❌ Không tìm thấy ArUco.")

    # ---------- SÀN Oxy (ID 0, 1, 2) ----------
    with tab_san:
        st.markdown("**Mặt sàn Oxy:** Cần **ID 0, 1, 2**")
        img_san = st.camera_input("Chụp mặt sàn Oxy", key="cam_san")

        if img_san is not None:
            img = cv2.imdecode(np.frombuffer(img_san.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detector = aruco.ArucoDetector(
                aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
                parameters
            )
            corners, ids, _ = detector.detectMarkers(gray)

            img_draw = img.copy()
            if ids is not None:
                aruco.drawDetectedMarkers(img_draw, corners, ids)
                centers = get_marker_centers(corners, ids)
                st.success(f"Phát hiện: `{list(ids.flatten())}`")

                pin = detect_color_centroid(img, "red")
                if pin is not None:
                    cv2.circle(img_draw, pin, 12, (0, 0, 255), 3)
                    cv2.putText(img_draw, "Ghim Do", (pin[0]+15, pin[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if all(k in centers for k in [0, 1, 2]):
                        pts_src = [centers[0], centers[1], centers[2]]
                        pts_dst = [(0, 0), (10, 0), (0, 10)]  # XY
                        result = compute_affine_coords(pin, pts_src, pts_dst)

                        if result:
                            x_cont, y_cont = result
                            auto_x = int(round(x_cont))
                            auto_y = int(round(y_cont))
                            st.session_state.x_red = auto_x
                            st.session_state.y_red = auto_y
                            st.session_state.z_red = 0

                            st.success(
                                f"🤖 **Affine (ID 0-1-2)**  \n"
                                f"Liên tục: X ≈ `{x_cont:.2f}`, Y ≈ `{y_cont:.2f}`  \n"
                                f"→ **Số nguyên: X = {auto_x}, Y = {auto_y}**"
                            )
                        else:
                            st.warning("Không tính được Affine.")
                    else:
                        st.warning("⚠️ Chưa đủ ID 0, 1, 2 dưới sàn.")
                else:
                    st.warning("⚠️ Không thấy ghim Đỏ.")

                st.image(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.error("❌ Không tìm thấy ArUco.")

    # ---------- Hiệu chỉnh + tính toán ----------
    st.markdown("---")
    st.markdown("#### ⚙️ Hiệu chỉnh tọa độ nguyên")

    st.markdown("🔴 **Điểm Đỏ (Nạn nhân)**")
    c1, c2, c3 = st.columns(3)
    x_red = c1.number_input("X", value=int(st.session_state.x_red), step=1, key="rx")
    y_red = c2.number_input("Y", value=int(st.session_state.y_red), step=1, key="ry")
    z_red = c3.number_input("Z", value=int(st.session_state.z_red), step=1, key="rz")

    st.markdown("🟡 **Điểm Vàng (Drone)**")
    c1, c2, c3 = st.columns(3)
    x_yellow = c1.number_input("X", value=int(st.session_state.x_yellow), step=1, key="yx")
    y_yellow = c2.number_input("Y", value=int(st.session_state.y_yellow), step=1, key="yy")
    z_yellow = c3.number_input("Z", value=int(st.session_state.z_yellow), step=1, key="yz")

    P_red = np.array([x_red, y_red, z_red], dtype=float)
    P_yellow = np.array([x_yellow, y_yellow, z_yellow], dtype=float)

    distance = np.linalg.norm(P_red - P_yellow)
    P_drop = np.array([P_red[0], P_red[1], P_yellow[2]])
    h = P_yellow[2] * 1000.0
    t_fall = np.sqrt(2 * h / 9.8) if h > 0 else 0.0
    d_ngang_km = (5.0 * t_fall) / 1000.0

    horiz = P_red[:2] - P_yellow[:2]
    norm = np.linalg.norm(horiz)
    if norm > 0:
        release_xy = P_yellow[:2] + (horiz / norm) * min(d_ngang_km, norm)
    else:
        release_xy = P_yellow[:2]
    P_release = np.array([release_xy[0], release_xy[1], P_yellow[2]])

    st.markdown("---")
    st.subheader("📊 Kết quả")
    st.write(f"🔴 Nạn nhân: `{list(P_red)}`")
    st.write(f"🟡 Drone: `{list(P_yellow)}`")
    st.warning(f"📏 Khoảng cách: **{distance:.2f} km**")
    st.success(f"📦 Thả thẳng đứng: `{list(P_drop)}`")
    st.info(f"✈️ Thả khi bay: `({P_release[0]:.2f}, {P_release[1]:.2f}, {P_release[2]})`")

with col_right:
    st.subheader("🌐 Mô hình 3D Oxyz")
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=[P_red[0]], y=[P_red[1]], z=[P_red[2]],
        mode="markers+text", marker=dict(size=11, color="red", symbol="diamond"),
        text=["Nạn nhân"], name="Nạn nhân"
    ))
    fig.add_trace(go.Scatter3d(
        x=[P_yellow[0]], y=[P_yellow[1]], z=[P_yellow[2]],
        mode="markers+text", marker=dict(size=11, color="gold", symbol="square"),
        text=["Drone"], name="Drone"
    ))
    fig.add_trace(go.Scatter3d(
        x=[P_release[0]], y=[P_release[1]], z=[P_release[2]],
        mode="markers+text", marker=dict(size=9, color="purple", symbol="cross"),
        text=["Điểm thả"], name="Điểm thả"
    ))
    fig.add_trace(go.Scatter3d(
        x=[P_yellow[0], P_red[0]], y=[P_yellow[1], P_red[1]], z=[P_yellow[2], P_red[2]],
        mode="lines", line=dict(color="orange", width=4, dash="dash"), name="Khoảng cách"
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, 11], title="Ox (km)"),
            yaxis=dict(range=[0, 11], title="Oy (km)"),
            zaxis=dict(range=[0, 11], title="Oz (km)"),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2)),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    st.plotly_chart(fig, use_container_width=True)

st.caption("💡 Đã sửa lại đúng: ID 0 = gốc, ID 1 = Ox, ID 2 = Oy, ID 3 = Oz")
