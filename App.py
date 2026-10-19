import cv2
import cv2.aruco as aruco
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Hộp Oxyz - Ox=6 | Oy=3 | Oz=2",
    layout="wide",
)

# ====================== SESSION STATE ======================
defaults = {
    "x_red": 2, "y_red": 1, "z_red": 0,
    "x_yellow": 3, "y_yellow": 0, "z_yellow": 1,
    "floor_ok": False,
    "wall_ok": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

MAX_X, MAX_Y, MAX_Z = 6, 3, 2

st.markdown("### 🚁 HỆ THỐNG OXYZ – Ox = 6 ô | Oy = 3 ô | Oz = 2 ô")

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

def compute_affine_coords(pin_pt, pts_src, pts_dst, max_x, max_y):
    if len(pts_src) < 3:
        return None
    M = cv2.getAffineTransform(np.float32(pts_src), np.float32(pts_dst))
    pt = np.array([[pin_pt]], dtype=np.float32)
    transformed = cv2.transform(pt, M)[0][0]
    x = float(np.clip(transformed[0], 0, max_x))
    y = float(np.clip(transformed[1], 0, max_y))
    return x, y

def detect_color_centroid(img, color="yellow"):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color == "yellow":
        mask = cv2.inRange(hsv, np.array([15, 70, 70]), np.array([40, 255, 255]))
    else:
        mask1 = cv2.inRange(hsv, np.array([0, 80, 60]), np.array([12, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 80, 60]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 60:
        return None
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

def preprocess_for_aruco(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

def create_detector():
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 35
    parameters.adaptiveThreshWinSizeStep = 6
    parameters.adaptiveThreshConstant = 5
    parameters.minMarkerPerimeterRate = 0.01
    parameters.maxMarkerPerimeterRate = 5.0
    parameters.polygonalApproxAccuracyRate = 0.07
    parameters.minCornerDistanceRate = 0.03
    parameters.minDistanceToBorder = 1
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 50
    parameters.cornerRefinementMinAccuracy = 0.05
    parameters.detectInvertedMarker = True
    return aruco.ArucoDetector(aruco_dict, parameters)

def detect_markers_robust(img):
    detector = create_detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = preprocess_for_aruco(img)
    corners1, ids1, _ = detector.detectMarkers(gray)
    corners2, ids2, _ = detector.detectMarkers(processed)
    if ids1 is not None and (ids2 is None or len(ids1) >= len(ids2)):
        return corners1, ids1, gray
    return corners2, ids2, processed

# ====================== GIAO DIỆN BÊN TRÁI ======================
with col_left:
    st.subheader("📷 Quét ArUco & Tự động cập nhật tọa độ")
    st.info(
        f"**Thang đo:** Ox = 0→{MAX_X} | Oy = 0→{MAX_Y} | Oz = 0→{MAX_Z}\n\n"
        "**ID:** 0=(0,0,0) | 1=(6,0,0) | 2=(0,3,0) | 3=(0,0,2)"
    )

    tab_tuong, tab_san = st.tabs(["🧱 Mặt Tường (Oxz)", "🟫 Mặt Sàn (Oxy)"])

    # ---------- TƯỜNG Oxz (Trực thăng - Vàng) ----------
    with tab_tuong:
        st.markdown("**Cần ID 0, 1, 3**")
        img_tuong = st.camera_input("Chụp mặt tường Oxz", key="cam_tuong")

        if img_tuong is not None:
            img = cv2.imdecode(np.frombuffer(img_tuong.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            corners, ids, debug_img = detect_markers_robust(img)
            img_draw = img.copy()

            if ids is not None:
                aruco.drawDetectedMarkers(img_draw, corners, ids)
                centers = get_marker_centers(corners, ids)
                st.success(f"🎯 Phát hiện: `{list(ids.flatten())}`")

                pin = detect_color_centroid(img, "yellow")
                if pin is not None:
                    cv2.circle(img_draw, pin, 12, (0, 255, 255), 3)
                    cv2.putText(img_draw, "Truc thang", (pin[0]+15, pin[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    if all(k in centers for k in [0, 1, 3]):
                        pts_src = [centers[0], centers[1], centers[3]]
                        pts_dst = [(0, 0), (MAX_X, 0), (0, MAX_Z)]
                        result = compute_affine_coords(pin, pts_src, pts_dst, MAX_X, MAX_Z)

                        if result:
                            x_cont, z_cont = result
                            auto_x = int(round(x_cont))
                            auto_z = int(round(z_cont))

                            st.session_state.x_yellow = auto_x
                            st.session_state.z_yellow = auto_z
                            st.session_state.y_yellow = 0
                            st.session_state.yx = auto_x
                            st.session_state.yz = auto_z
                            st.session_state.yy = 0
                            st.session_state.wall_ok = True

                            st.success(
                                f"🤖 **Đã cập nhật Trực thăng**\n"
                                f"Liên tục: X ≈ {x_cont:.2f}, Z ≈ {z_cont:.2f}\n"
                                f"→ **Số nguyên: X = {auto_x}, Z = {auto_z}**"
                            )
                        else:
                            st.warning("Không tính được Affine.")
                    else:
                        st.warning("⚠️ Chưa đủ ID 0, 1, 3.")
                else:
                    st.warning("⚠️ Không thấy Trực thăng (màu vàng).")

                st.image(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.error("❌ Không tìm thấy mã ArUco.")
                st.image(debug_img, caption="Ảnh debug", use_container_width=True, clamp=True)

    # ---------- SÀN Oxy (Nhà - Đỏ) ----------
    with tab_san:
        st.markdown("**Cần ID 0, 1, 2**")
        img_san = st.camera_input("Chụp mặt sàn Oxy", key="cam_san")

        if img_san is not None:
            img = cv2.imdecode(np.frombuffer(img_san.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            corners, ids, debug_img = detect_markers_robust(img)
            img_draw = img.copy()

            if ids is not None:
                aruco.drawDetectedMarkers(img_draw, corners, ids)
                centers = get_marker_centers(corners, ids)
                st.success(f"🎯 Phát hiện: `{list(ids.flatten())}`")

                pin = detect_color_centroid(img, "red")
                if pin is not None:
                    cv2.circle(img_draw, pin, 12, (0, 0, 255), 3)
                    cv2.putText(img_draw, "Nha", (pin[0]+15, pin[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if all(k in centers for k in [0, 1, 2]):
                        pts_src = [centers[0], centers[1], centers[2]]
                        pts_dst = [(0, 0), (MAX_X, 0), (0, MAX_Y)]
                        result = compute_affine_coords(pin, pts_src, pts_dst, MAX_X, MAX_Y)

                        if result:
                            x_cont, y_cont = result
                            auto_x = int(round(x_cont))
                            auto_y = int(round(y_cont))

                            st.session_state.x_red = auto_x
                            st.session_state.y_red = auto_y
                            st.session_state.z_red = 0
                            st.session_state.rx = auto_x
                            st.session_state.ry = auto_y
                            st.session_state.rz = 0
                            st.session_state.floor_ok = True

                            st.success(
                                f"🤖 **Đã cập nhật Nhà**\n"
                                f"Liên tục: X ≈ {x_cont:.2f}, Y ≈ {y_cont:.2f}\n"
                                f"→ **Số nguyên: X = {auto_x}, Y = {auto_y}**"
                            )
                        else:
                            st.warning("Không tính được Affine.")
                    else:
                        st.warning("⚠️ Chưa đủ ID 0, 1, 2.")
                else:
                    st.warning("⚠️ Không thấy Nhà (màu đỏ).")

                st.image(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.error("❌ Không tìm thấy mã ArUco.")
                st.image(debug_img, caption="Ảnh debug", use_container_width=True, clamp=True)

    # ---------- HIỆU CHỈNH ----------
    st.markdown("---")
    st.markdown("#### ⚙️ Tọa độ hiện tại (tự cập nhật sau khi chụp)")

    if st.session_state.floor_ok and st.session_state.wall_ok:
        st.success("✅ Đã quét đủ cả 2 mặt → tọa độ đã sẵn sàng tính toán")
    elif st.session_state.floor_ok:
        st.info("🏠 Đã có tọa độ Nhà (sàn)")
    elif st.session_state.wall_ok:
        st.info("🚁 Đã có tọa độ Trực thăng (tường)")

    st.markdown("🏠 **Nhà (màu đỏ)**")
    c1, c2, c3 = st.columns(3)
    x_red = c1.number_input("X (0-6)", min_value=0, max_value=MAX_X, step=1, key="rx")
    y_red = c2.number_input("Y (0-3)", min_value=0, max_value=MAX_Y, step=1, key="ry")
    z_red = c3.number_input("Z (0-2)", min_value=0, max_value=MAX_Z, step=1, key="rz")

    st.markdown("🚁 **Trực thăng (màu vàng)**")
    c1, c2, c3 = st.columns(3)
    x_yellow = c1.number_input("X (0-6)", min_value=0, max_value=MAX_X, step=1, key="yx")
    y_yellow = c2.number_input("Y (0-3)", min_value=0, max_value=MAX_Y, step=1, key="yy")
    z_yellow = c3.number_input("Z (0-2)", min_value=0, max_value=MAX_Z, step=1, key="yz")

    st.session_state.x_red = x_red
    st.session_state.y_red = y_red
    st.session_state.z_red = z_red
    st.session_state.x_yellow = x_yellow
    st.session_state.y_yellow = y_yellow
    st.session_state.z_yellow = z_yellow

    P_red = np.array([x_red, y_red, z_red], dtype=float)
    P_yellow = np.array([x_yellow, y_yellow, z_yellow], dtype=float)

    distance = np.linalg.norm(P_red - P_yellow)
    P_drop = np.array([P_red[0], P_red[1], P_yellow[2]])
    h_meters = P_yellow[2] * 1000.0
    t_fall = np.sqrt(2 * h_meters / 9.8) if h_meters > 0 else 0.0
    d_ngang_km = (5.0 * t_fall) / 1000.0

    horiz = P_red[:2] - P_yellow[:2]
    norm = np.linalg.norm(horiz)
    if norm > 0:
        release_xy = P_yellow[:2] + (horiz / norm) * min(d_ngang_km, norm)
    else:
        release_xy = P_yellow[:2]
    P_release = np.array([release_xy[0], release_xy[1], P_yellow[2]])

    st.markdown("---")
    st.subheader("📊 Kết quả tính toán")
    st.write(f"🏠 **Nhà:** `{list(P_red)}`")
    st.write(f"🚁 **Trực thăng:** `{list(P_yellow)}`")
    st.warning(f"📏 Khoảng cách: **{distance:.2f} đơn vị**")
    st.success(f"📦 Thả thẳng đứng: `{list(P_drop)}`")
    st.info(f"✈️ Thả khi bay: `({P_release[0]:.2f}, {P_release[1]:.2f}, {P_release[2]})`")

# ====================== CỘT PHẢI ======================
with col_right:
    st.subheader("🌐 Mô hình 3D")
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=[P_red[0]], y=[P_red[1]], z=[P_red[2]],
        mode="markers+text", marker=dict(size=12, color="red", symbol="diamond"),
        text=["Nhà"], name="Nhà"
    ))
    fig.add_trace(go.Scatter3d(
        x=[P_yellow[0]], y=[P_yellow[1]], z=[P_yellow[2]],
        mode="markers+text", marker=dict(size=12, color="gold", symbol="square"),
        text=["Trực thăng"], name="Trực thăng"
    ))
    fig.add_trace(go.Scatter3d(
        x=[P_release[0]], y=[P_release[1]], z=[P_release[2]],
        mode="markers+text", marker=dict(size=10, color="purple", symbol="cross"),
        text=["Điểm thả"], name="Điểm thả"
    ))
    fig.add_trace(go.Scatter3d(
        x=[P_yellow[0], P_red[0]], y=[P_yellow[1], P_red[1]], z=[P_yellow[2], P_red[2]],
        mode="lines", line=dict(color="orange", width=5, dash="dash"), name="Khoảng cách"
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, MAX_X + 1], title=f"Ox (0→{MAX_X})"),
            yaxis=dict(range=[0, MAX_Y + 1], title=f"Oy (0→{MAX_Y})"),
            zaxis=dict(range=[0, MAX_Z + 1], title=f"Oz (0→{MAX_Z})"),
            camera=dict(eye=dict(x=1.8, y=-1.8, z=1.4)),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    st.plotly_chart(fig, use_container_width=True)

st.caption(f"💡 Ox={MAX_X} ô | Oy={MAX_Y} ô | Oz={MAX_Z} ô  •  Nhà (đỏ) & Trực thăng (vàng)")
