from pathlib import Path
import cv2
import cv2.aruco as aruco
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Hộp Oxyz - Teachable Machine Mobile",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"

# ====================== SESSION STATE ======================
defaults = {
    "x_red": 3,
    "y_red": 2,
    "z_red": 3,  # nhà mặc định Z = 3
    "x_yellow": 5,
    "y_yellow": 0,
    "z_yellow": 1,
    "floor_ok": False,
    "wall_ok": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

MAX_X, MAX_Y, MAX_Z = 10, 10, 10
CLASS_NHA = 0
CLASS_TRUC_THANG = 1
MIN_SCORE = 0.60

st.markdown("### 🚁 HỆ THỐNG OXYZ – AI & ARUCO (TỐI ƯU MOBILE)")

col_left, col_right = st.columns([1, 1.2])


# ====================== MODEL TEACHABLE MACHINE ======================
@st.cache_resource(show_spinner="Đang tải model Teachable Machine...")
def load_tm_model():
    import tensorflow as tf
    from tensorflow.keras.layers import DepthwiseConv2D

    class FixedDepthwiseConv2D(DepthwiseConv2D):

        def __init__(self, *args, **kwargs):
            kwargs.pop("groups", None)
            super().__init__(*args, **kwargs)

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={"DepthwiseConv2D": FixedDepthwiseConv2D},
        compile=False,
    )
    labels = []
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            labels.append(parts[1].strip() if len(parts) > 1 else parts[0])
    return model, labels


def locate_class(img_bgr, model, class_index, min_score=MIN_SCORE):
    """Quét ảnh tối ưu tốc độ cho điện thoại (Resize ảnh trước khi cắt ô)."""
    h, w = img_bgr.shape[:2]

    # Giới hạn kích thước ảnh tối đa 640px để chống tràn RAM trên điện thoại/cloud
    max_dim = 640
    if max(h, w) > max_dim:
        scale_input = max_dim / max(h, w)
        img_bgr = cv2.resize(
            img_bgr, (int(w * scale_input), int(h * scale_input))
        )
        h, w = img_bgr.shape[:2]
    else:
        scale_input = 1.0

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    crops, metas = [], []
    for win in (60, 100, 140):
        if win >= min(h, w):
            continue
        stride = max(win // 2, 25)
        for y in range(0, h - win + 1, stride):
            for x in range(0, w - win + 1, stride):
                patch = cv2.resize(
                    rgb[y : y + win, x : x + win], (224, 224)
                )
                crops.append(patch)
                # Quy đổi ngược lại tọa độ về ảnh gốc ban đầu
                orig_x = (x + win / 2) / scale_input
                orig_y = (y + win / 2) / scale_input
                metas.append((orig_x, orig_y))

    if not crops:
        crops.append(cv2.resize(rgb, (224, 224)))
        metas.append((w / (2 * scale_input), h / (2 * scale_input)))

    best_score, best_xy = -1.0, None
    for start in range(0, len(crops), 32):
        batch = np.stack(crops[start : start + 32]).astype(np.float32)
        batch = (batch / 127.5) - 1.0  # Chuẩn hóa của Teachable Machine
        preds = model.predict(batch, verbose=0)
        for i, p in enumerate(preds):
            score = float(p[class_index])
            if score > best_score:
                best_score = score
                best_xy = metas[start + i]

    if best_xy is None or best_score < min_score:
        return None
    return int(best_xy[0]), int(best_xy[1]), best_score


# ====================== ARUCO + TỌA ĐỘ ======================
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


def detect_markers_robust(img):
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids, gray


# ====================== GIAO DIỆN ======================
with col_left:
    st.subheader("📷 Quét ArUco & AI Teachable Machine")
    st.info(
        f"**Thang đo:** Ox = 0→{MAX_X} | Oy = 0→{MAX_Y} | Oz = 0→{MAX_Z}\n\n"
        "**Mốc ID:** 0=(0,0,0) | 1=(10,0,0) | 2=(0,10,0) | 3=(0,0,10)\n\n"
        "Nhận diện Nhà & Trực thăng hoàn toàn bằng file **.h5**."
    )

    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        st.error(
            "Thiếu `keras_model.h5` hoặc `labels.txt` trong cùng thư mục."
        )
        st.stop()

    try:
        tm_model, tm_labels = load_tm_model()
    except Exception as exc:
        st.error(f"Không tải được model: {exc}")
        st.stop()

    tab_tuong, tab_san = st.tabs(["🧱 Mặt tường (Oxz)", "🟫 Mặt sàn (Oxy)"])

    # ---------- TƯỜNG: Trực thăng ----------
    with tab_tuong:
        st.markdown("**Cần ID 0, 1, 3** — Nhận diện **Trực thăng**")
        img_tuong = st.camera_input("Chụp mặt tường Oxz", key="cam_tuong")

        if img_tuong is not None:
            img = cv2.imdecode(
                np.frombuffer(img_tuong.getvalue(), np.uint8), cv2.IMREAD_COLOR
            )
            corners, ids, debug_img = detect_markers_robust(img)
            img_draw = img.copy()

            if ids is not None:
                aruco.drawDetectedMarkers(img_draw, corners, ids)
                centers = get_marker_centers(corners, ids)
                st.success(f"Phát hiện mã ArUco: `{list(ids.flatten())}`")

                with st.spinner("AI đang quét trực thăng..."):
                    found = locate_class(img, tm_model, CLASS_TRUC_THANG)

                if found is not None:
                    cx, cy, score = found
                    cv2.circle(img_draw, (cx, cy), 16, (0, 255, 255), 3)
                    cv2.putText(
                        img_draw,
                        f"Truc Thang {score:.0%}",
                        (cx + 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                    if all(k in centers for k in [0, 1, 3]):
                        pts_src = [centers[0], centers[1], centers[3]]
                        pts_dst = [(0, 0), (MAX_X, 0), (0, MAX_Z)]
                        result = compute_affine_coords(
                            (cx, cy), pts_src, pts_dst, MAX_X, MAX_Z
                        )
                        if result:
                            x_cont, z_cont = result
                            auto_x = int(round(x_cont))
                            auto_z = int(round(z_cont))
                            st.session_state.x_yellow = auto_x
                            st.session_state.z_yellow = auto_z
                            st.session_state.y_yellow = 0
                            st.session_state.wall_ok = True
                            st.success(
                                f"✅ **Đã nhận diện Trực thăng** ({score:.0%})\n\n"
                                f"Tọa độ nguyên: **X = {auto_x}, Z = {auto_z}**"
                            )
                        else:
                            st.warning("Không tính được tọa độ Affine.")
                    else:
                        st.warning("Thiếu một trong các mốc ID 0, 1, 3.")
                else:
                    st.warning(
                        f"Không tìm thấy Trực thăng (Độ tin cậy <"
                        f" {MIN_SCORE:.0%})."
                    )

                st.image(
                    cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )
            else:
                st.error("Không tìm thấy mã ArUco nào trên tường.")
                st.image(
                    debug_img,
                    caption="Ảnh Debug",
                    use_container_width=True,
                    clamp=True,
                )

    # ---------- SÀN: Nhà ----------
    with tab_san:
        st.markdown("**Cần ID 0, 1, 2** — Nhận diện **Nhà**")
        img_san = st.camera_input("Chụp mặt sàn Oxy", key="cam_san")

        if img_san is not None:
            img = cv2.imdecode(
                np.frombuffer(img_san.getvalue(), np.uint8), cv2.IMREAD_COLOR
            )
            corners, ids, debug_img = detect_markers_robust(img)
            img_draw = img.copy()

            if ids is not None:
                aruco.drawDetectedMarkers(img_draw, corners, ids)
                centers = get_marker_centers(corners, ids)
                st.success(f"Phát hiện mã ArUco: `{list(ids.flatten())}`")

                with st.spinner("AI đang quét Nhà..."):
                    found = locate_class(img, tm_model, CLASS_NHA)

                if found is not None:
                    cx, cy, score = found
                    cv2.circle(img_draw, (cx, cy), 16, (0, 0, 255), 3)
                    cv2.putText(
                        img_draw,
                        f"Nha {score:.0%}",
                        (cx + 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                    if all(k in centers for k in [0, 1, 2]):
                        pts_src = [centers[0], centers[1], centers[2]]
                        pts_dst = [(0, 0), (MAX_X, 0), (0, MAX_Y)]
                        result = compute_affine_coords(
                            (cx, cy), pts_src, pts_dst, MAX_X, MAX_Y
                        )
                        if result:
                            x_cont, y_cont = result
                            auto_x = int(round(x_cont))
                            auto_y = int(round(y_cont))
                            st.session_state.x_red = auto_x
                            st.session_state.y_red = auto_y
                            st.session_state.floor_ok = True
                            st.success(
                                f"✅ **Đã nhận diện Nhà** ({score:.0%})\n\n"
                                f"Tọa độ nguyên: **X = {auto_x}, Y = {auto_y}**"
                                f" (Z giữ = 3)"
                            )
                        else:
                            st.warning("Không tính được tọa độ Affine.")
                    else:
                        st.warning("Thiếu một trong các mốc ID 0, 1, 2.")
                else:
                    st.warning(
                        f"Không tìm thấy Nhà (Độ tin cậy < {MIN_SCORE:.0%})."
                    )

                st.image(
                    cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )
            else:
                st.error("Không tìm thấy mã ArUco nào dưới sàn.")
                st.image(
                    debug_img,
                    caption="Ảnh Debug",
                    use_container_width=True,
                    clamp=True,
                )

    # ---------- HIỆU CHỈNH THỦ CÔNG ----------
    st.markdown("---")
    st.markdown("#### ⚙️ Điều chỉnh tọa độ")

    c1, c2, c3 = st.columns(3)
    x_red = c1.number_input(
        "X Nhà",
        min_value=0,
        max_value=MAX_X,
        value=int(st.session_state.x_red),
        step=1,
        key="rx",
    )
    y_red = c2.number_input(
        "Y Nhà",
        min_value=0,
        max_value=MAX_Y,
        value=int(st.session_state.y_red),
        step=1,
        key="ry",
    )
    z_red = c3.number_input(
        "Z Nhà",
        min_value=0,
        max_value=MAX_Z,
        value=int(st.session_state.z_red),
        step=1,
        key="rz",
    )

    c1, c2, c3 = st.columns(3)
    x_yellow = c1.number_input(
        "X Trực thăng",
        min_value=0,
        max_value=MAX_X,
        value=int(st.session_state.x_yellow),
        step=1,
        key="yx",
    )
    y_yellow = c2.number_input(
        "Y Trực thăng",
        min_value=0,
        max_value=MAX_Y,
        value=int(st.session_state.y_yellow),
        step=1,
        key="yy",
    )
    z_yellow = c3.number_input(
        "Z Trực thăng",
        min_value=0,
        max_value=MAX_Z,
        value=int(st.session_state.z_yellow),
        step=1,
        key="yz",
    )

    st.session_state.x_red, st.session_state.y_red, st.session_state.z_red = (
        x_red,
        y_red,
        z_red,
    )
    (
        st.session_state.x_yellow,
        st.session_state.y_yellow,
        st.session_state.z_yellow,
    ) = (x_yellow, y_yellow, z_yellow)

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
    st.subheader("📊 Kết quả Phân tích Vật lý")
    st.write(f"🏠 **Nhà:** `{list(P_red)}`")
    st.write(f"🚁 **Trực thăng:** `{list(P_yellow)}`")
    st.warning(f"📏 Khoảng cách không gian: **{distance:.2f}** đơn vị")
    st.success(f"📦 Thả thẳng đứng: `{list(P_drop)}`")
    st.info(
        f"✈️ Thả khi bay: `({P_release[0]:.2f}, {P_release[1]:.2f},"
        f" {P_release[2]})`"
    )

with col_right:
    st.subheader("🌐 Mô hình 3D Oxyz")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=[P_red[0]],
            y=[P_red[1]],
            z=[P_red[2]],
            mode="markers+text",
            marker=dict(size=12, color="red", symbol="diamond"),
            text=["Nhà"],
            name="Nhà",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[P_yellow[0]],
            y=[P_yellow[1]],
            z=[P_yellow[2]],
            mode="markers+text",
            marker=dict(size=12, color="gold", symbol="square"),
            text=["Trực thăng"],
            name="Trực thăng",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[P_release[0]],
            y=[P_release[1]],
            z=[P_release[2]],
            mode="markers+text",
            marker=dict(size=10, color="purple", symbol="cross"),
            text=["Điểm thả"],
            name="Điểm thả",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[P_yellow[0], P_red[0]],
            y=[P_yellow[1], P_red[1]],
            z=[P_yellow[2], P_red[2]],
            mode="lines",
            line=dict(color="orange", width=5, dash="dash"),
            name="Khoảng cách",
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, MAX_X + 1], title=f"Ox (0→{MAX_X})"),
            yaxis=dict(range=[0, MAX_Y + 1], title=f"Oy (0→{MAX_Y})"),
            zaxis=dict(range=[0, MAX_Z + 1], title=f"Oz (0→{MAX_Z})"),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.3)),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    st.plotly_chart(fig, use_container_width=True)
