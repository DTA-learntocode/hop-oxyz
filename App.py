import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title=" -Hộp Oxyz- ",
    layout="wide",
)

# ====================== SESSION STATE ======================
defaults = {
    "x_red": 3, "y_red": 2, "z_red": 3,
    "x_yellow": 5, "y_yellow": 0, "z_yellow": 1,
    "floor_ok": False,
    "wall_ok": False,
    "points": {"O": None, "X": None, "Y": None, "Z": None, "B": None},
    "current_point": "O",
    "img_rgb": None,
    "img_bgr": None,
    "last_auto_house": None,
    "last_auto_heli": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

MAX_X, MAX_Y, MAX_Z = 10, 10, 10

st.markdown("### HỆ THỐNG OXYZ CỨU TRỢ")

col_left, col_right = st.columns([1.15, 1])

# ====================== HÀM TIỆN ÍCH ======================
def detect_color_centroid(img, color="yellow"):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if color == "yellow":
        mask = cv2.inRange(hsv, np.array([5, 100, 100]), np.array([25, 255, 255]))
        min_area = 300
    else:
        mask1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([7, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 100, 80]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        min_area = 100

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

def compute_floor_homography(pts):
    src = np.float32([pts["O"], pts["X"], pts["Y"], pts["B"]])
    dst = np.float32([[0, 0], [MAX_X, 0], [0, MAX_Y], [MAX_X, MAX_Y]])
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H

def compute_wall_affine(pts):
    src = np.float32([pts["O"], pts["X"], pts["Z"]])
    dst = np.float32([[0, 0], [MAX_X, 0], [0, MAX_Z]])
    return cv2.getAffineTransform(src, dst)

def apply_homography(H, pin):
    pt = np.array([[[pin[0], pin[1]]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)[0][0]
    return float(np.clip(out[0], 0, MAX_X)), float(np.clip(out[1], 0, MAX_Y))

def apply_affine(M, pin):
    pt = np.array([[[pin[0], pin[1]]]], dtype=np.float32)
    out = cv2.transform(pt, M)[0][0]
    return float(np.clip(out[0], 0, MAX_X)), float(np.clip(out[1], 0, MAX_Z))

# ====================== CỘT TRÁI ======================
with col_left:
    st.subheader("1. Chụp / tải ảnh mô hình")
    st.info(
        "**Cách chọn điểm:**\n"
        "1. Chọn điểm đang muốn gán (O → X → Y → Z → B)\n"
        "2. **Bấm chuột** lên ảnh → điểm được lưu ngay\n"
        "3. Có thể xóa điểm bằng nút bên dưới\n\n"
        "**nhà** = cam · **trực thăng** = đỏ"
    )

    img_file = st.camera_input("Chụp ảnh mô hình", key="cam") or \
               st.file_uploader("Hoặc tải ảnh lên", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        st.session_state.img_bgr = img_bgr
        st.session_state.img_rgb = img_rgb

        h_orig, w_orig = img_rgb.shape[:2]
        st.caption(f"Kích thước ảnh gốc: **{w_orig} × {h_orig}** px")

        st.markdown("---")
        st.subheader("2. Chọn 5 điểm tham chiếu (trực thăng thuộc (Oxz)")

        point_names = ["O", "X", "Y", "Z", "B"]
        labels = {
            "O": "O – gốc (0,0,0)",
            "X": "X – (10,0,0)",
            "Y": "Y – (0,10,0)",
            "Z": "Z – (0,0,10)",
            "B": "B – (10,10,0)",
        }

        current = st.radio(
            "Đang chọn điểm:",
            point_names,
            format_func=lambda x: labels[x],
            horizontal=True,
            key="radio_point"
        )
        st.session_state.current_point = current

        # ---------- CANVAS ----------
        max_display_width = 780
        scale = min(1.0, max_display_width / w_orig)
        disp_w = int(w_orig * scale)
        disp_h = int(h_orig * scale)

        img_display = cv2.resize(img_rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        bg_image = Image.fromarray(img_display)

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=bg_image,
            update_streamlit=True,
            height=disp_h,
            width=disp_w,
            drawing_mode="point",
            point_display_radius=6,
            key=f"canvas_{current}",
        )

        # ---------- XỬ LÝ LƯU ĐIỂM ----------
        if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                last_obj = objects[-1]
                obj_type = str(last_obj.get("type", "")).lower()

                if obj_type == "circle":
                    radius = last_obj.get("radius", 0)
                    x_disp = last_obj["left"] + radius
                    y_disp = last_obj["top"] + radius

                    u = int(round(x_disp / scale))
                    v = int(round(y_disp / scale))
                    u = max(0, min(u, w_orig - 1))
                    v = max(0, min(v, h_orig - 1))

                    if st.session_state.points.get(current) != (u, v):
                        st.session_state.points[current] = (u, v)
                        st.success(f"✅ Đã lưu **{current}** = ({u}, {v})")
                        st.rerun()

        st.caption("👉 Bấm chuột trực tiếp lên ảnh để chọn điểm")

        # Trạng thái 5 điểm
        st.markdown("##### Trạng thái 5 điểm:")
        cols = st.columns(5)
        for i, name in enumerate(point_names):
            with cols[i]:
                pt = st.session_state.points[name]
                if pt:
                    st.success(f"**{name}**\n`{pt}`")
                else:
                    st.warning(f"**{name}**\nchưa chọn")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Xóa điểm hiện tại"):
                st.session_state.points[current] = None
                st.rerun()
        with col_btn2:
            if st.button("Xóa tất cả điểm"):
                st.session_state.points = {"O": None, "X": None, "Y": None, "Z": None, "B": None}
                st.rerun()

        # ---------- PHÁT HIỆN MÀU ----------
        all_set = all(st.session_state.points[p] is not None for p in point_names)

        if all_set:
            st.markdown("---")
            st.subheader("3. Kết quả phát hiện")

            pts = st.session_state.points
            img = st.session_state.img_bgr

            pin_house = detect_color_centroid(img, "yellow")
            pin_heli  = detect_color_centroid(img, "red")

            img_draw = img.copy()
            colors_bgr = {
                "O": (0, 0, 255), "X": (0, 255, 0), "Y": (255, 255, 0),
                "Z": (255, 0, 255), "B": (0, 255, 255)
            }

            if pin_house:
                cv2.circle(img_draw, pin_house, 14, (0, 165, 255), 3)
                cv2.putText(img_draw, "nha", (pin_house[0]+12, pin_house[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            if pin_heli:
                cv2.circle(img_draw, pin_heli, 14, (0, 0, 255), 3)
                cv2.putText(img_draw, "truc thang", (pin_heli[0]+12, pin_heli[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            for name, pt in pts.items():
                cv2.circle(img_draw, pt, 10, colors_bgr[name], -1)
                cv2.putText(img_draw, name, (pt[0]+8, pt[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors_bgr[name], 2)

            st.image(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), use_container_width=True)

            try:
                H_floor = compute_floor_homography(pts)
                M_wall  = compute_wall_affine(pts)

                if pin_house:
                    xh, yh = apply_homography(H_floor, pin_house)
                    auto_x, auto_y = int(round(xh)), int(round(yh))
                    new_auto = (auto_x, auto_y, 3)

                    if st.session_state.last_auto_house != new_auto:
                        st.session_state.x_red = auto_x
                        st.session_state.y_red = auto_y
                        st.session_state.z_red = 3
                        st.session_state.rx = auto_x
                        st.session_state.ry = auto_y
                        st.session_state.rz = 3
                        st.session_state.last_auto_house = new_auto

                    st.session_state.floor_ok = True
                    st.success(f"**Nhà (cam)** → **({auto_x}, {auto_y}, 3)**")
                else:
                    st.warning("Không thấy nhà (cam)")

                if pin_heli:
                    xhe, zhe = apply_affine(M_wall, pin_heli)
                    auto_x, auto_z = int(round(xhe)), int(round(zhe))
                    new_auto = (auto_x, 0, auto_z)

                    if st.session_state.last_auto_heli != new_auto:
                        st.session_state.x_yellow = auto_x
                        st.session_state.y_yellow = 0
                        st.session_state.z_yellow = auto_z
                        st.session_state.yx = auto_x
                        st.session_state.yy = 0
                        st.session_state.yz = auto_z
                        st.session_state.last_auto_heli = new_auto

                    st.session_state.wall_ok = True
                    st.success(f"**Trực thăng (đỏ)** → **({auto_x}, 0, {auto_z})**")
                else:
                    st.warning("Không thấy trực thăng (đỏ)")

            except Exception as e:
                st.error(f"Lỗi tính toán: {e}")
        else:
            st.warning("Hãy chọn đủ 5 điểm trước.")

# ====================== CỘT PHẢI ======================
with col_right:
    st.subheader("Tọa độ hiện tại (có thể chỉnh tay)")

    if st.session_state.floor_ok and st.session_state.wall_ok:
        st.success("Đã có đủ tọa độ cả 2 vật")
    elif st.session_state.floor_ok:
        st.info("Đã có tọa độ nhà")
    elif st.session_state.wall_ok:
        st.info("Đã có tọa độ trực thăng")

    st.markdown("**nhà** (cam) — Z mặc định = 3")
    c1, c2, c3 = st.columns(3)
    x_red = c1.number_input("X", 0, MAX_X, key="rx")
    y_red = c2.number_input("Y", 0, MAX_Y, key="ry")
    z_red = c3.number_input("Z", 0, MAX_Z, key="rz")

    st.markdown("**trực thăng** (đỏ)")
    c1, c2, c3 = st.columns(3)
    x_yellow = c1.number_input("X", 0, MAX_X, key="yx")
    y_yellow = c2.number_input("Y", 0, MAX_Y, key="yy")
    z_yellow = c3.number_input("Z", 0, MAX_Z, key="yz")

    P_red = np.array([x_red, y_red, z_red], dtype=float)
    P_yellow = np.array([x_yellow, y_yellow, z_yellow], dtype=float)

    distance = np.linalg.norm(P_red - P_yellow)
    P_drop = np.array([P_red[0], P_red[1], P_yellow[2]])
    
    h_meters = P_yellow[2] * 1000.0
    t_fall = np.sqrt(2 * h_meters / 9.8) if h_meters > 0 else 0.0
    d_ngang_km = (5.0 * t_fall) / 1000.0
    d_ngang_donvi = d_ngang_km

    horiz = P_red[:2] - P_yellow[:2]
    norm = np.linalg.norm(horiz)
    
    if norm > 0:
        release_xy = P_yellow[:2] + (horiz / norm) * min(d_ngang_donvi, norm)
    else:
        release_xy = P_yellow[:2]
    
    P_release = np.array([release_xy[0], release_xy[1], P_yellow[2]])

    st.markdown("---")
    st.subheader("Kết quả tính toán chi tiết")

    st.write(f"**nhà:** `{list(P_red)}`")
    st.write(f"**trực thăng:** `{list(P_yellow)}`")
    st.warning(f"**Khoảng cách nhà đến trực thăng:** `{distance:.2f}` km")

    st.markdown("##### 1. Thả thẳng đứng")
    st.success(f"Tọa độ điểm thả: `{list(P_drop)}`")
    st.caption("Cùng X,Y với nhà – cùng độ cao Z với trực thăng")

    st.markdown("##### 2. Thả khi đang bay")
    st.info(f"Tọa độ điểm thả: `({P_release[0]:.2f}, {P_release[1]:.2f}, {P_release[2]})`")
    
    with st.expander("Chi tiết tính toán thả khi bay"):
        st.write(f"- Chiều cao rơi: **{h_meters:.1f} m** (Z = {P_yellow[2]} × 1000)")
        st.write(f"- Thời gian rơi tự do: **{t_fall:.2f} giây**")
        st.write(f"- Vận tốc bay ngang giả định: **5 m/s**")
        st.write(f"- Quãng đường ngang cần có: **{d_ngang_km:.3f} km**")
        st.write(f"- Khoảng cách ngang hiện tại: **{norm:.2f} km**")

    st.markdown("---")
    st.subheader("Mô hình 3D")

    fig3d = go.Figure()

    fig3d.add_trace(go.Scatter3d(
        x=[P_red[0]], y=[P_red[1]], z=[P_red[2]],
        mode="markers+text", marker=dict(size=12, color="orange", symbol="diamond"),
        text=["nhà"], name="nhà",
    ))
    fig3d.add_trace(go.Scatter3d(
        x=[P_yellow[0]], y=[P_yellow[1]], z=[P_yellow[2]],
        mode="markers+text", marker=dict(size=12, color="red", symbol="square"),
        text=["trực thăng"], name="trực thăng",
    ))
    fig3d.add_trace(go.Scatter3d(
        x=[P_drop[0]], y=[P_drop[1]], z=[P_drop[2]],
        mode="markers+text", marker=dict(size=9, color="lime", symbol="circle"),
        text=["thả thẳng"], name="thả thẳng đứng",
    ))
    fig3d.add_trace(go.Scatter3d(
        x=[P_release[0]], y=[P_release[1]], z=[P_release[2]],
        mode="markers+text", marker=dict(size=10, color="purple", symbol="cross"),
        text=["thả khi bay"], name="thả khi bay",
    ))
    fig3d.add_trace(go.Scatter3d(
        x=[P_yellow[0], P_red[0]], y=[P_yellow[1], P_red[1]], z=[P_yellow[2], P_red[2]],
        mode="lines", line=dict(color="orange", width=5, dash="dash"), name="khoảng cách",
    ))

    fig3d.update_layout(
        scene=dict(
            xaxis=dict(range=[0, MAX_X+1], title="Ox"),
            yaxis=dict(range=[0, MAX_Y+1], title="Oy"),
            zaxis=dict(range=[0, MAX_Z+1], title="Oz"),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.3)),
        ),
        margin=dict(l=0, r=0, b=0, t=20),
        height=400,
    )
    st.plotly_chart(fig3d, use_container_width=True)
