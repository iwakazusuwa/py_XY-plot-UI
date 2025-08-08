


# +
import streamlit as st
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from PIL import Image, ImageDraw
from collections import defaultdict
import matplotlib.pyplot as plt

# -----------------------------
# 🔧 1. 集計関数（ルール前後 & 座標出力対応）
# -----------------------------
def calculate_area_flags(resp_df, polygons, apply_rule=True):
    total_ids = len(resp_df)

    # 各回答者 × エリアのlike/dislike集計
    per_respondent_area = defaultdict(lambda: defaultdict(lambda: {"like": 0, "dislike": 0}))
    area_flags = {area: {"like": 0, "dislike": 0} for area in polygons}

    for idx, row in resp_df.iterrows():
        rid = row.get("Respondent ID", idx)
        for i in range(1, 3):
            lx, ly = row.get(f"like{i}_x"), row.get(f"like{i}_y")
            dx, dy = row.get(f"dislike{i}_x"), row.get(f"dislike{i}_y")
            if pd.notnull(lx) and pd.notnull(ly):
                for area_name, poly in polygons.items():
                    if poly.contains(Point(lx, ly)):
                        per_respondent_area[rid][area_name]["like"] += 1
                        break
            if pd.notnull(dx) and pd.notnull(dy):
                for area_name, poly in polygons.items():
                    if poly.contains(Point(dx, dy)):
                        per_respondent_area[rid][area_name]["dislike"] += 1
                        break

    for rid, area_data in per_respondent_area.items():
        for area, counts in area_data.items():
            like = counts["like"]
            dislike = counts["dislike"]
            if apply_rule:
                if like > 0 and dislike > 0:
                    continue
                elif like > 0:
                    area_flags[area]["like"] += like
                elif dislike > 0:
                    area_flags[area]["dislike"] += dislike
            else:
                area_flags[area]["like"] += like
                area_flags[area]["dislike"] += dislike

    area_summary = []
    for area, counts in area_flags.items():
        like = counts["like"]
        dislike = counts["dislike"]
        none = total_ids - like - dislike
        area_summary.append({
            "area": area,
            "like": like,
            "dislike": dislike,
            "none": none,
            "total": total_ids,
            "like_ratio": like / total_ids if total_ids else 0,
            "dislike_ratio": dislike / total_ids if total_ids else 0,
            "none_ratio": none / total_ids if total_ids else 0
        })

    area_df = pd.DataFrame(area_summary)

    # XY抽出（ルール適用後のみ）
    coord_rows = []
    for idx, row in resp_df.iterrows():
        rid = row.get("Respondent ID", idx)
        like_areas = {}
        dislike_areas = {}

        for i in range(1, 3):
            lx, ly = row.get(f"like{i}_x"), row.get(f"like{i}_y")
            dx, dy = row.get(f"dislike{i}_x"), row.get(f"dislike{i}_y")

            if pd.notnull(lx) and pd.notnull(ly):
                for area, poly in polygons.items():
                    if poly.contains(Point(lx, ly)):
                        like_areas[i] = area
                        break
            if pd.notnull(dx) and pd.notnull(dy):
                for area, poly in polygons.items():
                    if poly.contains(Point(dx, dy)):
                        dislike_areas[i] = area
                        break

        canceled_areas = set(like_areas.values()) & set(dislike_areas.values())

        row_dict = {"Respondent ID": rid}
        for i in range(1, 3):
            if i in like_areas and like_areas[i] not in canceled_areas:
                row_dict[f"like{i}_x"] = row.get(f"like{i}_x")
                row_dict[f"like{i}_y"] = row.get(f"like{i}_y")
            else:
                row_dict[f"like{i}_x"] = np.nan
                row_dict[f"like{i}_y"] = np.nan

            if i in dislike_areas and dislike_areas[i] not in canceled_areas:
                row_dict[f"dislike{i}_x"] = row.get(f"dislike{i}_x")
                row_dict[f"dislike{i}_y"] = row.get(f"dislike{i}_y")
            else:
                row_dict[f"dislike{i}_x"] = np.nan
                row_dict[f"dislike{i}_y"] = np.nan

        coord_rows.append(row_dict)

    coord_df = pd.DataFrame(coord_rows)

    return area_df, coord_df

# -----------------------------
# 🎯 相殺前の座標抽出関数
# -----------------------------
def extract_all_touch_coords(resp_df):
    rows = []
    for idx, row in resp_df.iterrows():
        rid = row.get("Respondent ID", idx)
        row_dict = {"Respondent ID": rid}
        for i in range(1, 3):
            row_dict[f"like{i}_x"] = row.get(f"like{i}_x")
            row_dict[f"like{i}_y"] = row.get(f"like{i}_y")
            row_dict[f"dislike{i}_x"] = row.get(f"dislike{i}_x")
            row_dict[f"dislike{i}_y"] = row.get(f"dislike{i}_y")
        rows.append(row_dict)
    return pd.DataFrame(rows)

# -----------------------------
# 🖼️ 座標を画像に描画
# -----------------------------
def draw_points_on_image(img, df, color_like=(255, 0, 0), color_dislike=(0, 0, 255), radius=10):
    draw = ImageDraw.Draw(img)
    for _, row in df.iterrows():
        for i in range(1, 3):
            lx = row.get(f"like{i}_x")
            ly = row.get(f"like{i}_y")
            dx = row.get(f"dislike{i}_x")
            dy = row.get(f"dislike{i}_y")

            if pd.notna(lx) and pd.notna(ly):
                draw.ellipse((lx - radius, ly - radius, lx + radius, ly + radius), fill=color_like)
            if pd.notna(dx) and pd.notna(dy):
                draw.ellipse((dx - radius, dy - radius, dx + radius, dy + radius), fill=color_dislike)
    return img

# -----------------------------
# 🖥️ Streamlit アプリ本体
# -----------------------------
st.title("画像エリアの好き嫌い集計・可視化ツール")

mode = st.radio("処理を選択してください", ["データ集計", "画像へのプロット"])

if mode == "データ集計":
    st.header("データアップロード")
    area_file = st.file_uploader("エリア定義CSV（area.csv）", type="csv")
    resp_file = st.file_uploader("回答データCSV（response.csv）", type="csv")

    if area_file and resp_file:
        area_df = pd.read_csv(area_file)
        resp_df = pd.read_csv(resp_file)

        polygons = {}
        for name, group in area_df.groupby("name"):
            points = [(x, y) for x, y in zip(group["x"], group["y"])]
            polygons[name] = Polygon(points)

        st.subheader("ルール適用前の集計")
        before_df, _ = calculate_area_flags(resp_df, polygons, apply_rule=False)
        st.dataframe(before_df)

        st.subheader("ルール適用後の集計")
        after_df, coord_df = calculate_area_flags(resp_df, polygons, apply_rule=True)
        st.dataframe(after_df)

        st.subheader("ルール適用前後の差分")
        diff_df = after_df[["area", "like", "dislike"]].copy()
        diff_df["like_diff"] = after_df["like"] - before_df["like"]
        diff_df["dislike_diff"] = after_df["dislike"] - before_df["dislike"]
        st.dataframe(diff_df[["area", "like_diff", "dislike_diff"]])

        # ここに散布図を追加
        st.subheader("ルール適用後の Like / Dislike 散布図")
        fig, ax = plt.subplots()
        ax.scatter(after_df["like"], after_df["dislike"])
        ax.set_xlabel("Like")
        ax.set_ylabel("Dislike")
        ax.set_title("各エリアの Like / Dislike 散布図")
        for i, row in after_df.iterrows():
            ax.annotate(row["area"], (row["like"], row["dislike"]))
        st.pyplot(fig)

        st.subheader("有効なタッチ座標一覧（相殺後）")
        st.dataframe(coord_df)

        st.subheader("ルール適用後の座標プロット")
        image_file = st.file_uploader("背景画像（プロット用）", type=["png", "jpg", "jpeg"], key="plot_img1")
        if image_file:
            image = Image.open(image_file).convert("RGB")
            plotted_img = draw_points_on_image(image.copy(), coord_df)
            st.image(plotted_img, caption="ルール適用後のプロット", use_container_width=True)

        st.subheader("相殺前の全タッチ座標プロット")
        all_coords_df = extract_all_touch_coords(resp_df)
        if image_file:
            full_plot_img = draw_points_on_image(image.copy(), all_coords_df)
            st.image(full_plot_img, caption="相殺前の全タッチプロット", use_container_width=True)

elif mode == "画像へのプロット":
    st.header("画像へのプロット")
    image_file = st.file_uploader("背景画像（.png / .jpg）", type=["png", "jpg", "jpeg"])
    resp_file = st.file_uploader("回答データCSV（response.csv）", type="csv")

    if image_file and resp_file:
        resp_df = pd.read_csv(resp_file)
        coord_df = extract_all_touch_coords(resp_df)

        image = Image.open(image_file).convert("RGB")
        plotted_img = draw_points_on_image(image.copy(), coord_df)

        st.image(plotted_img, caption="全タッチプロット", use_container_width=True)

# -


















# OKだけど散布図が無い

# import streamlit as st
# import pandas as pd
# import numpy as np
# from shapely.geometry import Point, Polygon
# from PIL import Image, ImageDraw
# from collections import defaultdict
# import matplotlib.pyplot as plt
#
# # -----------------------------
# # 🔧 1. 集計関数（ルール前後 & 座標出力対応）
# # -----------------------------
# def calculate_area_flags(resp_df, polygons, apply_rule=True):
#     total_ids = len(resp_df)
#
#     # 各回答者 × エリアのlike/dislike集計
#     per_respondent_area = defaultdict(lambda: defaultdict(lambda: {"like": 0, "dislike": 0}))
#     area_flags = {area: {"like": 0, "dislike": 0} for area in polygons}
#
#     for idx, row in resp_df.iterrows():
#         rid = row.get("Respondent ID", idx)
#         for i in range(1, 3):
#             lx, ly = row.get(f"like{i}_x"), row.get(f"like{i}_y")
#             dx, dy = row.get(f"dislike{i}_x"), row.get(f"dislike{i}_y")
#             if pd.notnull(lx) and pd.notnull(ly):
#                 for area_name, poly in polygons.items():
#                     if poly.contains(Point(lx, ly)):
#                         per_respondent_area[rid][area_name]["like"] += 1
#                         break
#             if pd.notnull(dx) and pd.notnull(dy):
#                 for area_name, poly in polygons.items():
#                     if poly.contains(Point(dx, dy)):
#                         per_respondent_area[rid][area_name]["dislike"] += 1
#                         break
#
#     for rid, area_data in per_respondent_area.items():
#         for area, counts in area_data.items():
#             like = counts["like"]
#             dislike = counts["dislike"]
#             if apply_rule:
#                 if like > 0 and dislike > 0:
#                     continue
#                 elif like > 0:
#                     area_flags[area]["like"] += like
#                 elif dislike > 0:
#                     area_flags[area]["dislike"] += dislike
#             else:
#                 area_flags[area]["like"] += like
#                 area_flags[area]["dislike"] += dislike
#
#     area_summary = []
#     for area, counts in area_flags.items():
#         like = counts["like"]
#         dislike = counts["dislike"]
#         none = total_ids - like - dislike
#         area_summary.append({
#             "area": area,
#             "like": like,
#             "dislike": dislike,
#             "none": none,
#             "total": total_ids,
#             "like_ratio": like / total_ids if total_ids else 0,
#             "dislike_ratio": dislike / total_ids if total_ids else 0,
#             "none_ratio": none / total_ids if total_ids else 0
#         })
#
#     area_df = pd.DataFrame(area_summary)
#
#     # XY抽出（ルール適用後のみ）
#     coord_rows = []
#     for idx, row in resp_df.iterrows():
#         rid = row.get("Respondent ID", idx)
#         like_areas = {}
#         dislike_areas = {}
#
#         for i in range(1, 3):
#             lx, ly = row.get(f"like{i}_x"), row.get(f"like{i}_y")
#             dx, dy = row.get(f"dislike{i}_x"), row.get(f"dislike{i}_y")
#
#             if pd.notnull(lx) and pd.notnull(ly):
#                 for area, poly in polygons.items():
#                     if poly.contains(Point(lx, ly)):
#                         like_areas[i] = area
#                         break
#             if pd.notnull(dx) and pd.notnull(dy):
#                 for area, poly in polygons.items():
#                     if poly.contains(Point(dx, dy)):
#                         dislike_areas[i] = area
#                         break
#
#         canceled_areas = set(like_areas.values()) & set(dislike_areas.values())
#
#         row_dict = {"Respondent ID": rid}
#         for i in range(1, 3):
#             if i in like_areas and like_areas[i] not in canceled_areas:
#                 row_dict[f"like{i}_x"] = row.get(f"like{i}_x")
#                 row_dict[f"like{i}_y"] = row.get(f"like{i}_y")
#             else:
#                 row_dict[f"like{i}_x"] = np.nan
#                 row_dict[f"like{i}_y"] = np.nan
#
#             if i in dislike_areas and dislike_areas[i] not in canceled_areas:
#                 row_dict[f"dislike{i}_x"] = row.get(f"dislike{i}_x")
#                 row_dict[f"dislike{i}_y"] = row.get(f"dislike{i}_y")
#             else:
#                 row_dict[f"dislike{i}_x"] = np.nan
#                 row_dict[f"dislike{i}_y"] = np.nan
#
#         coord_rows.append(row_dict)
#
#     coord_df = pd.DataFrame(coord_rows)
#
#     return area_df, coord_df
#
# # -----------------------------
# # 🎯 相殺前の座標抽出関数
# # -----------------------------
# def extract_all_touch_coords(resp_df):
#     rows = []
#     for idx, row in resp_df.iterrows():
#         rid = row.get("Respondent ID", idx)
#         row_dict = {"Respondent ID": rid}
#         for i in range(1, 3):
#             row_dict[f"like{i}_x"] = row.get(f"like{i}_x")
#             row_dict[f"like{i}_y"] = row.get(f"like{i}_y")
#             row_dict[f"dislike{i}_x"] = row.get(f"dislike{i}_x")
#             row_dict[f"dislike{i}_y"] = row.get(f"dislike{i}_y")
#         rows.append(row_dict)
#     return pd.DataFrame(rows)
#
# # -----------------------------
# # 🖼️ 座標を画像に描画
# # -----------------------------
# def draw_points_on_image(img, df, color_like=(255, 0, 0), color_dislike=(0, 0, 255), radius=10):
#     draw = ImageDraw.Draw(img)
#     for _, row in df.iterrows():
#         for i in range(1, 3):
#             lx = row.get(f"like{i}_x")
#             ly = row.get(f"like{i}_y")
#             dx = row.get(f"dislike{i}_x")
#             dy = row.get(f"dislike{i}_y")
#
#             if pd.notna(lx) and pd.notna(ly):
#                 draw.ellipse((lx - radius, ly - radius, lx + radius, ly + radius), fill=color_like)
#             if pd.notna(dx) and pd.notna(dy):
#                 draw.ellipse((dx - radius, dy - radius, dx + radius, dy + radius), fill=color_dislike)
#     return img
#
# # -----------------------------
# # 🖥️ Streamlit アプリ本体
# # -----------------------------
# st.title("画像エリアの好き嫌い集計・可視化ツール")
#
# mode = st.radio("処理を選択してください", ["データ集計", "画像へのプロット"])
#
# if mode == "データ集計":
#     st.header("データアップロード")
#     area_file = st.file_uploader("エリア定義CSV（area.csv）", type="csv")
#     resp_file = st.file_uploader("回答データCSV（response.csv）", type="csv")
#
#     if area_file and resp_file:
#         area_df = pd.read_csv(area_file)
#         resp_df = pd.read_csv(resp_file)
#
#         polygons = {}
#         for name, group in area_df.groupby("name"):
#             points = [(x, y) for x, y in zip(group["x"], group["y"])]
#             polygons[name] = Polygon(points)
#
#         st.subheader("ルール適用前の集計")
#         before_df, _ = calculate_area_flags(resp_df, polygons, apply_rule=False)
#         st.dataframe(before_df)
#
#         st.subheader("ルール適用後の集計")
#         after_df, coord_df = calculate_area_flags(resp_df, polygons, apply_rule=True)
#         st.dataframe(after_df)
#
#         st.subheader("ルール適用前後の差分")
#         diff_df = after_df[["area", "like", "dislike"]].copy()
#         diff_df["like_diff"] = after_df["like"] - before_df["like"]
#         diff_df["dislike_diff"] = after_df["dislike"] - before_df["dislike"]
#         st.dataframe(diff_df[["area", "like_diff", "dislike_diff"]])
#
#         st.subheader("有効なタッチ座標一覧（相殺後）")
#         st.dataframe(coord_df)
#
#         st.subheader("ルール適用後の座標プロット")
#         image_file = st.file_uploader("背景画像（プロット用）", type=["png", "jpg", "jpeg"], key="plot_img1")
#         if image_file:
#             image = Image.open(image_file).convert("RGB")
#             plotted_img = draw_points_on_image(image.copy(), coord_df)
#             st.image(plotted_img, caption="ルール適用後のプロット", use_container_width=True)
#
#         st.subheader("相殺前の全タッチ座標プロット")
#         all_coords_df = extract_all_touch_coords(resp_df)
#         if image_file:
#             full_plot_img = draw_points_on_image(image.copy(), all_coords_df)
#             st.image(full_plot_img, caption="相殺前の全タッチプロット", use_container_width=True)
#
# elif mode == "画像へのプロット":
#     st.header("画像へのプロット")
#     image_file = st.file_uploader("背景画像（.png / .jpg）", type=["png", "jpg", "jpeg"])
#     resp_file = st.file_uploader("回答データCSV（response.csv）", type="csv")
#
#     if image_file and resp_file:
#         resp_df = pd.read_csv(resp_file)
#         coord_df = extract_all_touch_coords(resp_df)
#
#         image = Image.open(image_file).convert("RGB")
#         plotted_img = draw_points_on_image(image.copy(), coord_df)
#
#         st.image(plotted_img, caption="全タッチプロット", use_container_width=True)
#











# OK

# import streamlit as st
# import pandas as pd
# import numpy as np
# from shapely.geometry import Point, Polygon
# from PIL import Image, ImageDraw
# from collections import defaultdict
# import matplotlib.pyplot as plt
#
# # -----------------------------
# # 🔧 1. 集計関数
# # -----------------------------
# def calculate_area_flags(resp_df, polygons, apply_rule=True):
#     total_ids = len(resp_df)
#
#     per_respondent_area = defaultdict(lambda: defaultdict(lambda: {"like": 0, "dislike": 0}))
#     area_flags = {area: {"like": 0, "dislike": 0} for area in polygons}
#
#     for idx, row in resp_df.iterrows():
#         rid = row.get("Respondent ID", idx)
#         for i in range(1, 3):
#             lx, ly = row.get(f"like{i}_x"), row.get(f"like{i}_y")
#             dx, dy = row.get(f"dislike{i}_x"), row.get(f"dislike{i}_y")
#
#             if pd.notnull(lx) and pd.notnull(ly):
#                 for area_name, poly in polygons.items():
#                     if poly.contains(Point(lx, ly)):
#                         per_respondent_area[rid][area_name]["like"] += 1
#                         break
#
#             if pd.notnull(dx) and pd.notnull(dy):
#                 for area_name, poly in polygons.items():
#                     if poly.contains(Point(dx, dy)):
#                         per_respondent_area[rid][area_name]["dislike"] += 1
#                         break
#
#     for rid, area_data in per_respondent_area.items():
#         for area, counts in area_data.items():
#             like = counts["like"]
#             dislike = counts["dislike"]
#
#             if apply_rule:
#                 if like > 0 and dislike > 0:
#                     continue
#                 elif like > 0:
#                     area_flags[area]["like"] += like
#                 elif dislike > 0:
#                     area_flags[area]["dislike"] += dislike
#             else:
#                 area_flags[area]["like"] += like
#                 area_flags[area]["dislike"] += dislike
#
#     area_summary = []
#     for area, counts in area_flags.items():
#         like = counts["like"]
#         dislike = counts["dislike"]
#         none = total_ids - like - dislike
#         area_summary.append({
#             "area": area,
#             "like": like,
#             "dislike": dislike,
#             "none": none,
#             "total": total_ids,
#             "like_ratio": like / total_ids if total_ids else 0,
#             "dislike_ratio": dislike / total_ids if total_ids else 0,
#             "none_ratio": none / total_ids if total_ids else 0
#         })
#
#     area_df = pd.DataFrame(area_summary)
#
#     # 座標抽出（ルール適用後のみ）
#     coord_rows = []
#     if apply_rule:
#         for idx, row in resp_df.iterrows():
#             rid = row.get("Respondent ID", idx)
#             like_areas = {}
#             dislike_areas = {}
#
#             for i in range(1, 3):
#                 lx, ly = row.get(f"like{i}_x"), row.get(f"like{i}_y")
#                 dx, dy = row.get(f"dislike{i}_x"), row.get(f"dislike{i}_y")
#
#                 if pd.notnull(lx) and pd.notnull(ly):
#                     for area, poly in polygons.items():
#                         if poly.contains(Point(lx, ly)):
#                             like_areas[i] = area
#                             break
#
#                 if pd.notnull(dx) and pd.notnull(dy):
#                     for area, poly in polygons.items():
#                         if poly.contains(Point(dx, dy)):
#                             dislike_areas[i] = area
#                             break
#
#             canceled_areas = set(like_areas.values()) & set(dislike_areas.values())
#
#             row_dict = {"Respondent ID": rid}
#             for i in range(1, 3):
#                 if i in like_areas and like_areas[i] not in canceled_areas:
#                     row_dict[f"like{i}_x"] = row.get(f"like{i}_x")
#                     row_dict[f"like{i}_y"] = row.get(f"like{i}_y")
#                 else:
#                     row_dict[f"like{i}_x"] = np.nan
#                     row_dict[f"like{i}_y"] = np.nan
#
#                 if i in dislike_areas and dislike_areas[i] not in canceled_areas:
#                     row_dict[f"dislike{i}_x"] = row.get(f"dislike{i}_x")
#                     row_dict[f"dislike{i}_y"] = row.get(f"dislike{i}_y")
#                 else:
#                     row_dict[f"dislike{i}_x"] = np.nan
#                     row_dict[f"dislike{i}_y"] = np.nan
#
#             coord_rows.append(row_dict)
#
#     coord_df = pd.DataFrame(coord_rows)
#     return area_df, coord_df
#
#
# # -----------------------------
# # 🎨 2. プロット関数
# # -----------------------------
# def draw_points_on_image(img, df, color_like=(255, 0, 0), color_dislike=(0, 0, 255), radius=10):
#     draw = ImageDraw.Draw(img)
#     for _, row in df.iterrows():
#         for i in range(1, 3):
#             lx, ly = row.get(f"like{i}_x"), row.get(f"like{i}_y")
#             dx, dy = row.get(f"dislike{i}_x"), row.get(f"dislike{i}_y")
#
#             if pd.notna(lx) and pd.notna(ly):
#                 draw.ellipse((lx - radius, ly - radius, lx + radius, ly + radius), fill=color_like)
#             if pd.notna(dx) and pd.notna(dy):
#                 draw.ellipse((dx - radius, dy - radius, dx + radius, dy + radius), fill=color_dislike)
#     return img
#
#
# # -----------------------------
# # 🖥️ 3. Streamlit UI
# # -----------------------------
# st.title("画像エリアの好き嫌い集計・可視化ツール")
#
# mode = st.radio("処理を選択してください", ["データ集計", "画像へのプロット"])
#
# if mode == "データ集計":
#     st.header("データアップロード")
#     area_file = st.file_uploader("エリア定義CSV（area.csv）", type="csv")
#     resp_file = st.file_uploader("回答データCSV（response.csv）", type="csv")
#     image_file = st.file_uploader("背景画像（プロット用・任意）", type=["png", "jpg", "jpeg"])
#
#     if area_file and resp_file:
#         area_df = pd.read_csv(area_file)
#         resp_df = pd.read_csv(resp_file)
#
#         polygons = {}
#         for name, group in area_df.groupby("name"):
#             points = [(x, y) for x, y in zip(group["x"], group["y"])]
#             polygons[name] = Polygon(points)
#
#         st.subheader("ルール適用前の集計")
#         before_df, _ = calculate_area_flags(resp_df, polygons, apply_rule=False)
#         st.dataframe(before_df)
#
#         st.subheader("ルール適用後の集計")
#         after_df, coord_df = calculate_area_flags(resp_df, polygons, apply_rule=True)
#         st.dataframe(after_df)
#
#         st.subheader("差分（適用後 - 適用前）")
#         diff_df = after_df[["area", "like", "dislike"]].set_index("area") - before_df[["area", "like", "dislike"]].set_index("area")
#         st.dataframe(diff_df.reset_index())
#
#         st.subheader("エリアごとの like / dislike 散布図（ルール適応後）")
#         fig, ax = plt.subplots()
#         ax.scatter(after_df["like"], after_df["dislike"])
#         for _, row in after_df.iterrows():
#             ax.text(row["like"], row["dislike"], row["area"], fontsize=9)
#         ax.set_xlabel("Like")
#         ax.set_ylabel("Dislike")
#         ax.set_title("Like vs Dislike by Area")
#         st.pyplot(fig)
#
#         st.subheader("有効なタッチ座標一覧（相殺後）")
#         st.dataframe(coord_df)
#
#         if image_file:
#             image = Image.open(image_file).convert("RGB")
#             plotted_img = draw_points_on_image(image.copy(), coord_df)
#             st.subheader("相殺後のタッチ座標プロット")
#             st.image(plotted_img, caption="ルール適用後のプロット", use_container_width=True)
#
# elif mode == "画像へのプロット":
#     st.header("画像へのプロット")
#     image_file = st.file_uploader("背景画像（.png / .jpg）", type=["png", "jpg", "jpeg"])
#     resp_file = st.file_uploader("回答データCSV（response.csv）", type="csv")
#
#     if image_file and resp_file:
#         resp_df = pd.read_csv(resp_file)
#         image = Image.open(image_file).convert("RGB")
#
#         # 相殺処理は行わない → 全タッチを表示
#         coord_df = resp_df.copy()
#         plotted_img = draw_points_on_image(image.copy(), coord_df)
#         st.image(plotted_img, caption="タッチプロット（ルール未適用）", use_container_width=True)
#





# import streamlit as st
# import pandas as pd
# import numpy as np
# from shapely.geometry import Point, Polygon
# from PIL import Image, ImageDraw
# from collections import defaultdict
#
# # -----------------------------
# # 🔧 1. 集計関数（ルール前後 & 座標出力対応）
# # -----------------------------
# def calculate_area_flags(resp_df, polygons, apply_rule=True):
#     total_ids = len(resp_df)
#
#     # 各回答者 × エリアのlike/dislike集計
#     per_respondent_area = defaultdict(lambda: defaultdict(lambda: {"like": 0, "dislike": 0}))
#     area_flags = {area: {"like": 0, "dislike": 0} for area in polygons}
#
#     for idx, row in resp_df.iterrows():
#         rid = row.get("Respondent ID", idx)
#
#         for i in range(1, 3):
#             lx, ly = row.get(f"like{i}_x"), row.get(f"like{i}_y")
#             dx, dy = row.get(f"dislike{i}_x"), row.get(f"dislike{i}_y")
#
#             if pd.notnull(lx) and pd.notnull(ly):
#                 for area_name, poly in polygons.items():
#                     if poly.contains(Point(lx, ly)):
#                         per_respondent_area[rid][area_name]["like"] += 1
#                         break
#
#             if pd.notnull(dx) and pd.notnull(dy):
#                 for area_name, poly in polygons.items():
#                     if poly.contains(Point(dx, dy)):
#                         per_respondent_area[rid][area_name]["dislike"] += 1
#                         break
#
#     # 相殺ルールの適用
#     for rid, area_data in per_respondent_area.items():
#         for area, counts in area_data.items():
#             like = counts["like"]
#             dislike = counts["dislike"]
#
#             if apply_rule:
#                 if like > 0 and dislike > 0:
#                     continue
#                 elif like > 0:
#                     area_flags[area]["like"] += like
#                 elif dislike > 0:
#                     area_flags[area]["dislike"] += dislike
#             else:
#                 area_flags[area]["like"] += like
#                 area_flags[area]["dislike"] += dislike
#
#     area_summary = []
#     for area, counts in area_flags.items():
#         like = counts["like"]
#         dislike = counts["dislike"]
#         none = total_ids - like - dislike
#         area_summary.append({
#             "area": area,
#             "like": like,
#             "dislike": dislike,
#             "none": none,
#             "total": total_ids,
#             "like_ratio": like / total_ids if total_ids else 0,
#             "dislike_ratio": dislike / total_ids if total_ids else 0,
#             "none_ratio": none / total_ids if total_ids else 0
#         })
#
#     area_df = pd.DataFrame(area_summary)
#
#     # XY抽出（ルール適用後のみ）
#     coord_rows = []
#     for idx, row in resp_df.iterrows():
#         rid = row.get("Respondent ID", idx)
#         like_areas = {}
#         dislike_areas = {}
#
#         for i in range(1, 3):
#             lx, ly = row.get(f"like{i}_x"), row.get(f"like{i}_y")
#             dx, dy = row.get(f"dislike{i}_x"), row.get(f"dislike{i}_y")
#
#             if pd.notnull(lx) and pd.notnull(ly):
#                 for area, poly in polygons.items():
#                     if poly.contains(Point(lx, ly)):
#                         like_areas[i] = area
#                         break
#
#             if pd.notnull(dx) and pd.notnull(dy):
#                 for area, poly in polygons.items():
#                     if poly.contains(Point(dx, dy)):
#                         dislike_areas[i] = area
#                         break
#
#         canceled_areas = set(like_areas.values()) & set(dislike_areas.values())
#
#         row_dict = {"Respondent ID": rid}
#         for i in range(1, 3):
#             if i in like_areas and like_areas[i] not in canceled_areas:
#                 row_dict[f"like{i}_x"] = row.get(f"like{i}_x")
#                 row_dict[f"like{i}_y"] = row.get(f"like{i}_y")
#             else:
#                 row_dict[f"like{i}_x"] = np.nan
#                 row_dict[f"like{i}_y"] = np.nan
#
#             if i in dislike_areas and dislike_areas[i] not in canceled_areas:
#                 row_dict[f"dislike{i}_x"] = row.get(f"dislike{i}_x")
#                 row_dict[f"dislike{i}_y"] = row.get(f"dislike{i}_y")
#             else:
#                 row_dict[f"dislike{i}_x"] = np.nan
#                 row_dict[f"dislike{i}_y"] = np.nan
#
#         coord_rows.append(row_dict)
#
#     coord_df = pd.DataFrame(coord_rows)
#
#     return area_df, coord_df
#
# # -----------------------------
# # 🎨 2. プロット関数
# # -----------------------------
# def draw_points_on_image(img, df, color_like=(255, 0, 0), color_dislike=(0, 0, 255)):
#     draw = ImageDraw.Draw(img)
#     for _, row in df.iterrows():
#         for i in range(1, 3):
#             lx = row.get(f"like{i}_x")
#             ly = row.get(f"like{i}_y")
#             dx = row.get(f"dislike{i}_x")
#             dy = row.get(f"dislike{i}_y")
#
#             if pd.notna(lx) and pd.notna(ly):
#                 draw.ellipse((lx - 10, ly - 10, lx + 10, ly + 10), fill=color_like)
#             if pd.notna(dx) and pd.notna(dy):
#                 draw.ellipse((dx - 10, dy - 10, dx + 10, dy + 10), fill=color_dislike)
#     return img
#
# def extract_touch_coords_only(resp_df):
#     rows = []
#     for idx, row in resp_df.iterrows():
#         rid = row.get("Respondent ID", idx)
#         row_dict = {"Respondent ID": rid}
#         for i in range(1, 3):
#             row_dict[f"like{i}_x"] = row.get(f"like{i}_x", np.nan)
#             row_dict[f"like{i}_y"] = row.get(f"like{i}_y", np.nan)
#             row_dict[f"dislike{i}_x"] = row.get(f"dislike{i}_x", np.nan)
#             row_dict[f"dislike{i}_y"] = row.get(f"dislike{i}_y", np.nan)
#         rows.append(row_dict)
#     return pd.DataFrame(rows)
#
# # -----------------------------
# # 🖥️ 3. Streamlit UI 本体
# # -----------------------------
# st.title("画像エリアの好き嫌い集計・可視化ツール")
#
# mode = st.radio("処理を選択してください", ["データ集計", "画像へのプロット"])
#
# if mode == "データ集計":
#     st.header("データアップロード")
#     area_file = st.file_uploader("エリア定義CSV（area.csv）", type="csv")
#     resp_file = st.file_uploader("回答データCSV（response.csv）", type="csv")
#
#     if area_file and resp_file:
#         area_df = pd.read_csv(area_file)
#         resp_df = pd.read_csv(resp_file)
#
#         polygons = {}
#         for name, group in area_df.groupby("name"):
#             points = [(x, y) for x, y in zip(group["x"], group["y"])]
#             polygons[name] = Polygon(points)
#
#         st.subheader("ルール適用前の集計")
#         before_df, _ = calculate_area_flags(resp_df, polygons, apply_rule=False)
#         st.dataframe(before_df)
#
#         st.subheader("ルール適用後の集計")
#         after_df, coord_df = calculate_area_flags(resp_df, polygons, apply_rule=True)
#         st.dataframe(after_df)
#
#         # 差分計算（like/dislikeのみ）
#         cols_to_compare = ["like", "dislike"]
#         diff_df = after_df.set_index("area")[cols_to_compare] - before_df.set_index("area")[cols_to_compare]
#         diff_df = diff_df.reset_index()
#
#         st.subheader("ルール適用前後の差分（like/dislikeのみ）")
#         st.dataframe(diff_df)
#
#         st.subheader("有効なタッチ座標一覧（相殺後）")
#         st.dataframe(coord_df)
#
#
# elif mode == "画像へのプロット":
#     st.header("画像へのプロット")
#     image_file = st.file_uploader("背景画像（.png / .jpg）", type=["png", "jpg", "jpeg"])
#     resp_file = st.file_uploader("回答データCSV（response.csv）", type="csv")
#
#     if image_file and resp_file:
#         resp_df = pd.read_csv(resp_file)
#         coord_df = extract_touch_coords_only(resp_df)
#
#         image = Image.open(image_file).convert("RGB")
#         plotted_img = draw_points_on_image(image.copy(), coord_df)
#
#         st.image(plotted_img, caption="ルール適用後のプロット", use_container_width=True)
#
#











