import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyBboxPatch, BoxStyle
from matplotlib import patheffects as pe
from PIL import Image, ImageDraw, ImageOps
import os
import numpy as np
import matplotlib.font_manager as fm

# --- 폰트 설정 ---
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    try:
        plt.rcParams['font.family'] = 'AppleGothic'
    except:
        pass
plt.rcParams['axes.unicode_minus'] = False

# --- 설정 변수 ---
N_BARS = 20
FPS = 30
DURATION_PER_FRAME = 40
LOGO_SHAPE_CIRCULAR = True
LOGO_SIZE_PX = 70

# ★ [추가] 막대 높이를 픽셀로 설정 ★
BAR_HEIGHT_PX = 70 

VIDEO_WIDTH_PX = 1080
VIDEO_HEIGHT_PX = 1920
DPI = 200

# --- 데이터 불러오기 및 보간 ---
df = pd.read_csv('youtube_data.csv', index_col='date', parse_dates=True)
df_interpolated = df.resample('D').mean().interpolate(method='linear')

# --- 안정적인 색상 매핑 ---
cmap = plt.get_cmap('tab10')
channel_colors = {channel: cmap(i % 10) for i, channel in enumerate(df.columns)}

# --- 로고 이미지를 원형으로 만드는 함수 ---
def create_circular_logo(img_path, size_px):
    if img_path is None or not os.path.exists(img_path):
        return None
    img = Image.open(img_path).convert("RGBA")
    
    # 원본 이미지를 정사각형으로 리사이즈
    size = (size_px, size_px)
    img = img.resize(size, Image.LANCZOS)

    if LOGO_SHAPE_CIRCULAR:
        # 원형 마스크 생성
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0) + size, fill=255)
        output = ImageOps.fit(img, size, centering=(0.5, 0.5))
        output.putalpha(mask)
        return output
    else:
        return img

# --- 로고 이미지 불러오기 ---
logo_cache = {}
logo_dir = 'logos'
default_logo_path = os.path.join(logo_dir, 'default.jpg')
for channel in df.columns:
    logo_path = os.path.join(logo_dir, f'{channel}.png')
    if os.path.exists(logo_path):
        logo_cache[channel] = create_circular_logo(logo_path, LOGO_SIZE_PX)
    else:
        print(f"'{channel}.png'를 찾을 수 없어 기본 로고를 사용합니다.")
        logo_cache[channel] = create_circular_logo(default_logo_path, LOGO_SIZE_PX)

# --- Figure 및 Axes 설정 (해상도 고정) ---
figsize = (VIDEO_WIDTH_PX / DPI, VIDEO_HEIGHT_PX / DPI)
fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
fig.set_facecolor('#000000')

# --- 부드러운 순위 변화를 위한 전역 변수 ---
previous_ranking = None
transition_frames = 10  # 순위 변화 시 전환 프레임 수

# --- 애니메이션 업데이트 함수 ---
def update(frame):
    global previous_ranking
    
    ax.clear()
    ax.set_facecolor('#000000')

    current_date = df_interpolated.index[frame]
    current_data = df_interpolated.loc[current_date].sort_values(ascending=False).head(N_BARS)

    max_val = current_data.max() or 0
    
    # X축 범위를 고정값으로 설정하여 아이콘이 잘리지 않도록 보장
    dynamic_xlim = max_val * 1.3  # 최대값의 130%로 고정
    if dynamic_xlim < 1: dynamic_xlim = 1

    ax.set_xlim(0, dynamic_xlim)
    ax.set_ylim(-0.5, N_BARS - 0.5)
    #ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=1)
    #ax.tick_params(axis='x', labelsize=14, colors='gray', length=0)
    ax.spines[:].set_visible(False)
    ax.get_yaxis().set_ticks([])
    
    # ★ [수정] 픽셀 높이를 데이터 단위로 정확하게 변환 ★
    # matplotlib transform을 사용하여 정확한 픽셀-데이터 변환
    # 1픽셀 높이를 데이터 좌표로 변환
    pixel_to_data_transform = ax.transData.inverted()
    one_pixel_height_data = abs(pixel_to_data_transform.transform((0, 1))[1] - pixel_to_data_transform.transform((0, 0))[1])
    bar_height_data = BAR_HEIGHT_PX * one_pixel_height_data

    # 현재 순위와 이전 순위 비교하여 부드러운 전환 계산
    current_ranking = list(current_data.index)
    
    if previous_ranking is None:
        # 첫 번째 프레임
        y_positions = np.arange(N_BARS)[::-1]
        ranking_positions = {channel: i for i, channel in enumerate(current_ranking)}
    else:
        # 이전 순위와 현재 순위 간의 부드러운 전환
        ranking_positions = {}
        
        # 각 채널의 목표 위치 계산
        target_positions = {channel: i for i, channel in enumerate(current_ranking)}
        
        # 이전 위치와 목표 위치 간의 보간
        for channel in current_ranking:
            if channel in previous_ranking:
                # 이전 위치에서 목표 위치로 부드럽게 이동
                prev_pos = previous_ranking.index(channel)
                target_pos = target_positions[channel]
                
                # 선형 보간 (더 부드러운 전환을 위해 ease-in-out 사용)
                progress = min(1.0, frame / transition_frames)
                # ease-in-out 함수 적용
                eased_progress = 3 * progress**2 - 2 * progress**3
                
                current_pos = prev_pos + (target_pos - prev_pos) * eased_progress
                ranking_positions[channel] = current_pos
            else:
                # 새로운 채널은 목표 위치에 바로 배치
                ranking_positions[channel] = target_positions[channel]
    
    # Y 위치 계산 (부드러운 전환을 위해 정렬)
    sorted_channels = sorted(ranking_positions.items(), key=lambda x: x[1])
    y_positions = np.arange(N_BARS)[::-1]
    
    # 부드러운 전환을 위해 채널별로 처리
    for channel, value in current_data.items():
        # 현재 채널의 Y 위치 계산 (부드러운 전환 적용)
        if channel in ranking_positions:
            # 부드러운 전환된 위치 사용
            smooth_rank = ranking_positions[channel]
            y_pos = N_BARS - 1 - smooth_rank  # Y축은 위에서 아래로
        else:
            # 기본 위치 사용
            y_pos = y_positions[list(current_data.index).index(channel)]
        
        # 일반 막대 그리기
        if value > 0:
            ax.barh(y_pos, value, height=bar_height_data, color=channel_colors[channel], zorder=2)

        # 채널 이름을 막대 안에서만 표시 (막대 너비를 넘지 않게)
        if value > 0:
            # 막대 너비를 고려한 텍스트 위치 계산
            text_x_pos = value * 0.5  # 막대 중앙에 배치
            
            # 텍스트가 막대 너비를 넘지 않도록 클리핑 박스 설정
            clip_box = plt.Rectangle((0, y_pos - bar_height_data/2), value, bar_height_data, 
                                   transform=ax.transData, clip_on=True)
            
            # 텍스트 표시 (클리핑 박스 적용)
            text_obj = ax.text(text_x_pos, y_pos, channel,
                             ha='center', va='center', fontsize=18, color='white', weight='bold', zorder=3,
                             clip_on=True)
            text_obj.set_clip_path(clip_box)

        # 아이콘과 수치 표시
        logo_img = logo_cache.get(channel)
        if logo_img:
            # 아이콘 크기를 정확한 픽셀 크기로 조정
            zoom_factor = 72.0 / fig.dpi  # 72 DPI 기준으로 정규화
            imagebox = OffsetImage(logo_img, zoom=zoom_factor)
            
            # 아이콘을 막대 끝에 정확히 배치
            ab = AnnotationBbox(imagebox, (value, y_pos),
                                xybox=(0, 0),
                                xycoords='data',
                                boxcoords="offset points",
                                frameon=False, pad=0,
                                box_alignment=(0.5, 0.5), zorder=3)
            # 클리핑 완전 비활성화
            ab.set_clip_on(False)
            ax.add_artist(ab)
            
            # 수치를 아이콘 오른쪽에 표시 (화면 범위 내에서 안전하게)
            text_x_pos = value + max_val * 0.05  # 최대값의 5%만큼 오프셋
            # 화면 범위를 벗어나지 않도록 제한
            text_x_pos = min(text_x_pos, dynamic_xlim * 0.95)
            ax.text(text_x_pos, y_pos, f'{value:,.0f}',
                    ha='left', va='center', fontsize=16, color='white', weight='semibold', zorder=3)
        else:
            # 아이콘이 없을 때는 막대 끝에 수치 표시 (화면 범위 내에서 안전하게)
            text_x_pos = value + max_val * 0.05
            text_x_pos = min(text_x_pos, dynamic_xlim * 0.95)
            ax.text(text_x_pos, y_pos, f'{value:,.0f}',
                    ha='left', va='center', fontsize=16, color='white', weight='semibold', zorder=3)

    ax.text(0.5, 0.95, '유튜브 채널 구독자',
            transform=fig.transFigure, ha='center', va='center', fontsize=32, color='white', weight='bold')
    ax.text(0.5, 0.1, current_date.strftime('%Y년 %m월 %d일'),
            transform=fig.transFigure, ha='center', va='center', fontsize=28, color='white', weight='semibold')
    
    # 다음 프레임을 위해 현재 순위를 이전 순위로 저장
    previous_ranking = current_ranking

# --- 애니메이션 생성 및 저장 ---
ani = animation.FuncAnimation(
    fig, update, frames=len(df_interpolated), interval=DURATION_PER_FRAME, blit=False
)

print("동영상 저장을 시작합니다...")
writer = animation.FFMpegWriter(fps=FPS, metadata=dict(artist='Me'), bitrate=3000)
ani.save('youtube_ranking_race_1080x1920.mp4', writer=writer)

print("애니메이션 동영상 저장이 완료되었습니다: youtube_ranking_race_1080x1920.mp4")