# 공통 유틸리티 함수
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

warnings.filterwarnings('ignore')

def setup_matplotlib():
    """matplotlib 한글 폰트 설정"""
    # 자주 사용되는 한글 폰트 목록
    korean_fonts = [
        'NanumGothic', 'NanumBarunGothic', 'Malgun Gothic',
        'AppleGothic', 'Noto Sans CJK KR', 'Noto Sans KR',
        'UnDotum', 'Baekmuk Gulim'
    ]

    # 시스템에 설치된 폰트 목록
    system_fonts = [f.name for f in fm.fontManager.ttflist]

    # 사용 가능한 한글 폰트 찾기
    for font in korean_fonts:
        if font in system_fonts:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            print(f"Korean font set: {font}")
            return True

    # 한글 폰트를 찾지 못한 경우
    print("Warning: Korean font not found. Using default font.")
    plt.rcParams['axes.unicode_minus'] = False
    return False

# 자동 실행
KOREAN_AVAILABLE = setup_matplotlib()
