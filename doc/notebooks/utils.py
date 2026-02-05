# 공통 유틸리티 함수
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

warnings.filterwarnings('ignore')

def setup_matplotlib():
    """matplotlib 한글 폰트 설정"""
    # 자주 사용되는 한글 폰트 목록 (우선순위 순)
    korean_fonts = [
        # Noto 폰트 (Linux에서 가장 흔함)
        'Noto Sans CJK KR',
        'Noto Sans CJK JP',  # fallback
        # 나눔 폰트
        'NanumSquare',
        'NanumGothic',
        'NanumBarunGothic',
        # Windows
        'Malgun Gothic',
        # macOS
        'AppleGothic',
        # 기타
        'Noto Sans KR',
        'UnDotum',
        'Baekmuk Gulim'
    ]

    # 시스템에 설치된 폰트 목록
    system_fonts = {f.name for f in fm.fontManager.ttflist}

    # 사용 가능한 한글 폰트 찾기
    for font in korean_fonts:
        if font in system_fonts:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            return True

    # 부분 매칭 시도 (폰트 이름이 약간 다를 수 있음)
    for font in korean_fonts:
        for sys_font in system_fonts:
            if font.lower().replace(' ', '') in sys_font.lower().replace(' ', ''):
                plt.rcParams['font.family'] = sys_font
                plt.rcParams['axes.unicode_minus'] = False
                return True

    # 한글 폰트를 찾지 못한 경우 - 경고 출력 없이 기본 설정만
    plt.rcParams['axes.unicode_minus'] = False
    return False

# 자동 실행
KOREAN_AVAILABLE = setup_matplotlib()
