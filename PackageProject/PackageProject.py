import os
from pathlib import Path

# 청크 크기: 95 MiB
CHUNK_SIZE = 95 * 1024 * 1024  # 95MB

# 분할 대상 디렉터리 (현재 스크립트 기준 상대 경로)
# 예: 'src/main/resources/static/images' 또는 '.' (전체 프로젝트)
TARGET_DIRS = [ './' ]

def is_part_file(path: Path) -> bool:
    """이미 분할된 .partNN 파일인지 확인"""
    import re
    return bool(re.search(r'\.part\d{2}$', path.name))


def split_file(file_path: Path, chunk_size: int = CHUNK_SIZE) -> None:
    """
    지정된 파일을 chunk_size 바이트 단위로 분할하여
    같은 디렉터리에 .partNN 형식으로 저장하고,
    완료 후 원본 파일은 삭제합니다.
    """
    file_size = file_path.stat().st_size

    with file_path.open('rb') as src:
        part = 1
        while True:
            data = src.read(chunk_size)
            if not data:
                break
            out_name = file_path.with_name(f"{file_path.name}.part{part:02d}")
            with out_name.open('wb') as dst:
                dst.write(data)
            print(f"  ✅ 생성: {out_name.name} ({len(data):,}바이트)")
            part += 1

    print(f"  🎉 완료: {file_path.name} → {part-1}개 파트 분할")

    try:
        file_path.unlink()
        print(f"  🗑️  원본 파일 삭제: {file_path.name}")
    except Exception as e:
        print(f"  ⚠️  원본 파일 삭제 중 오류 발생: {e}")


def scan_and_split(base_dir: Path, target_dirs: list, chunk_size: int = CHUNK_SIZE) -> None:
    """
    target_dirs 아래의 모든 파일을 재귀 탐색하여
    chunk_size 초과 파일을 자동으로 분할합니다.
    """
    total_found = 0
    total_split = 0

    for rel_dir in target_dirs:
        scan_dir = (base_dir / rel_dir).resolve()

        if not scan_dir.exists():
            print(f"❌ 디렉터리를 찾을 수 없습니다: {scan_dir}")
            continue

        print(f"\n📂 탐색 중: {scan_dir}")

        # 재귀적으로 모든 파일 탐색
        all_files = sorted(scan_dir.rglob('*'))
        candidates = []

        for file_path in all_files:
            if not file_path.is_file():
                continue
            if is_part_file(file_path):
                continue  # 이미 분할된 파일 건너뜀
            file_size = file_path.stat().st_size
            if file_size > chunk_size:
                candidates.append((file_path, file_size))

        total_found += len(candidates)

        if not candidates:
            print(f"  ℹ️  분할이 필요한 파일 없음 (모두 {chunk_size // (1024*1024)}MiB 이하)")
            continue

        print(f"  📋 분할 대상 파일 {len(candidates)}개 발견:\n")
        for file_path, file_size in candidates:
            rel = file_path.relative_to(base_dir)
            print(f"  ▶ {rel}  ({file_size / (1024*1024):.1f} MiB)")

        print()

        for file_path, file_size in candidates:
            rel = file_path.relative_to(base_dir)
            print(f"🔪 분할 시작: {rel}  ({file_size / (1024*1024):.1f} MiB)")
            split_file(file_path, chunk_size)
            total_split += 1
            print()

    print(f"{'='*50}")
    print(f"✨ 전체 완료: {total_found}개 파일 중 {total_split}개 분할됨")


if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent

    scan_and_split(base_dir, TARGET_DIRS, CHUNK_SIZE)
