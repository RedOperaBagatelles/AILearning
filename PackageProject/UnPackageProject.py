from pathlib import Path
import re

# 병합 대상 디렉터리 (현재 스크립트 기준 상대 경로)
TARGET_DIRS = [ './' ]


def merge_file(part01_path: Path) -> None:
    """
    .part01부터 시작하는 분할 파일들을 병합하여 원본 파일을 복원하고,
    병합 후 .partNN 파일들을 삭제합니다.
    """
    merged_file_name = re.sub(r'\.part\d{2}$', '', part01_path.name)
    merged_path = part01_path.with_name(merged_file_name)

    print(f"🔧 병합 시작: {merged_file_name}")

    with merged_path.open('wb') as merged_file:
        part_number = 1
        while True:
            part_file = part01_path.with_name(f"{merged_file_name}.part{part_number:02d}")
            if not part_file.exists():
                break
            with part_file.open('rb') as pf:
                data = pf.read()
                merged_file.write(data)
                print(f"  📦 병합 중: {part_file.name} ({len(data):,}바이트)")
            part_number += 1

    print(f"  🎉 병합 완료: {merged_path.name} ({merged_path.stat().st_size:,}바이트)")

    deleted = 0
    for i in range(1, part_number):
        part_file = part01_path.with_name(f"{merged_file_name}.part{i:02d}")
        if part_file.exists():
            part_file.unlink()
            print(f"  🗑️  삭제됨: {part_file.name}")
            deleted += 1

    print(f"  ✅ 총 {deleted}개 파트 파일 삭제 완료\n")


def scan_and_merge(base_dir: Path, target_dirs: list) -> None:
    """
    target_dirs 아래의 모든 .part01 파일을 재귀 탐색하여
    자동으로 병합합니다.
    """
    total_merged = 0

    for rel_dir in target_dirs:
        scan_dir = (base_dir / rel_dir).resolve()

        if not scan_dir.exists():
            print(f"❌ 디렉터리를 찾을 수 없습니다: {scan_dir}")
            continue

        print(f"\n📂 탐색 중: {scan_dir}")

        # .part01 파일만 찾기 (병합 시작점)
        part01_files = sorted(scan_dir.rglob('*.part01'))

        if not part01_files:
            print(f"  ℹ️  병합이 필요한 파일 없음 (.part01 파일 없음)")
            continue

        print(f"  📋 병합 대상 {len(part01_files)}개 발견:\n")
        for p in part01_files:
            rel = p.relative_to(base_dir)
            merged_name = re.sub(r'\.part\d{2}$', '', p.name)
            print(f"  ▶ {rel.parent / merged_name}")

        print()

        for part01_path in part01_files:
            merge_file(part01_path)
            total_merged += 1

    print(f"{'='*50}")
    print(f"✨ 전체 완료: {total_merged}개 파일 병합됨")


if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent

    scan_and_merge(base_dir, TARGET_DIRS)
