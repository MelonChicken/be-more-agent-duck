from collections import Counter

from src.behaviour_states import BehaviourClass


def generate_summary_text(timeline, total_sec):
    total_segments = len(timeline)
    behaviour_counts = Counter(item["behaviour"] for item in timeline)

    if total_segments == 0:
        return (
            "이번 세션에서 분석된 행동 구간이 없습니다. "
            "그 외 나온 행동은 없습니다. "
            "uncertain 비율은 약 0%입니다."
        )

    dominant, dominant_count = behaviour_counts.most_common(1)[0]
    dominant_ratio = round(dominant_count / total_segments * 100)
    uncertain_count = behaviour_counts.get(BehaviourClass.UNCERTAIN.value, 0)
    uncertain_ratio = round(uncertain_count / total_segments * 100)

    other_behaviours = [
        behaviour
        for behaviour in behaviour_counts
        if behaviour != dominant and behaviour != BehaviourClass.UNCERTAIN.value
    ]

    if other_behaviours:
        other_text = ", ".join(f"'{behaviour}'" for behaviour in other_behaviours)
        other_sentence = f"그 외 나온 행동은 {other_text}입니다."
    else:
        other_sentence = "그 외 나온 행동은 없습니다."

    warning_text = ""
    if uncertain_ratio > 20:
        warning_text = " uncertain 비율이 높아 분석 신뢰도가 낮을 수 있습니다."

    return (
        f"이번 세션에서 오리는 주로 '{dominant}' 상태였으며, "
        f"전체 구간의 약 {dominant_ratio}%를 차지했습니다. "
        f"{other_sentence} "
        f"uncertain 비율은 약 {uncertain_ratio}%입니다.{warning_text}"
    )


def generate_report(timeline, events, total_sec):
    summary = generate_summary_text(timeline, total_sec)
    return {
        "total_sec": total_sec,
        "total_segments": len(timeline),
        "timeline": timeline,
        "top_events": events[:5],
        "summary": summary,
    }
