from enum import Enum

DEFAULT_CONFIDENCE_THRESHOLD = 0.45
class BehaviourClass(Enum):
    """
    Enum class for the behaviour state of target animal
    """
    RESTING = "resting"
    FEEDING = "feeding"
    EXPLORING = "exploring"
    UNCERTAIN = "uncertain"

BEHAVIOUR_MESSAGES: dict = {
    BehaviourClass.RESTING: [
        "[RESTING] 지금은 꽥꽥 알림 검지 검지! 오리가 조용히 체력을 회복하는 중이야...",
        "[RESTING] 오리가 깃털을 고르고 있다면 휴식과 몸 관리가 함께 일어나는 중일 수도 있대..",
        "[RESTING] 오리가 몸을 웅크리고 있네. 깃털 사이에 공기를 머금으면 보온에 도움이 되기도 한대!"
    ],
    BehaviourClass.FEEDING: [
        "[FEEDING] 부리를 물속에 콕콕! 오리는 물풀, 작은 곤충, 씨앗 같은 걸 먹을 수 있다고 하더라",
                             "[FEEDING] 오리 맛집 탐색 성공!",
                             "[FEEDING] 부리 필터 작동 중! 작은 먹이를 골라내고 있어."
    ],
    BehaviourClass.EXPLORING: [
        "[EXPLORING] 뒤뚱뒤뚱 정찰대 출동! 오리가 주변을 살피며 이동하고 있어",
        "[EXPLORING] 오리 탐험가 모드 발동! 물가, 풀숲, 얕은 물은 오리가 관심을 가질 만한 장소래 히히",
        "[EXPLORING] 오리가 방향을 자주 바꾸고 있다면 주변 단서를 모으는 중일 수 있어"
    ],
    BehaviourClass.UNCERTAIN: [
        "[UNCERTAIN] 음... 지금은 쉬는 건지 탐색하는 건지 애매하네.. 오리는 잠깐 멈춰서 주변을 살피기도 하거든!",
        "[UNCERTAIN] 지금은 애매한 오리 모드야",
        "[UNCERTAIN] 으음 잘 모르겠으니 버블검 공주를 불러와야할 것 같아... 버블검 공주!!"
    ]
}

# print(BehaviourClass.RESTING.value)  # "resting"
# print(BEHAVIOUR_MESSAGES[BehaviourClass.FEEDING])  # ["냠냠...", ...]
