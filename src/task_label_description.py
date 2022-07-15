korean_singular_conversation_labels = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']

korean_singular_conversation_label_descriptions = {
    key: key + '의 감정을 표현한 것'
    for key in korean_singular_conversation_labels
}

TASK_LABELS_DESC = {
    "ksc": korean_singular_conversation_label_descriptions,
}
